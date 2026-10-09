"""vLLM serving + MCP-execution scoring harness for the gold test set.

Per candidate+tag: launch a vLLM OpenAI server (flags from train/configs/<key>.json,
optionally with a LoRA adapter), replay each gold item as an agent rollout
(chat.completions with tools -> execute emitted calls via ToolRegistry -> feed
results back, <= max-hops), and score:

  parse_ok          first rollout response yielded structured tool calls
                    (server parser OR harness <tool_call> fallback)
  tool_selection    strict = per-step ordered name sequence equals reference;
                      set    = per-step name set equals reference (both reported)
  arg_accuracy      per reference call, fraction of key args matching
                    (exact strings/enums/ids, numeric within 2%, geocoords <=500 m)
  execution_success all candidate calls executed ok
  grounded_answer   judge (JUDGE_MODELS[0]) final answer vs rubric, --judge-sample only
  refusal_ok        no_tool items: zero tool calls + direct answer
  hops_used, fallback_parse

Outputs eval/results/<candidate>_<tag>.json (per-item + aggregates) and appends a
summary row to eval/results/SUMMARY.md.

Usage (eval venv, NOT the train venv):
  eval/venv/bin/python -m eval.harness --candidate qwen3-4b-2507 --tag zeroshot \
      --gold data/test_gold_smoke.jsonl --gpu 0
  eval/venv/bin/python -m eval.harness --candidate qwen3-4b-2507 --tag ft \
      --adapter train/runs/qwen3-4b-2507 --gold data/test_gold.jsonl --gpu 1
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import pathlib
import random
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
sys.path.insert(0, str(ROOT))  # local ./mcp must shadow the PyPI `mcp` SDK (vllm dep)

import httpx  # noqa: E402

from src import config  # noqa: E402
from src.dataset_gen import _sys_assistant, _redact  # noqa: E402
from src.llm_client import chat_json  # noqa: E402
from src.render_templates import prepare_messages  # noqa: E402
from src.tool_registry import ToolRegistry  # noqa: E402

NAME = "harness"
EVAL_VENV = ROOT / "eval" / "venv"
CONFIGS_DIR = ROOT / "train" / "configs"
RUNS_DIR = ROOT / "train" / "runs"
RESULTS_DIR = ROOT / "eval" / "results"
LOGS_DIR = RESULTS_DIR / "logs"
MAX_TOOL_RESULT_CHARS = 3500
MAX_CALLS_PER_HOP = 8

# OOD (unseen-tools) eval: generic FC system instead of the in-car persona
_OOD_SYSTEM = (
    "You are a helpful assistant with access to a set of tools. "
    "When the user's request matches a tool, call it with correct arguments "
    "following the provided schemas. If no tool fits, answer directly."
)
_GEO_PROVIDERS = ("tomtom", "here", "keyless")

_openrouter_prefixes = ("qwen/", "z-ai/", "deepseek/", "google/", "mistralai/", "openai/gpt-oss")

_TOOL_CALL_RE = re.compile(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>")
_QWEN35_FN_RE = re.compile(
    r"(?s)<function=([^>]+)>\s*(.*?)\s*</function>")
_QWEN35_PARAM_RE = re.compile(r"(?s)<parameter=([^>]+)>\s*(.*?)\s*</parameter>\s*(?:\n|$)")


def log(msg: str) -> None:
    print(f"[{NAME}] {msg}", flush=True)


# ------------------------------------------------------------------ serving
def candidate_keys() -> list[str]:
    return sorted(p.stem for p in CONFIGS_DIR.glob("*.json"))


def load_cfg(key: str) -> dict:
    return json.loads((CONFIGS_DIR / f"{key}.json").read_text())


def resolve_adapter(key: str, spec: str) -> pathlib.Path | None:
    """spec 'auto' -> newest of the conventional run dirs; else explicit path."""
    if spec and spec != "auto":
        p = pathlib.Path(spec)
        if not p.is_absolute():
            p = ROOT / p
        return p if (p / "adapter_config.json").exists() else None
    for cand in (RUNS_DIR / key / "checkpoint-best", RUNS_DIR / key / "final", RUNS_DIR / key):
        if (cand / "adapter_config.json").exists():
            return cand
    return None


def gpu_used_mib(gpu: int) -> list[tuple[int, int]]:
    """[(pid, used_mib)] of compute apps on the given GPU index."""
    try:
        uuids = dict(
            line.split(", ")[:2]
            for line in subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                capture_output=True, text=True, check=True).stdout.splitlines())
        uuid = uuids.get(str(gpu), "")
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout
        apps = []
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 3 and parts[0] == uuid:
                apps.append((int(parts[1]), int(parts[2].replace(" MiB", ""))))
        return apps
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: nvidia-smi check failed ({e}); proceeding")
        return []


def assert_gpu_free(gpu: int) -> None:
    busy = [(pid, mib) for pid, mib in gpu_used_mib(gpu) if mib > 2048]
    if busy:
        raise SystemExit(f"GPU {gpu} busy: pids {busy} use >2 GiB; refusing to launch")


class VllmServer:
    """vLLM OpenAI server subprocess with health polling and hard cleanup."""

    def __init__(self, key: str, cfg: dict, adapter: pathlib.Path | None,
                 gpu: int, port: int, tag: str):
        self.key, self.cfg, self.adapter = key, cfg, adapter
        self.gpu, self.port, self.tag = gpu, port, tag
        self.proc: subprocess.Popen | None = None
        self.log_path = LOGS_DIR / f"serve_{key}_{tag}_{port}.log"
        self.serve_model = f"{key}-ft" if adapter else cfg["model_id"]

    def cmd(self) -> list[str]:
        # serve the BASE model always; the LoRA is attached via --lora-modules and
        # selected per-request through self.serve_model (= lora name when ft).
        cmd = [str(EVAL_VENV / "bin" / "vllm"), "serve", self.cfg["model_id"],
               "--port", str(self.port),
               "--dtype", self.cfg.get("dtype", "bfloat16"),
               "--max-model-len", "32768",
               "--gpu-memory-utilization", "0.90",
               "--tool-call-parser", self.cfg["parser"],
               "--enable-auto-tool-choice"]
        if self.cfg.get("reasoning_parser"):
            cmd += ["--reasoning-parser", self.cfg["reasoning_parser"]]
        if self.cfg.get("trust_remote_code"):
            cmd += ["--trust-remote-code"]
        if self.key == "qwen3.5-4b":
            cmd += ["--language-model-only"]
        if self.adapter:
            cmd += ["--enable-lora", "--max-lora-rank", "64",
                    "--lora-modules", f"{self.serve_model}={self.adapter}"]
        return cmd

    def start(self) -> None:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.gpu),
                   HF_HOME=str(ROOT / "hf_cache"),
                   VLLM_LOGGING_LEVEL="INFO",
                   PATH=str(EVAL_VENV / "bin") + os.pathsep + os.environ.get("PATH", ""))
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cmd = self.cmd()
        log(f"launching {self.key}/{self.tag} on gpu{self.gpu} port{self.port}: "
            f"{' '.join(cmd)}")
        with open(self.log_path, "w") as lf:
            self.proc = subprocess.Popen(cmd, stdout=lf, stderr=lf,
                                         start_new_session=True, env=env)
        atexit.register(self.kill)
        t0 = time.time()
        while time.time() - t0 < 600:
            if self.proc.poll() is not None:
                self.dump_tail(f"server exited early rc={self.proc.returncode}")
                raise SystemExit(f"vLLM server for {self.key}/{self.tag} died at startup")
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        log(f"server healthy after {time.time() - t0:.0f}s "
                            f"(model={self.serve_model})")
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(3)
        self.dump_tail("health timeout")
        raise SystemExit(f"vLLM server for {self.key}/{self.tag} not healthy after 600s")

    def dump_tail(self, why: str, lines: int = 40) -> None:
        log(f"SERVER FAILURE ({why}); tail of {self.log_path}:")
        try:
            print("\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:]))
        except OSError:
            pass

    def grep_log(self, pattern: str) -> list[str]:
        try:
            pat = re.compile(pattern)
            return [ln for ln in self.log_path.read_text(errors="replace").splitlines()
                    if pat.search(ln)]
        except OSError:
            return []

    def kill(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        for _ in range(20):
            if self.proc.poll() is not None:
                break
            time.sleep(0.5)
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self.proc.wait()
        log(f"server {self.key}/{self.tag} stopped")


# ------------------------------------------------------------------ rollout
def parse_fallback_tool_calls(content: str) -> list[dict]:
    """Harness-side <tool_call> parse (gemma hermes-transplant, qwen3.5 XML form)."""
    calls: list[dict] = []
    for m in _TOOL_CALL_RE.finditer(content or ""):
        blob = m.group(1).strip()
        obj = None
        try:
            obj = json.loads(blob)
        except Exception:
            try:  # truncated/extra prose: raw_decode the first balanced object
                obj, _ = json.JSONDecoder().raw_decode(blob[blob.index("{"):])
            except Exception:
                obj = None
        if isinstance(obj, dict) and obj.get("name"):
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except Exception:
                    args = {}
            calls.append({"name": obj["name"], "arguments": args or {}})
    if not calls:  # qwen3.5 XML-ish form (<function=name><parameter=..>)
        for m in _QWEN35_FN_RE.finditer(content or ""):
            args: dict = {}
            for pm in _QWEN35_PARAM_RE.finditer(m.group(2)):
                try:
                    args[pm.group(1)] = json.loads(pm.group(2))
                except Exception:
                    args[pm.group(1)] = pm.group(2)
            if m.group(1):
                calls.append({"name": m.group(1), "arguments": args})
    if not calls:  # phi4_mini convention: functools[{"name":..,"arguments":{..}}, ..]
        # (vLLM's phi4_mini_json never matched our training convention -> parse
        # server-side failed silently; this mirrors src/render_templates.py:426)
        text = content or ""
        idx = 0
        while True:
            i = text.find("functools[", idx)
            if i < 0:
                break
            depth, end = 0, -1
            for k in range(i + len("functools"), len(text)):
                if text[k] == "[":
                    depth += 1
                elif text[k] == "]":
                    depth -= 1
                    if depth == 0:
                        end = k
                        break
            if end < 0:
                break
            try:
                for obj in json.loads(text[i + len("functools"): end + 1]):
                    if isinstance(obj, dict) and obj.get("name"):
                        args = obj.get("arguments", obj.get("parameters", {}))
                        if isinstance(args, str):
                            try:
                                args = json.loads(args or "{}")
                            except Exception:
                                args = {}
                        calls.append({"name": obj["name"], "arguments": args or {}})
            except Exception:
                pass  # malformed body (curly quotes, bare names) stays unparsed
            idx = end + 1
    return calls[:MAX_CALLS_PER_HOP]


def _view_messages(fmt: str, messages: list[dict], tools: list[dict]) -> list[dict]:
    # gemma: prepare_messages injects HERMES tool instructions into the first user
    # turn (template itself takes no tools); other fmts pass tools to the template.
    view, _ = prepare_messages(fmt, messages, tools)
    # transport normalization: the API body requires tool_calls arguments as a
    # JSON *string* (prepare_messages may produce dicts for template rendering).
    for m in view:
        for tc in m.get("tool_calls") or []:
            a = tc.get("function", {}).get("arguments")
            if isinstance(a, dict):
                tc["function"]["arguments"] = json.dumps(a, ensure_ascii=False)
    return view


class Rollout:
    def __init__(self, client: httpx.Client, serve_model: str, fmt: str,
                 port: int, reg: ToolRegistry, max_hops: int, key: str,
                 ood: bool = False):
        self.client, self.serve_model, self.fmt = client, serve_model, fmt
        self.url = f"http://127.0.0.1:{port}/v1/chat/completions"
        self.reg, self.max_hops, self.key = reg, max_hops, key
        self.ood = ood

    def _chat(self, messages: list[dict], tools: list[dict]) -> dict:
        body: dict = {"model": self.serve_model, "messages": _view_messages(
            self.fmt, messages, tools),
            "temperature": 0.0, "max_tokens": 1024}
        # phi4_mini: tools are already folded into the system message by the
        # renderer (training format). ALSO passing tools= makes the server's
        # chat template inject a second, differently-formatted tool block ->
        # model goes off-distribution and answers in prose (0 calls ever).
        if self.fmt != "phi4_mini":
            body["tools"] = tools
        if self.key == "qwen3.5-4b":  # SERVING MUST MATCH training render
            body["chat_template_kwargs"] = {"enable_thinking": False}
        last = None
        for _ in range(3):
            try:
                r = self.client.post(self.url, json=body, timeout=300.0)
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]
                if r.status_code == 400 and "maximum context length" in r.text \
                        and body["max_tokens"] > 256:
                    body["max_tokens"] = 256  # long-rollout item: squeeze output budget
                    continue
                last = RuntimeError(f"{r.status_code}: {r.text[:200]}")
                if r.status_code < 500 and r.status_code != 429:
                    break
            except httpx.HTTPError as e:
                last = e
            time.sleep(2)
        raise RuntimeError(f"chat completion failed: {last}")

    def run(self, item: dict) -> dict:
        lang, mode = item["language"], item["mode"]
        tools = item["tools"]
        system = _OOD_SYSTEM if self.ood else _sys_assistant(lang, "any", mode)
        messages: list[dict] = [{"role": "system", "content": system}]
        rec: dict = {
            "gold_id": item["gold_id"], "language": lang, "mode": mode,
            "parse_ok": False, "fallback_parse": 0, "hops_used": 0,
            "answers": [], "calls": [],  # calls: {step,name,args,ok,provider,fallback}
        }
        for turn_idx, user_text in enumerate(item["user_turns"], start=1):
            messages.append({"role": "user", "content": user_text})
            answered = False
            for _hop in range(self.max_hops):
                msg = self._chat(messages, tools)
                content = msg.get("content") or ""
                tc = msg.get("tool_calls") or []
                source = "server"
                if not tc:
                    fb = parse_fallback_tool_calls(content)
                    if fb:
                        tc = [{"function": {"name": c["name"],
                                            "arguments": json.dumps(c["arguments"],
                                                                    ensure_ascii=False)}}
                              for c in fb]
                        source = "fallback"
                        rec["fallback_parse"] += 1
                if not tc:
                    messages.append({"role": "assistant", "content": content})
                    rec["answers"].append(content.strip())
                    answered = True
                    break
                rec["parse_ok"] = True
                rec["hops_used"] += 1
                norm_calls = []
                for i, t in enumerate(tc[:MAX_CALLS_PER_HOP]):
                    raw_args = t["function"]["arguments"]
                    args_str = (json.dumps(raw_args, ensure_ascii=False)
                                if isinstance(raw_args, dict) else (raw_args or "{}"))
                    norm_calls.append({"id": f"c{turn_idx}_{i}", "type": "function",
                                       "function": {"name": t["function"]["name"],
                                                    "arguments": args_str}})
                messages.append({"role": "assistant", "content": content,
                                 "tool_calls": norm_calls})
                for t in norm_calls:
                    fname = t["function"]["name"]
                    try:
                        args = json.loads(t["function"]["arguments"])
                    except Exception:
                        args = {}
                    if self.ood:
                        # foreign tools: no executor — stub the result; provider
                        # stays a real geo provider only if the name collides
                        out = {"ok": True, "result": {"_ood_stub": True}}
                    else:
                        out = self.reg.execute(fname, args)
                    provider = (self.reg.providers.get(fname, "ood")
                                if self.ood
                                else self.reg.providers.get(fname, "other"))
                    rec["calls"].append({"step": turn_idx, "name": fname, "args": args,
                                         "ok": bool(out["ok"]), "provider": provider,
                                         "fallback": source == "fallback"})
                    payload = _redact(json.dumps(out["result"], ensure_ascii=False)
                                      if out["ok"]
                                      else json.dumps({"error": out["error"]},
                                                      ensure_ascii=False))[:MAX_TOOL_RESULT_CHARS]
                    messages.append({"role": "tool", "tool_call_id": t["id"],
                                     "content": payload})
                if self.ood:
                    return rec  # single emission is all the OOD eval scores
            if not answered:
                rec["answers"].append("")
        return rec


# ------------------------------------------------------------------ scoring
def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    import math
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


_LAT_KEYS = {"lat", "latitude"}
_LON_KEYS = {"lon", "lng", "longitude"}
_COORD_TOL_M = 500.0


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _as_coord(v):
    """(lat, lon) tuple from a "lat,lon" string / [lat,lon] / {lat,lon}, else None."""
    if isinstance(v, str):
        parts = v.strip().split(",")
        if len(parts) == 2:
            try:
                return (float(parts[0]), float(parts[1]))
            except ValueError:
                return None
        return None
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(_is_num(x) for x in v):
        return (float(v[0]), float(v[1]))
    if isinstance(v, dict) and _is_num(v.get("lat")) and _is_num(v.get("lon", v.get("lng"))):
        return (float(v["lat"]), float(v.get("lon", v.get("lng"))))
    return None


_POSITIVE_KEYS = ("position", "center", "origin", "destination", "coordinate", "at",
                  "waypoint", "location")


def _extract_coords(d: dict) -> tuple[list[tuple], dict]:
    """Pull (lat,lon) pairs out of an args dict; returns (pairs, remaining args)."""
    pairs, rest = [], dict(d)
    used = set()
    for k, v in d.items():
        lk = k.lower()
        if lk in _LAT_KEYS and _is_num(v):
            for lk2 in _LON_KEYS:
                if lk2 in d and _is_num(d[lk2]):
                    pairs.append((float(v), float(d[lk2])))
                    used.update((k, lk2))
                    break
        elif isinstance(v, dict) and _is_num(v.get("lat")) and _is_num(v.get("lon", v.get("lng"))):
            pairs.append((float(v["lat"]), float(v.get("lon", v.get("lng")))))
            used.add(k)
        elif (isinstance(v, (list, tuple)) and len(v) == 2
              and all(_is_num(x) for x in v) and lk in _POSITIVE_KEYS):
            pairs.append((float(v[0]), float(v[1])))
            used.add(k)
        elif lk in _POSITIVE_KEYS and isinstance(v, str) and _as_coord(v):
            pairs.append(_as_coord(v))
            used.add(k)
    for k in used:
        rest.pop(k, None)
    return pairs, rest


def _match_value(rv, cv) -> bool:
    rc, cc = _as_coord(rv), _as_coord(cv)
    if rc and cc:
        return _haversine_m(rc, cc) <= _COORD_TOL_M
    if _is_num(rv) and _is_num(cv):
        return abs(float(rv) - float(cv)) <= max(1e-6, 0.02 * abs(float(rv)))
    if isinstance(rv, bool) or isinstance(cv, bool):
        return rv == cv
    if isinstance(rv, dict) and isinstance(cv, dict):
        rp, rrest = _extract_coords(rv)
        cp, crest = _extract_coords(cv)
        if len(rp) != len(cp) or not all(
                _haversine_m(a, b) <= _COORD_TOL_M for a, b in zip(rp, cp)):
            return False
        return set(rrest) == set(crest) and all(_match_value(rrest[k], crest[k]) for k in rrest)
    if isinstance(rv, (list, tuple)) and isinstance(cv, (list, tuple)):
        if len(rv) != len(cv):
            return False
        if len(rv) == 2 and all(_is_num(x) for x in rv) and all(_is_num(x) for x in cv):
            return _haversine_m((rv[0], rv[1]), (cv[0], cv[1])) <= _COORD_TOL_M
        return all(_match_value(a, b) for a, b in zip(rv, cv))
    if isinstance(rv, str) and isinstance(cv, str):
        return rv.strip() == cv.strip() or rv.strip().lower() == cv.strip().lower()
    return rv == cv


def provider_of(name: str) -> str:
    if name.startswith("tomtom-"):
        return "tomtom"
    if name.startswith("here_"):
        return "here"
    if name.startswith(("osrm_", "nominatim_", "open_meteo_")):
        return "keyless"
    return "other"


def _arg_match_frac(ref_args: dict, cand_args: dict | None) -> float:
    if cand_args is None:
        return 0.0
    rp, rrest = _extract_coords(ref_args)
    cp, crest = _extract_coords(cand_args)
    total, matched = 0, 0
    for i, a in enumerate(rp):
        total += 1
        if i < len(cp) and _haversine_m(a, cp[i]) <= _COORD_TOL_M:
            matched += 1
    for k, rv in rrest.items():
        total += 1
        if k in crest and _match_value(rv, crest[k]):
            matched += 1
    return matched / total if total else 1.0


def score_item(item: dict, rec: dict, do_judge: bool) -> dict:
    refs = item["reference"]["tool_calls"]
    calls = rec["calls"]
    out = dict(rec)
    out["n_ref_calls"] = len(refs)

    # tool_selection: per-step name sequences (reference vs candidate)
    ref_by_step: dict[int, list[str]] = {}
    for c in refs:
        ref_by_step.setdefault(c["step"], []).append(c["name"])
    cand_by_step: dict[int, list[str]] = {}
    for c in calls:
        cand_by_step.setdefault(c["step"], []).append(c["name"])
    steps = sorted(set(ref_by_step) | set(cand_by_step))
    out["tool_selection_strict"] = bool(ref_by_step) and all(
        ref_by_step.get(s, []) == cand_by_step.get(s, []) for s in steps)
    out["tool_selection_set"] = bool(ref_by_step) and all(
        set(ref_by_step.get(s, [])) == set(cand_by_step.get(s, [])) for s in steps)

    # arg_accuracy + per-provider pairing: greedy pair ref/cand calls by (step, name)
    cand_pool: dict[tuple, list[dict]] = {}
    for c in calls:
        cand_pool.setdefault((c["step"], c["name"]), []).append(c)
    paireds = []
    for rc in refs:
        pool = cand_pool.get((rc["step"], rc["name"]))
        cc = pool.pop(0) if pool else None
        paireds.append((rc, cc))
    fracs = [_arg_match_frac(rc["args"], cc["args"] if cc else None) for rc, cc in paireds]
    out["arg_accuracy"] = sum(fracs) / len(fracs) if fracs else None

    out["execution_success"] = bool(calls) and all(c["ok"] for c in calls)
    out["exec_ok_calls"] = sum(1 for c in calls if c["ok"])
    out["n_calls"] = len(calls)

    if item["mode"] == "no_tool":
        out["refusal_ok"] = (not calls) and all(a.strip() for a in rec["answers"])
        out["arg_accuracy"] = None
        out["tool_selection_strict"] = out["tool_selection_set"] = None
        out["execution_success"] = None
    elif item.get("ood") or item["mode"] == "ood":
        # unseen tools: no executor -> exec metrics meaningless; score selection,
        # args and hallucination rates instead
        out["refusal_ok"] = None
        out["execution_success"] = None
        out["exec_ok_calls"] = None
        out["per_provider"] = {}
        item_tool_names = {t["function"]["name"] for t in item["tools"]}
        n_calls = len(calls)
        n_bad = sum(1 for c in calls if c["name"] not in item_tool_names)
        n_geo = sum(1 for c in calls if c["provider"] in _GEO_PROVIDERS)
        out["ood_hallucination_rate"] = (round(n_bad / n_calls, 4)
                                         if n_calls else None)
        out["geo_hallucination_rate"] = (round(n_geo / n_calls, 4)
                                         if n_calls else None)
    else:
        out["refusal_ok"] = None

    # per-provider
    prov: dict[str, dict] = {}
    for rc, cc in paireds:
        p = provider_of(rc["name"])
        d = prov.setdefault(p, {"ref_calls": 0, "arg_match": 0.0, "cand_calls": 0,
                                "cand_ok": 0})
        d["ref_calls"] += 1
        d["arg_match"] += _arg_match_frac(rc["args"], cc["args"] if cc else None)
    for c in calls:
        d = prov.setdefault(provider_of(c["name"]),
                            {"ref_calls": 0, "arg_match": 0.0, "cand_calls": 0, "cand_ok": 0})
        d["cand_calls"] += 1
        d["cand_ok"] += int(c["ok"])
    out["per_provider"] = {
        p: {"ref_calls": d["ref_calls"],
            "arg_accuracy": round(d["arg_match"] / d["ref_calls"], 4) if d["ref_calls"] else None,
            "cand_calls": d["cand_calls"],
            "exec_ok_rate": round(d["cand_ok"] / d["cand_calls"], 4) if d["cand_calls"] else None}
        for p, d in prov.items()}

    out["grounded"] = None
    if do_judge and item["mode"] == "tool_use" and rec["answers"]:
        ans = rec["answers"][-1]
        if ans.strip():
            out["grounded"] = judge_answer(item, ans)
    return out


def judge_answer(item: dict, answer: str) -> dict | None:
    model = config.JUDGE_MODELS[0]
    prompt = (
        "You grade a navigation assistant's final answer against a rubric.\n"
        f"Rubric: {item['reference']['rubric']}\n"
        f"User request: {' | '.join(item['user_turns'])[:600]}\n"
        f"Assistant answer: {answer[:1500]}\n\n"
        'Output ONLY JSON: {"score": <0-10 number>, "reason": "<one sentence>"} '
        "Score 7+ if all key facts from the rubric are present and correct "
        "(language of the answer should match the user's language)."
    )
    kw: dict = {"temperature": 0.2, "max_tokens": 300}
    if model.startswith(_openrouter_prefixes):
        kw["extra_payload"] = {"reasoning": {"enabled": False}}
    try:
        blob = chat_json(model, [{"role": "user", "content": prompt}], **kw)
    except Exception:  # noqa: BLE001
        return None
    try:
        score = float(blob["score"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"score": round(max(0.0, min(10.0, score)), 2),
            "pass": bool(score >= 7.0),
            "judge": model,
            "reason": str(blob.get("reason", ""))[:200]}


def aggregate(rows: list[dict], grouped: bool = True) -> dict:
    n = len(rows)
    tool_rows = [r for r in rows if r["mode"] in ("tool_use", "ood")]
    no_rows = [r for r in rows if r["mode"] == "no_tool"]
    ood_rows = [r for r in rows if r["mode"] == "ood"]
    judged = [r for r in rows if r.get("grounded")]

    def rate(rows_, key):
        vals = [r[key] for r in rows_ if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    agg: dict = {
        "n": n,
        "parse_ok_rate": rate(rows, "parse_ok"),
        "tool_selection_strict": rate(tool_rows, "tool_selection_strict"),
        "tool_selection_set": rate(tool_rows, "tool_selection_set"),
        "arg_accuracy": rate(tool_rows, "arg_accuracy"),
        "execution_success_rate": rate(tool_rows, "execution_success"),
        "ood_hallucination_rate": rate(ood_rows, "ood_hallucination_rate"),
        "geo_hallucination_rate": rate(ood_rows, "geo_hallucination_rate"),
        "n_ood": len(ood_rows),
        "fallback_parse_rate": (round(sum(1 for r in rows if r["fallback_parse"] > 0) / n, 4)
                                if n else None),
        "avg_hops": rate(rows, "hops_used"),
        "refusal_ok_rate": rate(no_rows, "refusal_ok"),
        "grounded_rate": (round(sum(1 for r in judged if r["grounded"]["pass"]) / len(judged), 4)
                          if judged else None),
        "n_judged": len(judged),
    }
    agg["by_provider"] = {}
    for p in sorted({p for r in rows for p in r.get("per_provider", {})}):
        agg["by_provider"][p] = merge_providers(
            [r["per_provider"][p] for r in rows if p in r.get("per_provider", {})])
    if grouped:
        agg["by_language"] = {}
        for lang in sorted({r["language"] for r in rows}):
            sub = [r for r in rows if r["language"] == lang]
            agg["by_language"][lang] = aggregate(sub, grouped=False)
    return agg


def merge_providers(entries: list[dict]) -> dict:
    ref_calls = sum(e["ref_calls"] for e in entries)
    arg = [e["arg_accuracy"] for e in entries if e["arg_accuracy"] is not None]
    cand_calls = sum(e["cand_calls"] for e in entries)
    ok = [e["exec_ok_rate"] for e in entries if e["exec_ok_rate"] is not None]
    return {"ref_calls": ref_calls,
            "arg_accuracy": round(sum(arg) / len(arg), 4) if arg else None,
            "cand_calls": cand_calls,
            "exec_ok_rate": round(sum(ok) / len(ok), 4) if ok else None}


# ------------------------------------------------------------------ driver
def run_one(key: str, tag: str, args, gold: list[dict]) -> dict:
    cfg = load_cfg(key)
    adapter = None
    if tag == "ft":
        adapter = resolve_adapter(key, args.adapter)
        if adapter is None:
            log(f"{key}: no LoRA adapter found for tag=ft (auto search under {RUNS_DIR}/{key}); "
                f"SKIPPING")
            return {"skipped": f"no adapter for {key}"}
    port = args.port if args.port >= 0 else 8120 + candidate_keys().index(key)
    assert_gpu_free(args.gpu)
    server = VllmServer(key, cfg, adapter, args.gpu, port, tag)
    server.start()
    t0 = time.time()
    try:
        items = gold[: args.limit] if args.limit else gold
        judge_set = set()
        if args.judge_sample:
            idxs = list(range(len(items)))
            rng = random.Random(0)
            rng.shuffle(idxs)
            judge_set = {items[i]["gold_id"] for i in idxs[: args.judge_sample]}
        reg = ToolRegistry()
        try:
            with httpx.Client() as client:
                roll = Rollout(client, server.serve_model, cfg["fmt"], port, reg,
                               args.max_hops, key, ood=args.ood)
                rows: list[dict] = []

                def job(item: dict) -> dict:
                    rec = roll.run(item)
                    do_judge = (item["gold_id"] in judge_set
                                and item["mode"] == "tool_use")
                    return score_item(item, rec, do_judge)

                with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                    futs = [ex.submit(job, it) for it in items]
                    for i, fut in enumerate(as_completed(futs)):
                        try:
                            rows.append(fut.result())
                        except Exception as e:  # noqa: BLE001
                            log(f"item failed: {type(e).__name__}: {str(e)[:200]}")
                        if (i + 1) % 5 == 0 or i + 1 == len(items):
                            log(f"{key}/{tag}: {i + 1}/{len(items)} items scored")
                rows.sort(key=lambda r: r["gold_id"])
        finally:
            reg.close()
        result = {
            "meta": {
                "candidate": key, "tag": tag, "adapter": str(adapter) if adapter else None,
                "serve_model": server.serve_model, "gpu": args.gpu, "port": port,
                "gold": args.gold, "n_items": len(rows), "max_hops": args.max_hops,
                "ood": bool(args.ood),
                "runtime_s": round(time.time() - t0, 1),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "vllm": vllm_version(),
                "adapter_loaded_log_lines": server.grep_log(
                    r"Loaded PEFT|LoRA adapter|adapter.*loaded")[:5],
            },
            "items": rows,
            "aggregate": aggregate(rows),
        }
        out_path = RESULTS_DIR / f"{key}_{tag}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
        append_summary(key, tag, adapter, result)
        log(f"{key}/{tag}: wrote {out_path}")
        log(f"{key}/{tag} AGGREGATE: {json.dumps(result['aggregate'] | {'by_language': '...'}, default=str)}")
        return result
    finally:
        server.kill()
        atexit.unregister(server.kill)


def vllm_version() -> str:
    try:
        out = subprocess.run([str(EVAL_VENV / "bin" / "vllm"), "--version"],
                             capture_output=True, text=True, timeout=120)
        return (out.stdout or out.stderr).strip().splitlines()[-1]
    except Exception:  # noqa: BLE001
        return "?"


def append_summary(key: str, tag: str, adapter: pathlib.Path | None, result: dict) -> None:
    a = result["aggregate"]
    path = RESULTS_DIR / "SUMMARY.md"
    if result["meta"].get("ood"):
        marker = "## OOD generalization (unseen tools)"
        if not path.exists() or marker not in path.read_text(encoding="utf-8"):
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"\n{marker}\n\n"
                        "| ts | candidate | tag | n | sel_strict | sel_set | arg_acc | "
                        "ood_hall | geo_hall | fallback |\n"
                        "|---|---|---|---|---|---|---|---|---|---|\n")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"| {time.strftime('%m-%d %H:%M')} | {key} | {tag} "
                    f"{'(' + adapter.name + ')' if adapter else ''} | {a['n']} "
                    f"| {a['tool_selection_strict']} | {a['tool_selection_set']} "
                    f"| {a['arg_accuracy']} | {a['ood_hallucination_rate']} "
                    f"| {a['geo_hallucination_rate']} | {a['fallback_parse_rate']} |\n")
        return
    if not path.exists():
        path.write_text(
            "# Eval summary\n\n"
            "| ts | candidate | tag | n | parse_ok | sel_strict | sel_set | arg_acc | "
            "exec_ok | fallback | refusal_ok | grounded | avg_hops |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"| {time.strftime('%m-%d %H:%M')} | {key} | {tag} "
                f"{'(' + adapter.name + ')' if adapter else ''} | {a['n']} "
                f"| {a['parse_ok_rate']} | {a['tool_selection_strict']} "
                f"| {a['tool_selection_set']} | {a['arg_accuracy']} "
                f"| {a['execution_success_rate']} | {a['fallback_parse_rate']} "
                f"| {a['refusal_ok_rate']} | {a['grounded_rate']} | {a['avg_hops']} |\n")


def load_gold(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main() -> int:
    global RESULTS_DIR, LOGS_DIR  # noqa: PLW0603
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", choices=candidate_keys())
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--adapter", default="auto",
                    help="LoRA adapter dir for tag=ft (path or 'auto')")
    ap.add_argument("--gold", default=str(ROOT / "data" / "test_gold_smoke.jsonl"))
    ap.add_argument("--gpu", type=int, default=0, choices=[0, 1])
    ap.add_argument("--port", type=int, default=-1,
                    help="override port (default 8120+candidate index)")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    ap.add_argument("--tag", default="zeroshot",
                    choices=["zeroshot", "ft", "both"])
    ap.add_argument("--max-hops", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="cap items (0=all)")
    ap.add_argument("--judge-sample", type=int, default=0,
                    help="rubric-judge N items with JUDGE_MODELS[0]")
    ap.add_argument("--ood", action="store_true",
                    help="OOD mode: unseen-tool gold (e.g. data/test_gold_ood.jsonl); "
                         "generic FC system, stubbed execution, hallucination rates")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    keys = candidate_keys() if args.all else ([args.candidate] if args.candidate else [])
    if not keys:
        ap.error("--candidate KEY or --all required")
    tags = ["zeroshot", "ft"] if args.tag == "both" else [args.tag]
    RESULTS_DIR = pathlib.Path(args.out_dir)
    LOGS_DIR = RESULTS_DIR / "logs"
    gold = load_gold(args.gold)
    log(f"gold: {len(gold)} items from {args.gold}; candidates={keys} tags={tags}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for key in keys:
        for tag in tags:
            try:
                res = run_one(key, tag, args, gold)
                if res.get("skipped"):
                    failures += 1
            except SystemExit as e:
                log(f"{key}/{tag} aborted: {e}")
                failures += 1
    return 1 if failures and failures == len(keys) * len(tags) else 0


if __name__ == "__main__":
    sys.exit(main())
