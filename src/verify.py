"""APIGen-style verification stage for synthetic function-calling dialogues.

Stages (all on by default, each skippable via CLI flag):
  structural  - message/role/tool_call pairing sanity
  schema      - tool_call arguments vs registry JSON schema (hand-rolled subset)
  exec        - replay recorded tool calls through the live registry (cache-aware)
  secrets     - scan for apikey/token leaks (reject + scrub in output copy)
  language    - stopword heuristic + one-judge confirmation of flagged dialogs
  judge       - dual-LLM quality scores (alignment/grounding/naturalness/language)
  mmr         - 5-gram Jaccard dedup (greedy, threshold 0.6)

Writes one verified jsonl (including failed rows with verdicts), a summary json,
and a markdown report. Raw input files are never modified.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src import config
from src import tool_registry as tr
from src.llm_client import LLMError, chat_json
from src.tool_registry import ToolRegistry

NAME = "verify"
STAGE_ORDER = ["structural", "schema", "exec", "secrets", "language", "judge", "mmr"]
VALID_ROLES = {"system", "user", "assistant", "tool"}
JUDGE_DIMS = ["alignment", "grounding", "naturalness", "language"]
LANG_NAME = {"en": "English", "de": "German", "pl": "Polish", "fr": "French", "es": "Spanish"}

_print_lock = threading.Lock()
_judge_down: set[str] = set()
_judge_lock = threading.Lock()
_judge_streak: Counter = Counter()


def log(msg: str) -> None:
    with _print_lock:
        print(f"[{NAME}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- structural
def check_structural(row: dict, known_tools: set[str]) -> tuple[list[str], list[dict]]:
    """Returns (failure reasons, parsed tool_calls as [{id, name, args_json}])."""
    reasons: list[str] = []
    calls: list[dict] = []
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return ["messages missing or not a list"], calls
    if msgs[0].get("role") != "system":
        reasons.append("first message is not system")
    call_ids: dict[str, int] = {}
    tool_resp_ids: dict[str, int] = {}
    n_users = 0
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role not in VALID_ROLES:
            reasons.append(f"msg[{i}] invalid role {role!r}")
            continue
        content = m.get("content")
        if role in ("user", "system", "tool") and not (isinstance(content, str) and content.strip()):
            reasons.append(f"msg[{i}] {role} content empty")
        if role == "assistant":
            if content is not None and not isinstance(content, str):
                reasons.append(f"msg[{i}] assistant content not a string")
            tcs = m.get("tool_calls")
            if tcs:
                if not isinstance(tcs, list):
                    reasons.append(f"msg[{i}] tool_calls not a list")
                    continue
                for tc in tcs:
                    if not isinstance(tc, dict):
                        reasons.append(f"msg[{i}] tool_call not an object")
                        continue
                    tc_id = tc.get("id")
                    fn = tc.get("function") or {}
                    name = fn.get("name")
                    args_json = fn.get("arguments")
                    if not isinstance(tc_id, str) or not tc_id:
                        reasons.append(f"msg[{i}] tool_call missing id")
                        tc_id = None
                    if tc.get("type") != "function":
                        reasons.append(f"msg[{i}] tool_call type != function")
                    if name not in known_tools:
                        reasons.append(f"msg[{i}] tool_call name not in registry: {name!r}")
                    if isinstance(args_json, dict):
                        args_json = json.dumps(args_json, ensure_ascii=False)
                    if not isinstance(args_json, str):
                        reasons.append(f"msg[{i}] tool_call arguments not a JSON string")
                    else:
                        try:
                            if not isinstance(json.loads(args_json), dict):
                                reasons.append(f"msg[{i}] tool_call arguments not a JSON object")
                        except Exception:
                            reasons.append(f"msg[{i}] tool_call arguments invalid JSON")
                    if tc_id:
                        if tc_id in call_ids:
                            reasons.append(f"msg[{i}] duplicate tool_call id {tc_id}")
                        call_ids[tc_id] = i
                        calls.append({"id": tc_id, "name": name, "args_json": args_json})
        elif role == "tool":
            tc_id = m.get("tool_call_id")
            if not isinstance(tc_id, str) or not tc_id:
                reasons.append(f"msg[{i}] tool message missing tool_call_id")
            else:
                tool_resp_ids[tc_id] = i
        elif role == "user":
            n_users += 1
    for tc_id, mi in call_ids.items():
        if tc_id not in tool_resp_ids:
            reasons.append(f"tool_call {tc_id} (msg[{mi}]) has no tool response")
    for tc_id, mi in tool_resp_ids.items():
        if tc_id not in call_ids:
            reasons.append(f"tool response (msg[{mi}]) matches no tool_call id")
    if n_users == 0:
        reasons.append("no user turns")
    last = msgs[-1]
    if last.get("role") != "assistant" or not (last.get("content") or "").strip():
        reasons.append("does not end with non-empty assistant text")
    return reasons, calls


# ---------------------------------------------------------------- schema
def _type_ok(val: Any, typ: str) -> bool:
    if typ == "string":
        return isinstance(val, str)
    if typ == "number":
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    if typ == "integer":
        return (isinstance(val, int) and not isinstance(val, bool)) or (
            isinstance(val, float) and val.is_integer())
    if typ == "boolean":
        return isinstance(val, bool)
    if typ == "array":
        return isinstance(val, list)
    if typ == "object":
        return isinstance(val, dict)
    if typ == "null":
        return val is None
    return True  # unknown type keyword: lenient


def _check_value(val: Any, sch: dict, path: str, reasons: list[str]) -> None:
    typ = sch.get("type")
    if isinstance(typ, list):
        if not any(_type_ok(val, t) for t in typ):
            reasons.append(f"{path}: type {type(val).__name__} not in {typ}")
            return
    elif typ and not _type_ok(val, typ):
        reasons.append(f"{path}: expected {typ}, got {type(val).__name__}")
        return
    if "enum" in sch and val not in sch["enum"]:
        reasons.append(f"{path}: {str(val)[:60]!r} not in enum {sch['enum']}")
    items = sch.get("items")
    if isinstance(val, list) and isinstance(items, dict):
        for j, el in enumerate(val):
            _check_value(el, items, f"{path}[{j}]", reasons)
    props = sch.get("properties")
    if isinstance(val, dict) and isinstance(props, dict):
        for req in sch.get("required", []):
            if req not in val:
                reasons.append(f"{path}: missing required '{req}'")
        for k, v in val.items():
            if k in props:
                _check_value(v, props[k], f"{path}.{k}", reasons)


def check_schema(call: dict, schema: dict | None, reasons: list[str], notes: list[str]) -> None:
    """Validates the registry-normalized args (what actually hit the API) against the schema."""
    if schema is None or call["args_json"] is None:
        return  # unknown tool / invalid JSON already flagged by structural
    try:
        args = json.loads(call["args_json"])
    except Exception:
        return
    norm = tr._normalize_args(call["name"], args)
    label = f"call {call['id'][:12]} {call['name']}"
    pre = len(reasons)
    _check_value(norm, schema.get("parameters") or {}, label, reasons)
    extra = sorted(set(norm) - set((schema.get("parameters") or {}).get("properties", {})))
    if extra:
        notes.append(f"{label}: extra params {extra} (ignored, registry-normalized)")


# ---------------------------------------------------------------- secrets
_SECRET_PATTERNS = [
    re.compile(r"\b(api[-_]?key|access[-_]?token|subscription[-_]?key|token|key)"
               r"\s*=\s*([A-Za-z0-9_.\-]{8,})", re.IGNORECASE),
    re.compile(r'["\'](api[-_]?key|access[-_]?token|subscription[-_]?key|token|key)["\']'
               r'\s*:\s*["\']([A-Za-z0-9_.\-]{8,})["\']', re.IGNORECASE),
]


def _scrub_str(t: str, hits: list[int]) -> str:
    out = t
    for pat in _SECRET_PATTERNS:
        hits[0] += len(pat.findall(out))
        out = pat.sub(lambda m: f"{m.group(1)}=***", out)
    return out


def _scrub_any(obj: Any, hits: list[int]) -> tuple[Any, bool]:
    """Recursively scrub secrets in dicts/lists/strings. Returns (new_obj, changed)."""
    if isinstance(obj, str):
        s = _scrub_str(obj, hits)
        return s, s != obj
    if isinstance(obj, dict):
        changed = False
        out = {}
        for k, v in obj.items():
            nv, ch = _scrub_any(v, hits)
            out[k], changed = nv, changed or ch
        return out, changed
    if isinstance(obj, list):
        changed = False
        out = []
        for v in obj:
            nv, ch = _scrub_any(v, hits)
            out.append(nv)
            changed = changed or ch
        return out, changed
    return obj, False


def scan_secrets(row: dict) -> tuple[list[str], dict]:
    """Scan messages + exec_log. Returns (failure reasons, {field: scrubbed copy})."""
    hits: list[int] = [0]
    cleaned: dict[str, Any] = {}

    # messages: content strings + tool_call argument strings (structure-preserving)
    msgs = row.get("messages")
    if isinstance(msgs, list):
        new_msgs: list[dict] | None = None
        for i, m in enumerate(msgs):
            if not isinstance(m, dict):
                continue
            dirty = False
            new_m = dict(m)
            if isinstance(m.get("content"), str):
                s = _scrub_str(m["content"], hits)
                if s != m["content"]:
                    new_m["content"], dirty = s, True
            tcs = m.get("tool_calls")
            if isinstance(tcs, list):
                new_tcs = []
                for tc in tcs:
                    tc = dict(tc)
                    fn = dict(tc.get("function") or {})
                    if isinstance(fn.get("arguments"), str):
                        s = _scrub_str(fn["arguments"], hits)
                        if s != fn["arguments"]:
                            fn["arguments"], dirty = s, True
                    tc["function"] = fn
                    new_tcs.append(tc)
                new_m["tool_calls"] = new_tcs
            if dirty:
                if new_msgs is None:
                    new_msgs = [dict(x) if isinstance(x, dict) else x for x in msgs]
                new_msgs[i] = new_m
        if new_msgs is not None:
            cleaned["messages"] = new_msgs

    # exec_log: any string anywhere (error strings embed full request URLs)
    elog = row.get("exec_log")
    if isinstance(elog, list):
        new_elog, changed = _scrub_any(elog, hits)
        if changed:
            cleaned["exec_log"] = new_elog

    reasons = [f"secret leak x{hits[0]} (scrubbed in output copy)"] if hits[0] else []
    return reasons, cleaned


# ---------------------------------------------------------------- language
_LANG_STOP = {
    "en": frozenset("the i you to and a is it of for on with my me we in at do that this can could "
                    "would please have what when how are be if or so not just about there here get "
                    "want need from your while due".split()),
    "de": frozenset("ich du wir sie er der die das den dem ein eine einen und oder aber mit von nach "
                    "für auf im ist bin habe hat nicht kein möchte kann bitte danke wenn wie was wo "
                    "dann noch schon mir mich sehr auch hier fahre fahren soll gerne".split()),
    "pl": frozenset("nie się jest jestem mam masz chcę chciałbym proszę dziękuję oraz lub ale że to "
                    "ten ta te dla przy po za na do z w o od jak co gdzie kiedy czy tylko jeszcze "
                    "już bardzo mnie mi cię ty wy my on ona może by bym żeby".split()),
    "fr": frozenset("je tu il elle nous vous le la les un une des du de et ou mais que qui ne pas "
                    "pour avec dans sur en au aux ce cette ces mon ma mes votre vos est suis ai peut "
                    "pouvez voudrais veux souhaite merci si comme très où quand y moi toi leur sa "
                    "son ses ça puis ensuite bonjour à".split()),
    "es": frozenset("el la los las un una unos unas de del y e o u pero que quién no más para por "
                    "con en sin sobre al lo su sus mi mis tu tus estoy soy es son está tengo quiero "
                    "necesito gracias sí como muy bien luego después dónde cuando se le les me te "
                    "nos si ha hay esta este esto estos estas yo tú usted podría gustaría".split()),
}
_LANG_DIACRITICS = {"pl": "ąćęłńóśźż", "de": "äöüß", "es": "áéíóúñü¿¡",
                    "fr": "àâçéèêëîïôùûœ", "en": ""}


def _lang_tokens(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.lower(), re.UNICODE)


def lang_scores(text: str) -> dict[str, float] | None:
    """Stopword+diacritic score per language; None if the text is too short to judge."""
    toks = _lang_tokens(text)
    if len(toks) < 6:
        return None
    out = {}
    for lang, stop in _LANG_STOP.items():
        frac = sum(1 for t in toks if t in stop) / len(toks)
        dia = sum(1 for ch in text.lower() if ch in _LANG_DIACRITICS[lang]) / len(toks)
        out[lang] = frac + 0.05 * dia
    return out


def check_language_heuristic(row: dict) -> tuple[list[str], bool]:
    """Per-user-turn stopword/diacritic check. Returns (reasons, flagged_for_judge)."""
    declared = row.get("lang") or row.get("language") or "en"
    reasons: list[str] = []
    for turn in (m.get("content") or "" for m in row.get("messages") or []
                 if m.get("role") == "user"):
        scores = lang_scores(turn)
        if scores is None:
            continue
        guess = max(scores, key=scores.get)
        if guess != declared and scores[guess] >= 0.12 and scores[guess] > scores[declared] + 0.04:
            msg = f"heuristic: a user turn looks {guess} (declared {declared})"
            if msg not in reasons:
                reasons.append(msg)
    return reasons, bool(reasons)


def judge_language(first_user_turn: str, model: str | None = None) -> str | None:
    # Default to the first configured judge (local vLLM pair); the old OpenRouter
    # default died with the monthly quota.
    if model is None:
        model = config.JUDGE_MODELS[0]
    prompt = (
        "Detect the natural language of this short message from a car driver. "
        "Answer with the ISO 639-1 code only as JSON.\n\n"
        f"Message: {first_user_turn[:600]}\n\n"
        'Respond exactly: {"language": "<en|de|pl|fr|es>"}'
    )
    try:
        blob = chat_json(model, [{"role": "user", "content": prompt}],
                         temperature=0.0, max_tokens=300)
    except LLMError:
        return None
    lang = str(blob.get("language", "")).strip().lower()[:2]
    return lang if lang in config.LANGS else None


# ---------------------------------------------------------------- judge
def render_dialog(row: dict, tool_catalog: list[tuple[str, str]]) -> str:
    lines = [f"Declared language: {row.get('lang', '?')} | mode: {row.get('mode', '?')}",
             "Tools available to the assistant:"]
    lines += [f"- {n}: {d}" for n, d in tool_catalog]
    lines.append("--- Dialog ---")
    for m in row.get("messages") or []:
        role = m.get("role")
        if role == "system":
            continue
        if role == "tool":
            lines.append(f"[tool result {(m.get('tool_call_id') or '')[:10]}]: "
                         f"{(m.get('content') or '')[:4000]}")
        elif role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                lines.append(f"[assistant tool_call] {fn.get('name')}({(fn.get('arguments') or '')[:300]})")
            if (m.get("content") or "").strip():
                lines.append(f"[assistant]: {m['content'][:2000]}")
        else:
            lines.append(f"[{role}]: {(m.get('content') or '')[:2000]}")
    return "\n".join(lines)


_MODE_HINTS = {
    "no_tool": "mode no_tool: this dialog INTENTIONALLY contains no tool calls (chitchat, opinion, "
               "general knowledge, or a non-navigation request). Absence of tool calls is CORRECT "
               "here by design and must NOT lower alignment; do not penalize the assistant for not "
               "navigating, not using tools, or the task not being a navigation task — that is "
               "exactly what this dialog is supposed to be. Judge only whether the assistant's "
               "plain answer serves what the user actually asked in this dialog. If you score "
               "alignment below 7 in this mode, your notes MUST quote the user request and explain "
               "why the answer fails it (not merely the absence of tools).",
    "error_recovery": "mode error_recovery: the user deliberately makes a slip (bad place name, "
                      "ambiguous spot, wrong format). Judge the recovery: a corrected tool call "
                      "that succeeds, or a precise clarifying question, is excellent. An honest, "
                      "plain statement of what could not be found is acceptable; silently "
                      "inventing data is not.",
    "tool_use": "mode tool_use: live-data task; tool calls are expected and the final answer "
                "should be grounded in them.",
}


def judge_prompt(row: dict, tool_catalog: list[tuple[str, str]]) -> str:
    mode_hint = _MODE_HINTS.get(row.get("mode", ""))
    return (
        "You are a strict quality judge for synthetic function-calling dialogues used to "
        "fine-tune an in-car navigation assistant.\n\n"
        + render_dialog(row, tool_catalog)
        + ("\n\nDialog type: " + mode_hint + "\n" if mode_hint else "")
        + "\nScore each dimension 1-10 (10 = excellent):\n"
          "- alignment: do the tool calls and the final answer serve the user's actual task? "
          "Every explicit user constraint (avoid tolls, scenic route, stops, charging, timing) "
          "must be either satisfied or honestly reported as unmet; silently dropping one is a "
          "major alignment fault. A task that genuinely cannot be completed with the available "
          "tools, where the assistant explains this and asks for what it needs, is GOOD "
          "alignment, not a failure.\n"
          "- grounding: every factual claim in the final answer (places, road numbers, cities "
          "passed, distances, durations, traffic, weather) must come from the tool results shown "
          "above or from the user's own words. Correct arithmetic over tool-provided numbers "
          "(summing leg distances/durations, converting units, rounding) is grounded. IMPORTANT: "
          "if you score grounding 6 or lower, your notes MUST quote the exact ungrounded phrase "
          "from the final answer and name the tool result (or its absence) that contradicts it. "
          "If you cannot quote such evidence, do not score grounding below 7.\n"
          "- naturalness: does the dialog read like a natural, helpful conversation? Briefly "
          "acknowledging a tool failure in plain language is natural; pasting raw errors, JSON "
          "or tool-call markup into the reply is not.\n"
          "- language: do user and assistant turns consistently match the declared language?\n"
          "Be strict but evidence-based: grounded navigation data must come from tool results, "
          "not memory. Your scores must be consistent with your own notes: never give a low "
          "score whose stated reason contradicts the note (e.g. alignment 2 while the note says "
          "the answer is appropriate).\n\n"
          'Respond with JSON only: {"alignment": <1-10>, "grounding": <1-10>, '
          '"naturalness": <1-10>, "language": <1-10>, "notes": "<one short sentence>"}'
    )


def _clamp_score(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return max(1.0, min(10.0, f))


_OPENROUTER_PREFIXES = ("qwen/", "z-ai/", "deepseek/", "google/", "mistralai/", "openai/gpt-oss")


def run_judge(model: str, row: dict, tool_catalog: list[tuple[str, str]]) -> dict | None:
    """Returns {dim: score..., avg, notes} or None if the judge is unusable for this row."""
    with _judge_lock:
        if model in _judge_down:
            return None
    kw: dict[str, Any] = {"temperature": 0.2, "max_tokens": 800}
    if model.startswith(_OPENROUTER_PREFIXES):
        # thinking models would otherwise burn the whole token budget before content
        kw["extra_payload"] = {"reasoning": {"enabled": False}}
    blob: Any = None
    for attempt in range(2):
        try:
            blob = chat_json(model, [{"role": "user", "content": judge_prompt(row, tool_catalog)}], **kw)
            if isinstance(blob, dict) and all(_clamp_score(blob.get(d)) is not None for d in JUDGE_DIMS):
                break
            blob = None
            log(f"judge {model}: malformed scores (attempt {attempt + 1})")
        except LLMError as e:
            log(f"judge {model} attempt {attempt + 1} failed: {str(e)[:160]}")
    if blob is None:
        with _judge_lock:
            _judge_streak[model] += 1
            if _judge_streak[model] >= 3:
                _judge_down.add(model)
                log(f"judge {model} marked DOWN after repeated failures")
        return None
    with _judge_lock:
        _judge_streak[model] = 0
    out = {d: _clamp_score(blob.get(d)) for d in JUDGE_DIMS}
    out["avg"] = sum(out[d] for d in JUDGE_DIMS) / len(JUDGE_DIMS)
    out["notes"] = str(blob.get("notes", ""))[:300]
    return out


# ---------------------------------------------------------------- mmr
_TOK_RE = re.compile(r"[\w'-]+", re.UNICODE)


def _shingles5(text: str) -> frozenset:
    toks = _TOK_RE.findall(text.lower())
    return frozenset(tuple(toks[i:i + 5]) for i in range(len(toks) - 4))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------- per-row worker
def verify_row(row: dict, ctx: dict) -> dict:
    v: dict[str, Any] = {"passed": True, "stages_failed": [], "stages_skipped": [],
                         "reasons": {}, "judges": {}, "mmr_sim_max": 0.0, "notes": []}
    skip = ctx["skip"]

    def fail(stage: str, reasons: list[str]) -> None:
        if not reasons:
            return
        v["stages_failed"].append(stage)
        v["reasons"][stage] = reasons
        v["passed"] = False

    msgs_ok = isinstance(row.get("messages"), list) and bool(row["messages"])

    # structural + collect tool calls (needed by schema even if structural is skipped)
    calls: list[dict] = []
    if not msgs_ok:
        if not skip["structural"]:
            fail("structural", ["messages missing or not a list"])
    else:
        reasons, calls = check_structural(row, ctx["known_tools"])
        if not skip["structural"]:
            fail("structural", reasons)
        else:
            v["stages_skipped"].append("structural")

    # schema
    if skip["schema"]:
        v["stages_skipped"].append("schema")
    else:
        s_reasons: list[str] = []
        for call in calls:
            schema = ctx["registry"].get_schema(call["name"]) if ctx["registry"] else None
            check_schema(call, schema, s_reasons, v["notes"])
        fail("schema", s_reasons)

    # exec replay (results precomputed globally, cache-aware)
    if skip["exec"]:
        v["stages_skipped"].append("exec")
    else:
        e_reasons: list[str] = []
        n_ok = n_bad = 0
        for e in row.get("exec_log") or []:
            key = ctx["key_fn"](e.get("name"), e.get("args") or {})
            if key is None:
                e_reasons.append(f"unknown tool in exec_log: {e.get('name')!r}")
                continue
            rep = ctx["replay"].get(key)
            if rep is None:
                continue
            if e.get("ok") and not rep["ok"]:
                n_bad += 1
                e_reasons.append(f"replay failed: {e.get('name')} ({str(rep.get('error'))[:120]})")
            elif e.get("ok"):
                n_ok += 1
        v["exec"] = {"recorded_ok": n_ok, "replay_failed": n_bad}
        fail("exec", e_reasons)

    # secrets
    if skip["secrets"]:
        v["stages_skipped"].append("secrets")
    else:
        s_reasons, cleaned = scan_secrets(row)
        if cleaned:
            for field, scrubbed in cleaned.items():
                row[f"_scrubbed_{field}"] = scrubbed
        fail("secrets", s_reasons)

    # language
    if skip["language"]:
        v["stages_skipped"].append("language")
    else:
        declared = row.get("lang") or row.get("language") or "en"
        l_reasons, flagged = check_language_heuristic(row)
        if flagged:
            first_user = next((m.get("content") or "" for m in row["messages"]
                               if m.get("role") == "user"), "")
            confirmed = judge_language(first_user)
            if confirmed is None:
                v["notes"].append("language flagged by heuristic but judge unavailable; not confirmed")
            elif confirmed != declared:
                l_reasons.append(f"judge confirms {confirmed} != declared {declared}")
            else:
                # spec rule: judge confirmed the first turn matches the declared language;
                # keep the mixed-language suspicion as a note instead of a failure
                v["notes"].append("mixed-language user turns suspected ("
                                  + "; ".join(l_reasons) + f"); judge confirmed first turn = {declared}")
                l_reasons = []
        fail("language", l_reasons)

    # dual-LLM judge
    if skip["judge"]:
        v["stages_skipped"].append("judge")
    elif msgs_ok:
        judges: dict[str, dict] = {}
        unavailable: list[str] = []
        for model in config.JUDGE_MODELS:
            res = run_judge(model, row, ctx["tool_catalog"])
            if res is None:
                unavailable.append(model)
            else:
                judges[model] = res
        v["judges"] = judges
        if unavailable:
            v["notes"].append("judge unavailable: " + ", ".join(unavailable))
        if not judges:
            fail("judge", ["all judges unavailable"])
        else:
            avgs = [j["avg"] for j in judges.values()]
            min_dim = min(min(j[d] for d in JUDGE_DIMS) for j in judges.values())
            j_reasons: list[str] = []
            if min(avgs) < config.JUDGE_PASS_SCORE:
                pair = ", ".join(f"{m}={a:.1f}" for m, a in zip(judges, avgs))
                j_reasons.append(f"judge avg < {config.JUDGE_PASS_SCORE} ({pair})")
            if min_dim < 5:
                j_reasons.append(f"dimension scored < 5 (min={min_dim:.0f})")
            fail("judge", j_reasons)
    return v


# ---------------------------------------------------------------- io helpers
def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    log(f"{p.name}:{i + 1} unparseable line, skipping: {e}")
                    continue
                row["_src_file"], row["_src_line"] = p.name, i + 1
                rows.append(row)
    return rows


def write_report(args: argparse.Namespace, summary: dict, out_rows: list[dict]) -> None:
    t = summary["totals"]
    lines = [
        "# Verification report (pilot)", "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | runtime: {t['runtime_s']}s", "",
        "## Totals", "",
        f"- input rows: **{t['in']}** | passed: **{t['passed']}** | failed: **{t['failed']}** "
        f"| pass rate: **{t['pass_rate'] * 100:.1f}%**",
        f"- inputs: {', '.join('`' + p + '`' for p in summary['params']['in'])}", "",
        "## Per-stage failures", "",
        "| stage | pass | fail | skipped |", "|---|---|---|---|",
    ]
    for stage, st in summary["stages"].items():
        lines.append(f"| {stage} | {st['pass']} | {st['fail']} | {st['skipped']} |")
    lines += ["", "## Breakdown by mode / language / teacher", ""]
    for title, key in (("Mode", "by_mode"), ("Language", "by_lang"), ("Teacher", "by_teacher")):
        lines += [f"### {title}", "| group | total | passed | pass rate |", "|---|---|---|---|"]
        for g, st in sorted(summary[key].items()):
            lines.append(f"| {g} | {st['total']} | {st['passed']} | {st['passed'] / st['total'] * 100:.0f}% |")
        lines.append("")
    lines += ["## Judge scores", ""]
    for model, m in summary["judges"]["score_mean"].items():
        histo = " ".join(f"{b}:{c}" for b, c in summary["judges"]["score_histogram"][model].items() if c)
        dims = ", ".join(f"{d}={m[d]}" for d in JUDGE_DIMS)
        lines.append(f"- `{model}` (n={m['n']}): {dims}, avg={m['avg']} | histogram(avg): {histo}")
    if summary["judges"]["unavailable_models"]:
        lines.append(f"- unavailable judges: {', '.join(summary['judges']['unavailable_models'])} "
                     "(affected dialogs passed on the remaining judge alone, noted per row)")
    ex = summary["exec"]
    lines += ["", "## Exec replay", "",
              f"- unique recorded calls: {ex['unique_calls']} "
              f"(cache hits: {ex['cache_hits']}, live: {ex['live_calls']})",
              f"- recorded-ok calls: {ex['replay_ok']} replayed ok, {ex['replay_fail']} failed on replay",
              f"- recorded-fail calls that succeed now (recovered): {ex['recovered']}", "",
              "## Language-consistency catches (expected ~3 French mislabels)", ""]
    lang_fails = [(r["id"], r["verification"]["reasons"].get("language"))
                  for r in out_rows if "language" in r["verification"]["stages_failed"]]
    if lang_fails:
        for rid, rs in lang_fails:
            lines.append(f"- `{rid}`: {'; '.join(rs or [])}")
    else:
        lines.append("- none found")
    lines += ["", "## Rejection reasons histogram", ""]
    for reason, n in list(summary["reject_reasons"].items())[:25]:
        lines.append(f"- {n}x {reason}")
    examples = [r for r in out_rows if not r["verification"]["passed"]][:8]
    if examples:
        lines += ["", "## Example rejected dialogs", ""]
        for r in examples:
            first_user = next((m.get("content", "")[:90] for m in r["messages"]
                               if m.get("role") == "user"), "")
            vv = r["verification"]
            lines.append(f"- `{r['id']}` ({vv['src_file']}:{vv['src_line']}, {r['mode']}/{r['lang']}) "
                         f"failed={vv['stages_failed']}: \"{first_user}\"")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"report -> {args.report}")


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="APIGen-style verification of raw dialog jsonl files")
    ap.add_argument("--in", nargs="+", type=Path, required=True, help="raw dialog jsonl files")
    ap.add_argument("--out", type=Path, default=config.VERIFIED_DIR / "pilot30_verified.jsonl")
    ap.add_argument("--summary", type=Path, default=config.VERIFIED_DIR / "pilot30_summary.json")
    ap.add_argument("--report", type=Path, default=config.ROOT / "reports" / "VERIFY_PILOT.md")
    ap.add_argument("--workers", type=int, default=4, help="max concurrent LLM calls")
    ap.add_argument("--mmr-threshold", type=float, default=0.6)
    ap.add_argument("--no-structural", action="store_true")
    ap.add_argument("--no-schema", action="store_true")
    ap.add_argument("--no-exec", action="store_true")
    ap.add_argument("--no-secrets", action="store_true")
    ap.add_argument("--no-language", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--no-mmr", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    skip = {s: getattr(args, f"no_{s}") for s in STAGE_ORDER}
    in_files: list[Path] = getattr(args, "in")
    rows = load_rows(in_files)
    log(f"loaded {len(rows)} rows from {len(in_files)} files")

    need_registry = not (skip["structural"] and skip["schema"] and skip["exec"])
    reg: ToolRegistry | None = None
    if need_registry:
        reg = ToolRegistry()
        log(f"registry ready: {len(reg.schemas)} tools")

    ctx: dict[str, Any] = {
        "skip": skip, "registry": reg,
        "known_tools": {s["function"]["name"] for s in reg.schemas} if reg else set(),
        "tool_catalog": [], "replay": {}, "key_fn": lambda name, args: None,
    }
    if reg:
        ctx["tool_catalog"] = [
            (s["function"]["name"],
             (s["function"].get("description") or "").split(". ")[0].split("\n")[0][:110])
            for s in reg.schemas
        ]

        def key_fn(name: str | None, args_: Any) -> str | None:
            if name not in reg.providers or not isinstance(args_, dict):
                return None
            norm = tr._normalize_args(name, args_)
            return tr._cache_key(reg.providers[name], name, norm)

        ctx["key_fn"] = key_fn

    # ---- exec replay phase (sequential, deduped globally, mostly cache hits)
    exec_stats = {"unique_calls": 0, "cache_hits": 0, "live_calls": 0,
                  "replay_ok": 0, "replay_fail": 0, "recovered": 0}
    if not skip["exec"] and reg:
        uniq: dict[str, tuple[str, dict]] = {}
        for row in rows:
            for e in row.get("exec_log") or []:
                k = ctx["key_fn"](e.get("name"), e.get("args") or {})
                if k and k not in uniq:
                    uniq[k] = (e["name"], tr._normalize_args(e["name"], e["args"] or {}))
        exec_stats["unique_calls"] = len(uniq)
        log(f"exec replay: {len(uniq)} unique tool calls")
        for j, (k, (name, norm)) in enumerate(sorted(uniq.items()), 1):
            res = reg.execute(name, norm)
            ctx["replay"][k] = res
            exec_stats["cache_hits" if res["cached"] else "live_calls"] += 1
            if j % 20 == 0 or j == len(uniq):
                log(f"exec replay {j}/{len(uniq)} (cache_hits={exec_stats['cache_hits']})")
        for row in rows:
            for e in row.get("exec_log") or []:
                rep = ctx["replay"].get(ctx["key_fn"](e.get("name"), e.get("args") or {}))
                if rep is None:
                    continue
                if e.get("ok"):
                    exec_stats["replay_ok" if rep["ok"] else "replay_fail"] += 1
                elif rep["ok"]:
                    exec_stats["recovered"] += 1
        reg.flush_cache()

    # ---- parallel per-row stages (judges inside; <=workers concurrent LLM calls)
    for row in rows:
        row["_verification"] = None
    if rows:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(verify_row, row, ctx): row for row in rows}
            for done, (fut, row) in enumerate(futures.items(), 1):
                try:
                    row["_verification"] = fut.result()
                except Exception as e:  # noqa: BLE001
                    log(f"verify {row.get('id')} crashed: {e!r}")
                    row["_verification"] = {
                        "passed": False, "stages_failed": ["internal"], "stages_skipped": [],
                        "reasons": {"internal": [repr(e)[:200]]}, "judges": {},
                        "mmr_sim_max": 0.0, "notes": []}
                if done % 10 == 0 or done == len(rows):
                    log(f"row checks {done}/{len(rows)}")

    # ---- sequential MMR pass (greedy keep in first-seen order)
    kept: list[frozenset] = []
    for row in rows:
        v = row["_verification"]
        if skip["mmr"]:
            v["stages_skipped"].append("mmr")
            continue
        users = " ".join((m.get("content") or "") for m in row.get("messages") or []
                         if m.get("role") == "user")
        prof = _shingles5(users)
        sim_max = max((_jaccard(prof, k) for k in kept), default=0.0)
        v["mmr_sim_max"] = round(sim_max, 4)
        if sim_max >= args.mmr_threshold:
            v["stages_failed"].append("mmr")
            v["reasons"]["mmr"] = [f"5-gram Jaccard {sim_max:.2f} >= {args.mmr_threshold} vs a kept dialog"]
            v["passed"] = False
        elif v["passed"]:
            kept.append(prof)

    # ---- write output jsonl (including failed rows with verdicts)
    out_rows = []
    for row in rows:
        v = row["_verification"]
        rec = {k: val for k, val in row.items() if not k.startswith("_")}
        for field in ("messages", "exec_log"):
            if f"_scrubbed_{field}" in row:
                rec[field] = row[f"_scrubbed_{field}"]
        v_out = {k: v[k] for k in ("passed", "stages_failed", "judges", "mmr_sim_max", "notes")}
        v_out["stages_skipped"] = v.get("stages_skipped", [])
        v_out["reasons"] = v.get("reasons", {})
        if "exec" in v:
            v_out["exec"] = v["exec"]
        v_out["src_file"], v_out["src_line"] = row["_src_file"], row["_src_line"]
        rec["verification"] = v_out
        out_rows.append(rec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- summary json
    passed = sum(1 for r in out_rows if r["verification"]["passed"])
    summary: dict[str, Any] = {
        "params": {"in": [str(p) for p in in_files], "out": str(args.out),
                   "workers": args.workers, "mmr_threshold": args.mmr_threshold,
                   "skipped": [s for s in STAGE_ORDER if skip[s]],
                   "judge_models": config.JUDGE_MODELS, "judge_pass_score": config.JUDGE_PASS_SCORE},
        "totals": {"in": len(out_rows), "passed": passed, "failed": len(out_rows) - passed,
                   "pass_rate": round(passed / len(out_rows), 4) if out_rows else 0.0,
                   "runtime_s": round(time.time() - t0, 1)},
        "stages": {}, "by_mode": {}, "by_lang": {}, "by_teacher": {},
        "reject_reasons": {}, "exec": exec_stats,
        "judges": {"unavailable_models": sorted(_judge_down), "score_mean": {}, "score_histogram": {}},
    }
    for stage in STAGE_ORDER:
        n_fail = sum(1 for r in out_rows if stage in r["verification"]["stages_failed"])
        n_skip = sum(1 for r in out_rows if stage in r["verification"]["stages_skipped"])
        summary["stages"][stage] = {"fail": n_fail, "skipped": n_skip,
                                    "pass": len(out_rows) - n_fail - n_skip}
    for key, field in (("by_mode", "mode"), ("by_lang", "lang"), ("by_teacher", "teacher")):
        agg: dict[str, dict] = {}
        for r in out_rows:
            g = r.get(field) or "?"
            agg.setdefault(g, {"total": 0, "passed": 0})
            agg[g]["total"] += 1
            agg[g]["passed"] += 1 if r["verification"]["passed"] else 0
        summary[key] = agg
    reasons: Counter = Counter()
    for r in out_rows:
        for stage, rs in r["verification"]["reasons"].items():
            for reason in rs:
                reasons[f"{stage}: {reason[:110]}"] += 1
    summary["reject_reasons"] = dict(reasons.most_common())
    for model in config.JUDGE_MODELS:
        vals = [r["verification"]["judges"][model] for r in out_rows
                if model in r["verification"]["judges"]]
        if vals:
            summary["judges"]["score_mean"][model] = {
                **{d: round(sum(v[d] for v in vals) / len(vals), 2) for d in JUDGE_DIMS},
                "avg": round(sum(v["avg"] for v in vals) / len(vals), 2), "n": len(vals)}
            summary["judges"]["score_histogram"][model] = {
                str(b): sum(1 for v in vals if int(v["avg"]) == b) for b in range(1, 11)}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    if reg:
        reg.close()
    write_report(args, summary, out_rows)
    log(f"done: {passed}/{len(out_rows)} passed -> {args.out} (runtime {summary['totals']['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
