"""Gold test-set builder: teacher-solved, tool-execution-verified eval items.

Reuses dataset_gen machinery (seed loading, user-sim, live tool loop) but with a
dedicated selection stream (random.Random(20260101)) so items are DISJOINT from
training seeds. Chosen seed ids are recorded to data/gold_seed_ids.json as a
guard file (build_dataset can exclude them later).

Item structure (one jsonl line):
  {"gold_id", "language", "mode": tool_use|no_tool, "tools": [...14-tool sample...],
   "user_turns": [str, ...]  (1-3 turns),
   "reference": {"tool_calls": [{"step", "name", "args"}...]  (replay-verified),
                 "final_answer_lang", "rubric"},
   "teacher", "user_teacher", "seed_id", "seed_kind", "preferred_provider", "created"}

Verification-lite: every reference call replayed OK via ToolRegistry (cache hits),
language-hint check, secret redaction, dedup on normalized user_turns[0].
No MMR/judge-pairs here — gold wants coverage, not dedup.

Usage:
  .venv/bin/python -m eval.build_gold --n 24 --out data/test_gold_smoke.jsonl
  .venv/bin/python -m eval.build_gold --n 700 --out data/test_gold.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from src import config
from src.dataset_gen import (_LANG_HINTS, _redact, acceptable, generate_dialog,
                             load_seeds)
from src.llm_client import LLMError, chat, chat_json
from src.tool_registry import ToolRegistry

NAME = "gold_builder"
SELECTION_SEED = 20260101
SEED_IDS_PATH = config.DATA_DIR / "gold_seed_ids.json"
MAX_USER_TURNS = 3

_openrouter_prefixes = ("qwen/", "z-ai/", "deepseek/", "google/", "mistralai/", "openai/gpt-oss")

_lock = threading.Lock()
_stats: dict[str, int] = {}


def _bump(k: str, n: int = 1) -> None:
    with _lock:
        _stats[k] = _stats.get(k, 0) + n


# ----------------------------------------------------------------- rubric
def _rubric_via_llm(teacher: str, lang: str, user_turns: list[str],
                    final_answer: str, calls: list[dict]) -> dict:
    """Short judge rubric (key facts: origin/dest/stop names, provider, numbers)."""
    calls_txt = "\n".join(f"- {c['name']}: {json.dumps(c['args'], ensure_ascii=False)[:200]}"
                          for c in calls[:8]) or "(none)"
    prompt = (
        "You write a SHORT grading rubric for an eval item. Given the user's request, "
        "the tools that were called and the reference answer, list the key facts any "
        "correct final answer must contain (place names as written in the answer, "
        "provider names if mentioned, key numbers like duration/distance/temperature). "
        f"Write the rubric in English, 2-4 sentences, max 90 words.\n\n"
        f"User request: {json.dumps(' | '.join(user_turns)[:600], ensure_ascii=False)}\n\n"
        f"Tool calls:\n{calls_txt}\n\n"
        f"Reference answer:\n{final_answer[:1200]}\n\n"
        'Output ONLY JSON: {"rubric": "...", "key_entities": ["..."]}'
    )
    kw: dict = {"temperature": 0.2, "max_tokens": 400}
    if "qwen" in teacher:  # thinking models burn the budget; glm/known-mandatory ones keep it
        kw["extra_payload"] = {"reasoning": {"enabled": False}}
    try:
        blob = chat_json(teacher, [{"role": "user", "content": prompt}], **kw)
    except LLMError as e:
        if "easoning" not in str(e):  # retry once without reasoning control
            raise
        kw.pop("extra_payload", None)
        blob = chat_json(teacher, [{"role": "user", "content": prompt}], **kw)
    rubric = str(blob.get("rubric") or "").strip()
    if len(rubric) < 20:
        raise LLMError("empty rubric")
    ents = [str(e) for e in (blob.get("key_entities") or []) if str(e).strip()][:8]
    return {"rubric": _redact(rubric), "key_entities": ents}


def _rubric_fallback(lang: str, seed: dict, calls: list[dict], final_answer: str) -> dict:
    o, d = seed.get("origin"), seed.get("destination")
    provs = sorted({c["name"].split("-")[0].split("_")[0] for c in calls})
    facts = []
    if o:
        facts.append(f"origin: {o}")
    if d:
        facts.append(f"destination: {d}")
    if provs:
        facts.append(f"data from provider(s): {', '.join(provs)}")
    facts.append("answer must be grounded in tool results, in " + lang)
    return {"rubric": "; ".join(facts), "key_entities": [x for x in (o, d) if x]}


# ----------------------------------------------------------------- extraction
def _extract_reference(messages: list[dict]) -> tuple[list[str], list[dict], str]:
    """User turns, reference tool calls (step = 1-based user turn being handled),
    final assistant answer."""
    user_turns: list[str] = []
    calls: list[dict] = []
    final_answer = ""
    n_user = 0
    for m in messages:
        if m["role"] == "user":
            n_user += 1
            user_turns.append((m.get("content") or "").strip())
        elif m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                args = tc["function"].get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args or "{}")
                    except Exception:
                        args = {}
                calls.append({"step": n_user, "name": tc["function"]["name"],
                              "args": args or {}})
            if m.get("content"):
                final_answer = m["content"].strip()
    return user_turns, calls, final_answer


def _dedup_calls(calls: list[dict]) -> list[dict]:
    seen: set = set()
    out = []
    for c in calls:
        key = (c["step"], c["name"], json.dumps(c["args"], sort_keys=True, ensure_ascii=False))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _lang_ok(lang: str, user_turns: list[str]) -> bool:
    for t in user_turns[:2]:
        for other, pat in _LANG_HINTS.items():
            if other != lang and pat.search(t):
                return False
    return True


def build_item(reg: ToolRegistry, mode: str, lang: str, seed: dict, teacher: str,
               preferred: str, max_turns: int, item_rng: random.Random) -> dict:
    """One teacher-solved dialog -> verified gold item (raises on any gate failure)."""
    dialog = generate_dialog(reg, seed, teacher, None, lang, preferred, mode,
                             max_turns=max_turns, seed_rng=item_rng)
    ok, why = acceptable(dialog)
    if not ok:
        raise ValueError(f"gate: {why}")
    user_turns, calls, final_answer = _extract_reference(dialog["messages"])
    if not (1 <= len(user_turns) <= MAX_USER_TURNS + 2) or not user_turns:
        raise ValueError(f"unusable user turn count {len(user_turns)}")
    trimmed = len(user_turns) > MAX_USER_TURNS
    if trimmed:
        user_turns = user_turns[:MAX_USER_TURNS]
        calls = [c for c in calls if c["step"] <= MAX_USER_TURNS]
    calls = _dedup_calls(calls)
    if mode == "tool_use" and not calls:
        raise ValueError("tool_use item without reference calls")
    if mode == "no_tool" and calls:
        raise ValueError("no_tool item with reference calls")
    if not _lang_ok(lang, user_turns):
        raise ValueError("language mismatch")
    user_turns = [_redact(u) for u in user_turns]

    # verification-lite: replay every reference call (cache hits make this free)
    for c in calls:
        out = reg.execute(c["name"], c["args"], use_cache=True)
        if not out["ok"]:
            raise ValueError(f"replay failed: {c['name']}: {str(out['error'])[:120]}")
    _bump("replayed", len(calls))

    if mode == "tool_use":
        try:
            rub = _rubric_via_llm(teacher, lang, user_turns, final_answer, calls)
            _bump("rubric_llm")
        except Exception as e:  # noqa: BLE001
            rub = _rubric_fallback(lang, seed, calls, final_answer)
            _bump("rubric_fallback")
            with _lock:
                print(f"[{NAME}] rubric fallback: {type(e).__name__}: {str(e)[:100]}",
                      file=sys.stderr)
    else:
        rub = {"rubric": ("Chitchat/general-knowledge turn: correct behavior is to answer "
                          "directly WITHOUT any tool call, in " + lang + ", helpfully and "
                          "briefly. Topic: " + user_turns[0][:160]),
               "key_entities": []}

    return {
        "gold_id": f"gold-{uuid.uuid4().hex[:12]}",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "language": lang, "mode": mode,
        "tools": dialog["tools"],
        "user_turns": user_turns,
        "reference": {
            "tool_calls": calls,
            "final_answer_lang": lang,
            "rubric": rub["rubric"],
            "key_entities": rub.get("key_entities", []),
        },
        "teacher": teacher, "user_teacher": dialog["user_teacher"],
        "seed_id": seed["id"], "seed_kind": seed["kind"],
        "preferred_provider": dialog["preferred_provider"],
        "trimmed_user_turns": trimmed,
    }


# ----------------------------------------------------------------- main
def _plan(n: int, langs: list[str], no_tool_n: int) -> list[tuple[str, str]]:
    """Deterministic (mode, lang) plan: tool_use balanced across langs, no_tool round-robin."""
    plan: list[tuple[str, str]] = []
    for i in range(no_tool_n):
        plan.append(("no_tool", langs[i % len(langs)]))
    rest = n - no_tool_n
    for i in range(rest):
        plan.append(("tool_use", langs[i % len(langs)]))
    return plan


def _pick_seed(sel_rng: random.Random, seeds: list[dict], lang: str, mode: str) -> dict:
    """Draw a seed compatible with the requested language (English/none seeds adapt;
    non-English seeds only for their own language)."""
    for _ in range(10):
        s = sel_rng.choice(seeds)
        sl = s.get("language")
        if sl not in config.LANGS or sl == "en" or sl == lang:
            return s
    return sel_rng.choice(seeds)  # give up balancing; caller's lang may be overridden


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=24, help="total items")
    ap.add_argument("--out", default=str(config.DATA_DIR / "test_gold_smoke.jsonl"))
    ap.add_argument("--langs", default=",".join(config.LANGS))
    ap.add_argument("--modes", default="tool_use,no_tool",
                    help="enabled modes (subset of tool_use,no_tool)")
    ap.add_argument("--no-tool-n", type=int, default=-1,
                    help="no_tool item count (default: n/6, only when no_tool enabled)")
    ap.add_argument("--teachers", default=",".join(config.OPENAI_TEACHERS + config.OPENROUTER_TEACHERS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=8)
    args = ap.parse_args()

    langs = [x for x in args.langs.split(",") if x]
    modes = [x for x in args.modes.split(",") if x]
    no_tool_n = 0
    if "no_tool" in modes:
        no_tool_n = args.no_tool_n if args.no_tool_n >= 0 else max(1, round(args.n / 6))
    no_tool_n = min(no_tool_n, args.n if modes == ["no_tool"] else max(0, args.n - len(langs)))
    plan = _plan(args.n, langs, no_tool_n)
    teachers = [t for t in args.teachers.split(",") if t]

    seeds = load_seeds()
    print(f"[{NAME}] {len(seeds)} usable seeds; plan: {args.n} items "
          f"({len(plan) - no_tool_n} tool_use / {no_tool_n} no_tool over {langs})")

    sel_rng = random.Random(SELECTION_SEED)
    seen_first_turns: set[str] = set()
    if config.DATA_DIR.joinpath("test_gold.jsonl").exists():  # dedup against full set too
        with open(config.DATA_DIR / "test_gold.jsonl", encoding="utf-8") as f:
            for line in f:
                try:
                    u0 = json.loads(line)["user_turns"][0]
                    seen_first_turns.add(re.sub(r"\s+", " ", u0.lower().strip()))
                except Exception:  # noqa: BLE001
                    pass

    reg = ToolRegistry()
    items: list[dict] = []
    try:
        def job(i: int, mode: str, lang: str) -> dict:
            for attempt in range(4):
                with _lock:
                    seed = _pick_seed(sel_rng, seeds, lang, mode)
                    teacher = sel_rng.choice(teachers)
                    preferred = sel_rng.choice(["any", "any", "tomtom", "here", "keyless"])
                    item_rng = random.Random(sel_rng.randrange(2**31))
                if mode != "no_tool" and seed.get("language") in config.LANGS \
                        and seed["language"] not in (None, "en", lang):
                    continue  # keep the requested language balance
                try:
                    item = build_item(reg, mode, lang, seed, teacher, preferred,
                                      args.max_turns, item_rng)
                except Exception as e:  # noqa: BLE001
                    _bump(f"reject_{mode}")
                    with _lock:
                        print(f"[{NAME}] item {i} attempt {attempt} ({mode}/{lang}) "
                              f"rejected: {str(e)[:140]}", file=sys.stderr)
                    continue
                key = re.sub(r"\s+", " ", item["user_turns"][0].lower().strip())
                with _lock:
                    if key in seen_first_turns:
                        _bump("dup_first_turn")
                        continue
                    seen_first_turns.add(key)
                if seed.get("language") in config.LANGS and seed["language"] != "en":
                    item["language"] = seed["language"]  # non-English seed keeps its language
                return item
            raise RuntimeError(f"item {i} unacceptable after 4 attempts")

        with open(args.out, "w", encoding="utf-8") as fout:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(job, i, m, l): i for i, (m, l) in enumerate(plan)}
                for fut in as_completed(futs):
                    i = futs[fut]
                    try:
                        item = fut.result()
                    except Exception as e:  # noqa: BLE001
                        print(f"[{NAME}] item {i} FAILED: {e}", file=sys.stderr)
                        _bump("gen_fail")
                        continue
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    fout.flush()
                    items.append(item)
                    _bump("ok")
                    print(f"[{NAME}] {len(items)}/{args.n} {item['mode']}/{item['language']} "
                          f"turns={len(item['user_turns'])} refs={len(item['reference']['tool_calls'])} "
                          f"teacher={item['teacher']}", flush=True)
    finally:
        reg.close()

    # guard file: seed ids used for gold (build_dataset can exclude them)
    used = sorted({it["seed_id"] for it in items})
    old: list[str] = []
    if SEED_IDS_PATH.exists():
        try:
            old = json.loads(SEED_IDS_PATH.read_text())
        except Exception:  # noqa: BLE001
            old = []
    merged = sorted(set(old) | set(used))
    SEED_IDS_PATH.write_text(json.dumps(merged, indent=0))
    print(f"[{NAME}] guard file {SEED_IDS_PATH.name}: {len(old)} -> {len(merged)} seed ids")

    # ---- report
    from collections import Counter
    print("\n==== GOLD SMOKE REPORT ====")
    print(f"file: {args.out}  items: {len(items)}")
    print(f"by mode:   {dict(Counter(i['mode'] for i in items))}")
    print(f"by lang:   {dict(Counter(i['language'] for i in items))}")
    print(f"by seed:   {dict(Counter(i['seed_kind'] for i in items))}")
    print(f"teachers:  {dict(Counter(i['teacher'] for i in items))}")
    n_calls = sum(len(i["reference"]["tool_calls"]) for i in items)
    print(f"reference calls: {n_calls} total, all replayed OK; "
          f"multi-call steps: "
          f"{sum(1 for i in items for s in {c['step'] for c in i['reference']['tool_calls']} if sum(1 for c in i['reference']['tool_calls'] if c['step'] == s) > 1)}")
    print(f"turns distribution: {dict(Counter(len(i['user_turns']) for i in items))}")
    print(f"stats: {dict(sorted(_stats.items()))}")
    if items:
        s = next(i for i in items if i["mode"] == "tool_use") if any(i["mode"] == "tool_use" for i in items) else items[0]
        print(f"\nsample {s['gold_id']} ({s['mode']}/{s['language']}):")
        print(f"  user[0]: {s['user_turns'][0][:160]}")
        for c in s["reference"]["tool_calls"][:4]:
            print(f"  ref step{c['step']}: {c['name']} {json.dumps(c['args'], ensure_ascii=False)[:120]}")
        print(f"  rubric: {s['reference']['rubric'][:220]}")
    return 0 if len(items) >= 1 else 1


if __name__ == "__main__":
    sys.exit(main())
