"""OOD gold builder: unseen-tool function-calling eval items.

Source: NousResearch/hermes-function-calling-v1 func-calling-singleturn.json
(1,893 rows, ungated, Apache-2.0). Tools there are completely disjoint from our
21 geo/map tools -> clean out-of-domain generalization probe.

Emits gold-shaped items (compatible with eval/harness.py --ood):
  {"gold_id", "language": "en", "mode": "ood", "ood": true,
   "user_turns": [human], "tools": [...wrapped schemas...],
   "reference": {"tool_calls": [{"step": 1, "name", "args"}...]},
   "teacher": "hermes-fc", "preferred_provider": "any",
   "category", "subcategory", "task", "created"}

Usage:
  .venv/bin/python -m eval.build_ood --n 300 --out data/test_gold_ood.jsonl
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import random
import re
from pathlib import Path

DEFAULT_SOURCE = Path(
    "/var/tmp/fcft_hf_cache/hub/datasets--NousResearch--hermes-function-calling-v1/"
    "snapshots/dae3e1d28cfbcf4b915c04ea1e072030529b4bda/func-calling-singleturn.json"
)
# dataset mixes real newlines and literal "\n" text; bodies may be JSON or python-repr
TOOL_CALL_RE = re.compile(
    r"<tool_call>(?:\\n|\n|\s)*(\{.*?\})(?:\\n|\n|\s)*</tool_call>", re.DOTALL)


def _parse_call_blob(raw: str) -> dict | None:
    for loader in (json.loads, ast.literal_eval):
        try:
            blob = loader(raw)
            return blob if isinstance(blob, dict) else None
        except (ValueError, SyntaxError):
            continue
    return None


def parse_tool_calls(text: str) -> list[dict]:
    """Extract reference calls. NOTE: ~793 'Information Extraction' rows put the
    tool name inside arguments AND pass args that mismatch the declared schema
    (queries vs *_questions params) — unusable as gold, dropped via name check."""
    calls: list[dict] = []
    for m in TOOL_CALL_RE.finditer(text):
        blob = _parse_call_blob(m.group(1))
        if blob is None:
            continue
        name = str(blob.get("name") or "").strip()
        args = blob.get("arguments")
        if name and isinstance(args, dict):
            calls.append({"name": name, "args": args})
    return calls


def build_items(rows: list[dict]) -> tuple[list[dict], dict]:
    stats = {"rows": len(rows), "tools_parse_fail": 0, "no_calls": 0,
             "name_not_in_tools": 0, "no_user_turn": 0, "kept": 0}
    items: list[dict] = []
    for r in rows:
        try:
            tools = json.loads(r["tools"])
        except (json.JSONDecodeError, TypeError):
            stats["tools_parse_fail"] += 1
            continue
        if not isinstance(tools, list) or not tools:
            stats["tools_parse_fail"] += 1
            continue
        names = set()
        for t in tools:
            fn = t.get("function") or {}
            if not fn.get("name"):
                stats["tools_parse_fail"] += 1
                names = None  # type: ignore[assignment]
                break
            names.add(fn["name"])
        if names is None:
            continue
        human = next((c.get("value") or "").strip()
                     for c in r["conversations"] if c.get("from") == "human")
        if not human:
            stats["no_user_turn"] += 1
            continue
        gpt = next((c.get("value") or "")
                   for c in r["conversations"] if c.get("from") == "gpt")
        calls = parse_tool_calls(gpt)
        if not calls:
            stats["no_calls"] += 1
            continue
        if any(c["name"] not in names for c in calls):
            stats["name_not_in_tools"] += 1
            continue
        items.append({
            "gold_id": f"ood-hermes-{r['id']}",
            "language": "en",
            "mode": "ood",
            "ood": True,
            "user_turns": [human],
            "tools": tools,
            "reference": {"tool_calls": [
                {"step": 1, "name": c["name"], "args": c["args"]}
                for c in calls]},
            "teacher": "hermes-fc",
            "user_teacher": "hermes-fc",
            "preferred_provider": "any",
            "category": r.get("category") or "unknown",
            "subcategory": r.get("subcategory") or "",
            "task": (r.get("task") or "")[:200],
            "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        })
        stats["kept"] += 1
    return items, stats


def stratified_sample(items: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {}
    for it in items:
        by_cat.setdefault(it["category"], []).append(it)
    for lst in by_cat.values():
        rng.shuffle(lst)
    total = len(items)
    out: list[dict] = []
    for cat in sorted(by_cat):
        k = round(n * len(by_cat[cat]) / total)
        out.extend(by_cat[cat][:k])
    if len(out) > n:
        rng.shuffle(out)
        out = out[:n]
    return sorted(out, key=lambda x: x["gold_id"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--out", type=Path, default=Path("data/test_gold_ood.jsonl"))
    args = ap.parse_args()

    rows = json.loads(args.source.read_text())
    items, stats = build_items(rows)
    sample = stratified_sample(items, args.n, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for it in sample:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    cats: dict[str, int] = {}
    for it in sample:
        cats[it["category"]] = cats.get(it["category"], 0) + 1
    n_calls = sum(len(it["reference"]["tool_calls"]) for it in sample)
    print(f"[build_ood] filters: {stats}")
    print(f"[build_ood] wrote {len(sample)} items -> {args.out}")
    print(f"[build_ood] ref calls total {n_calls} "
          f"(avg {n_calls / max(len(sample), 1):.2f}/item)")
    print(f"[build_ood] categories: " +
          ", ".join(f"{c}={k}" for c, k in sorted(cats.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
