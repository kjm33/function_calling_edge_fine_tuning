"""Verify render_templates against real pilot dialogs + write RENDER_EXAMPLES.md.

For each format (qwen3_5, qwen3_2507, llama3_json, phi4_mini, gemma_hermes) and
each of 3 real pilot30 dialogs (+1 synthetic multi-turn case):
  1. reconstruction: concat(segment texts) == native apply_chat_template render
  2. masking: every assistant span lands in train=True segments (and nothing
     else does): #train segments == #assistant messages, assistant prose is a
     substring of the train text, system/user/tool prose of the mask text
  3. no empty segments
  4. round-trip: tool calls parsed back from train segments equal the source
     calls (name + parsed arguments) — catches argument-encoding bugs

Usage: python scripts/verify_render.py [--data data/raw_trajectories/pilot30.jsonl]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.render_templates import (  # noqa: E402
    FORMATS,
    extract_tool_calls,
    full_render,
    load_tokenizer,
    prepare_messages,
    render_segments,
)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def source_tool_calls(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        for c in m.get("tool_calls") or []:
            args = c["function"].get("arguments")
            if isinstance(args, str):
                args = json.loads(args)
            out.append({"name": c["function"]["name"], "arguments": args})
    return out


def pick_dialogs(path: Path) -> list[tuple[str, dict]]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    multi = seq = no_tool = None
    used: set[str] = set()

    def did(r: dict) -> str:
        return r.get("dialog_id") or r.get("id") or ""

    for r in rows:
        if did(r) in used:
            continue
        msgs = r["messages"]
        rounds = sum(1 for m in msgs if m.get("tool_calls"))
        maxcalls = max((len(m.get("tool_calls") or []) for m in msgs), default=0)
        if maxcalls >= 2 and multi is None:
            multi = r
            used.add(did(r))
        elif rounds >= 2 and seq is None:
            seq = r
            used.add(did(r))
        elif r.get("mode") == "no_tool" and no_tool is None:
            no_tool = r
            used.add(did(r))
        if multi and seq and no_tool:
            break
    picked = [("multi-call", multi), ("sequential-rounds", seq), ("no-tool", no_tool)]
    picked = [(tag, r) for tag, r in picked if r]

    # synthetic multi-turn: real tool round + a second user query afterwards —
    # exercises the qwen3_5 think-block placement for assistants before/after
    # the last user query.
    if picked:
        base = json.loads(json.dumps(picked[0][1]))
        tag, last = base["messages"][-1]["role"], base["messages"][-1]
        followup = {"role": "user", "content": "Bitte fasse das Ergebnis in einem Satz zusammen."}
        closing = {"role": "assistant",
                   "content": "Kurz gesagt: alles wurde gefunden und die Route berechnet."}
        if last.get("role") != "assistant":
            base["messages"] += [followup, closing]
        else:
            base["messages"] += [followup, closing]
        picked.append(("synthetic-multi-turn", base))
    return picked


def verify_case(fmt: str, tag: str, row: dict) -> dict:
    messages, tools = row["messages"], row.get("tools") or []
    segs = render_segments(messages, tools, fmt)
    reference = full_render(messages, tools, fmt)
    joined = "".join(s["text"] for s in segs)

    res = {"fmt": fmt, "tag": tag, "dialog": row.get("dialog_id") or row.get("id"),
           "segments": len(segs), "errors": []}
    if joined != reference:
        res["errors"].append(f"reconstruction mismatch: {len(joined)} vs {len(reference)} chars")
    if not all(s["text"] for s in segs):
        res["errors"].append("empty segment found")

    view, _ = prepare_messages(fmt, messages, tools)
    n_assist = sum(1 for m in view if m.get("role") == "assistant")
    n_train = sum(1 for s in segs if s["train"])
    if n_train != n_assist:
        res["errors"].append(f"train segments ({n_train}) != assistant messages ({n_assist})")

    train_text = norm("".join(s["text"] for s in segs if s["train"]))
    mask_text = norm("".join(s["text"] for s in segs if not s["train"]))
    for m in messages:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        haystack = train_text if m.get("role") == "assistant" else mask_text
        role = m.get("role")
        if norm(content) in haystack:
            continue
        # llama3_json renders role=tool string content through Jinja `tojson`
        # (strings pass its `is iterable` test) -> escaped+quoted form.
        if role == "tool" and norm(json.dumps(content, ensure_ascii=False)) in haystack:
            continue
        flat = lambda s: re.sub(r"[\\\"]", "", s)
        if role == "tool" and flat(norm(content)) in flat(haystack):
            continue
        res["errors"].append(
            f"{role} content not in {'train' if role == 'assistant' else 'mask'} text: "
            f"{content[:60]!r}")

    src_calls = source_tool_calls(messages)
    got_calls = extract_tool_calls(fmt, segs)
    for c in got_calls:
        if not isinstance(c.get("arguments"), dict):
            res["errors"].append(f"round-trip arguments not an object ({fmt}): "
                                 f"{json.dumps(c, ensure_ascii=False)[:120]}")
    if got_calls != src_calls:
        for i, (a, b) in enumerate(zip(src_calls, got_calls)):
            if a != b:
                res["errors"].append(f"round-trip call {i} ({fmt}): src={json.dumps(a, ensure_ascii=False)[:160]} "
                                     f"got={json.dumps(b, ensure_ascii=False)[:160]}")
        if len(src_calls) != len(got_calls):
            res["errors"].append(f"round-trip count {fmt}: src={len(src_calls)} got={len(got_calls)}")
    res["n_calls"] = len(src_calls)
    res["segments_out"] = segs
    return res


def dump_example(segs: list[dict], max_lines: int = 60) -> str:
    lines: list[str] = []
    for i, s in enumerate(segs):
        lines.append(f"▸ segment {i} [{'TRAIN' if s['train'] else 'mask'}]")
        for ln in s["text"].split("\n"):
            lines.append(ln)
    if len(lines) > max_lines:
        lines = lines[:max_lines - 1] + ["…"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(ROOT / "data/raw_trajectories/pilot30.jsonl"))
    p.add_argument("--report", default=str(ROOT / "reports/RENDER_EXAMPLES.md"))
    p.add_argument("--max-lines", type=int, default=60)
    args = p.parse_args(argv)

    dialogs = pick_dialogs(Path(args.data))
    if len(dialogs) < 3:
        print(f"FATAL: only {len(dialogs)} usable pilot dialogs found", file=sys.stderr)
        return 1

    all_ok = True
    report: list[str] = [
        "# Render examples — neutral dialogs → per-model template segments",
        "",
        f"Generated by `scripts/verify_render.py` on {time.strftime('%Y-%m-%d %H:%M')} "
        f"from `{Path(args.data).name}` (3 real dialogs + 1 synthetic multi-turn).",
        "Segments are truncated to ~60 lines; `[TRAIN]` = assistant tokens (loss on),"
        " `[mask]` = loss off.",
        "",
        "| format | model actually loaded | reconstruction | masking | round-trip |",
        "|---|---|---|---|---|",
    ]

    per_format: dict[str, list[dict]] = {}
    for fmt in sorted(FORMATS):
        tok = load_tokenizer(fmt)
        results = []
        for tag, row in dialogs:
            try:
                results.append(verify_case(fmt, tag, row))
            except Exception as e:
                results.append({"fmt": fmt, "tag": tag,
                                "dialog": row.get("dialog_id") or row.get("id"),
                                "segments": 0,
                                "errors": [f"{type(e).__name__}: {e}"], "segments_out": []})
        per_format[fmt] = results
        fmt_ok = all(not r["errors"] for r in results)
        all_ok &= fmt_ok
        report.append(f"| `{fmt}` | `{tok.name_or_path}` "
                      f"| {'✅' if fmt_ok else '❌'} reconstruction"
                      f" {'✅' if fmt_ok else '❌'} masking"
                      f" {'✅' if fmt_ok else '❌'} round-trip |")

        report += ["", f"## {fmt} — {tok.name_or_path}", ""]
        for r in results:
            status = "PASS" if not r["errors"] else "FAIL"
            report.append(f"### [{status}] {r['tag']} — {r['dialog']} "
                          f"({r['segments']} segments)")
            for e in r["errors"]:
                report.append(f"- ERROR: {e}")
            if r.get("segments_out"):
                report += ["", "```text", dump_example(r["segments_out"], args.max_lines), "```", ""]
        report.append("")

    report_path = Path(args.report)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"{'FORMAT':<14} {'MODEL':<42} RESULT")
    for fmt in sorted(FORMATS):
        tok = load_tokenizer(fmt)
        per = per_format[fmt]
        ok = all(not r["errors"] for r in per)
        detail = "" if ok else "; ".join(e for r in per for e in r["errors"])[:200]
        print(f"{fmt:<14} {tok.name_or_path:<42} {'PASS' if ok else 'FAIL'} {detail}")
    print(f"\nreport -> {report_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
