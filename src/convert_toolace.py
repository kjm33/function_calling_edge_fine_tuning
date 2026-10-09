"""Convert ToolACE (Team-ACE/ToolACE data.json) into our neutral dialog format.

ToolACE rows: {system: <FC prompt + embedded tools JSON array + call-format
instructions>, conversations: [{from: user|assistant|tool, value}]} with
pythonic call syntax: [Name(k=v, k2="str"), Other(x=1)].

Output rows match our verified-corpus schema so they flow through
build_dataset --general-mix and render_templates unchanged:
  {id, lang, mode, source, tools: [{type,function{name,description,parameters}}],
   messages: [{role, content, tool_calls?, tool_call_id?}]}

Filters: unparseable call turns, empty assistant turns, tool-result count
mismatches, dialogs with dangling calls.

Usage:
    python -m src.convert_toolace --in /path/to/data.json --out data/toolace/clean.jsonl
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

TOOLS_MARKER = "invoke:\n"

NEUTRAL_SYSTEM = (
    "You are a helpful assistant with access to external tools. "
    "Use the provided tools when they can answer the user's request, following "
    "each tool's JSON schema exactly. If no tool fits the request, or required "
    "parameters are missing, say so instead of calling a tool."
)


def extract_tools(system: str) -> list[dict] | None:
    if TOOLS_MARKER not in system:
        return None
    rest = system.split(TOOLS_MARKER, 1)[1].strip()
    start = rest.find("[")
    if start < 0:
        return None
    try:
        arr, _ = json.JSONDecoder().raw_decode(rest[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list) or not arr:
        return None
    return arr


def normalize_tool(t: dict) -> dict | None:
    if not isinstance(t, dict):
        return None
    fn = t.get("function") if isinstance(t.get("function"), dict) else t
    name = fn.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    params = fn.get("parameters")
    if not isinstance(params, dict) or not params.get("properties"):
        params = {"type": "object", "properties": {}}
    params = dict(params)
    req = params.get("required")
    if req is not None and not isinstance(req, list):
        params.pop("required", None)
    return {
        "type": "function",
        "function": {
            "name": name.strip(),
            "description": (fn.get("description") or name.strip())[:2000],
            "parameters": params,
        },
    }


def _split_top(s: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    cur: list[str] = []
    for ch in s:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            cur.append(ch)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    parts.append("".join(cur))
    return parts


def _parse_kwargs(argstr: str) -> dict:
    args: dict = {}
    argstr = argstr.strip()
    if not argstr:
        return args
    for part in _split_top(argstr):
        k, sep, v = part.partition("=")
        if not sep or not k.strip():
            raise ValueError(f"bad kwarg: {part[:60]!r}")
        v = v.strip()
        try:
            args[k.strip()] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            args[k.strip()] = v
    return args


def parse_call_turn(value: str) -> list[tuple[str, dict]]:
    v = value.strip()
    if not (v.startswith("[") and v.endswith("]")):
        raise ValueError("not a call turn")
    inner = v[1:-1].strip()
    if not inner:
        return []
    calls: list[tuple[str, dict]] = []
    for item in _split_top(inner):
        item = item.strip()
        m = re.match(r"^(.+?)\((.*)\)$", item, re.S)
        if not m:
            raise ValueError(f"not a call: {item[:60]!r}")
        name = m.group(1).strip().strip("'\"")
        if not name:
            raise ValueError("empty tool name")
        calls.append((name, _parse_kwargs(m.group(2))))
    return calls


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def convert_row(idx: int, row: dict, keep_truncated: bool = False) -> tuple[dict | None, str]:
    tools_raw = extract_tools(row.get("system") or "")
    if tools_raw is None:
        return None, "tools_extract_fail"
    tools = [t for t in (normalize_tool(x) for x in tools_raw) if t]
    if not tools:
        return None, "tools_normalize_fail"
    tool_names = {t["function"]["name"] for t in tools}

    messages: list[dict] = [{"role": "system", "content": NEUTRAL_SYSTEM}]
    pending: list[tuple[str, str]] = []  # (call_id, name) awaiting results
    n_call_turns = 0
    user_text: list[str] = []
    truncated = False

    for ci, conv in enumerate(row.get("conversations") or []):
        role, value = conv.get("from"), conv.get("value") or ""
        if role == "user":
            messages.append({"role": "user", "content": value})
            user_text.append(value)
        elif role == "assistant":
            v = value.strip()
            if not v:
                return None, "empty_assistant"
            if v.startswith("[") and v.endswith("]"):
                try:
                    calls = parse_call_turn(v)
                except ValueError:
                    return None, "call_parse_fail"
                if not calls:
                    return None, "empty_call_list"
                tcs = []
                for j, (name, args) in enumerate(calls):
                    if name not in tool_names:
                        return None, "call_name_not_in_tools"
                    cid = f"call_toolace_{idx}_{ci}_{j}"
                    tcs.append({
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    })
                    pending.append((cid, name))
                messages.append({"role": "assistant", "content": "", "tool_calls": tcs})
                n_call_turns += 1
            else:
                if pending:
                    return None, "answer_before_results"
                messages.append({"role": "assistant", "content": value})
        elif role == "tool":
            try:
                payload = json.loads(value)
            except json.JSONDecodeError:
                return None, "tool_json_fail"
            results = payload if isinstance(payload, list) else [payload]
            if len(results) != len(pending):
                return None, "result_count_mismatch"
            for (cid, name), res in zip(pending, results):
                body = res.get("results", res) if isinstance(res, dict) else res
                messages.append({
                    "role": "tool",
                    "tool_call_id": cid,
                    "name": name,
                    "content": json.dumps(body, ensure_ascii=False)[:8000],
                })
            pending = []
        else:
            return None, f"unknown_role_{role}"

    if pending:
        # ToolACE single-turn dialogs end right after the call emission (no
        # tool result / final answer shipped). Those rows still teach the
        # core query->call mapping; keep them behind --keep-truncated.
        if keep_truncated:
            truncated = True
        else:
            return None, "dangling_calls"
    if not any(m["role"] == "assistant" for m in messages):
        return None, "no_assistant_turn"

    langs = "zh" if any(has_cjk(u) for u in user_text) else "en"
    mode = "tool_use" if n_call_turns else "no_tool"
    return {
        "id": f"toolace-{idx:05d}",
        "lang": langs,
        "mode": mode,
        "source": "toolace",
        "truncated": truncated,
        "tools": tools,
        "messages": messages,
        "stats": {"turns": len(messages), "call_turns": n_call_turns},
    }, "ok"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", required=True, help="ToolACE data.json")
    p.add_argument("--out", default="data/toolace/clean.jsonl")
    p.add_argument("--keep-truncated", action="store_true",
                   help="keep dialogs that end right after call emission (no tool result/final answer)")
    args = p.parse_args(argv)

    src = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = json.loads(src.read_text(encoding="utf-8"))
    kept: list[dict] = []
    drop: Counter = Counter()
    for idx, row in enumerate(rows):
        conv, why = convert_row(idx, row, keep_truncated=args.keep_truncated)
        if conv is None:
            drop[why] += 1
        else:
            kept.append(conv)

    with out.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_call_rows = sum(1 for r in kept if r["mode"] == "tool_use")
    n_trunc = sum(1 for r in kept if r.get("truncated"))
    print(f"[toolace] read={len(rows)} kept={len(kept)} "
          f"(tool_use={n_call_rows} no_tool={len(kept)-n_call_rows} truncated={n_trunc})")
    print(f"[toolace] drops: {dict(drop.most_common())}")
    print(f"[toolace] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
