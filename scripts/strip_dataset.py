"""Strip tool definitions from neutral dialogs, keeping tool names only.

Reads neutral JSONL rows (messages + tools), replaces the full `tools` schemas
with a `tool_names` list and writes rows to the output dir. All other fields
(messages, verification, exec_log, stats, ids) are passed through untouched.

Usage:
    python scripts/strip_dataset.py --in data/final/train.jsonl data/final/val.jsonl \
        --out-dir data/final_stripped
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] {path}:{ln}: skipping unparseable line ({e})", file=sys.stderr)
    return rows


def strip_row(row: dict) -> dict:
    out = dict(row)
    tools = out.pop("tools", None)
    names = []
    if tools:
        for t in tools:
            fn = t.get("function", {}) if isinstance(t, dict) else {}
            name = fn.get("name") or t.get("name")
            if name:
                names.append(name)
    if names:
        out["tool_names"] = names
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", nargs="+", required=True, dest="inputs")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": []}
    total_in = total_out = 0
    for src in args.inputs:
        path = Path(src)
        if not path.is_absolute():
            path = ROOT / path
        rows = read_jsonl(path)
        stripped = [strip_row(r) for r in rows]
        out_path = out_dir / path.name
        with out_path.open("w", encoding="utf-8") as f:
            for row in stripped:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        size_in, size_out = path.stat().st_size, out_path.stat().st_size
        total_in += size_in
        total_out += size_out
        entry = {
            "source": str(path),
            "output": str(out_path),
            "rows": len(stripped),
            "bytes_in": size_in,
            "bytes_out": size_out,
        }
        stats["files"].append(entry)
        print(f"[strip] {path.name}: {len(stripped)} rows, "
              f"{size_in / 1e6:.1f}MB -> {size_out / 1e6:.1f}MB "
              f"({(1 - size_out / size_in) * 100:.0f}% smaller)")

    stats["bytes_in_total"] = total_in
    stats["bytes_out_total"] = total_out
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[strip] total: {total_in / 1e6:.1f}MB -> {total_out / 1e6:.1f}MB "
          f"({(1 - total_out / total_in) * 100:.0f}% smaller)")
    print(f"[strip] stats -> {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
