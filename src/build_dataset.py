"""Verified neutral dialogs -> stratified train/val split (+ general-chat mix).

Reads verified JSONL rows (neutral format: messages + tools), drops rows whose
"verification" field marks them as failed (rows without the field pass as-is,
so raw pilots can be used for smoke runs), dedups by dialog id, splits
stratified by (language x mode) and writes train.jsonl + val.jsonl + stats.json.
Rows are written untouched; per-model rendering happens at train time
(src/render_templates.py).

Usage:
    python -m src.build_dataset --in data/verified/part1.jsonl ... \
        [--out-dir data/final] [--val-ratio 0.05] [--max-train N] \
        [--general-mix data/general_chat.jsonl] [--mix-frac 0.12] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "final"


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


def dialog_id(row: dict) -> str | None:
    return row.get("dialog_id") or row.get("id")


def row_language(row: dict) -> str:
    return row.get("language") or row.get("lang") or "en"


def row_mode(row: dict) -> str:
    return row.get("mode") or "unknown"


def verification_passed(row: dict) -> bool:
    v = row.get("verification")
    if v is None:
        return True  # un-verified rows (e.g. raw pilots) accepted as-is
    if isinstance(v, bool):
        return v
    if isinstance(v, dict):
        return v.get("passed") is True
    return False


def stratified_split(rows: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row_language(row), row_mode(row))].append(row)
    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []
    for key in sorted(groups):
        members = groups[key][:]
        rng.shuffle(members)
        n = len(members)
        k = 0
        if n >= 2:
            k = min(max(1, round(n * val_ratio)), n - 1)
        val.extend(members[:k])
        train.extend(members[k:])
    rng.shuffle(train)
    return train, val


def load_general_mix(path: Path) -> list[dict]:
    rows = []
    for row in read_jsonl(path):
        if not row.get("messages"):
            continue
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", nargs="+", required=True, dest="inputs", help="verified neutral jsonl files")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--max-train", type=int, default=0, help="cap train rows (0 = no cap)")
    p.add_argument("--general-mix", default="", help="optional general-chat jsonl to mix into train")
    p.add_argument("--mix-frac", type=float, default=0.12,
                   help="general rows appended = round(mix_frac * tool-use train size)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_read = 0
    dropped_verification = 0
    no_id = 0
    kept: list[dict] = []
    seen: set[str] = set()
    duplicates = 0
    per_source: Counter = Counter()
    for src in args.inputs:
        path = Path(src)
        rows = read_jsonl(path)
        total_read += len(rows)
        for row in rows:
            if not verification_passed(row):
                dropped_verification += 1
                continue
            did = dialog_id(row)
            if not did:
                no_id += 1
                continue
            if did in seen:
                duplicates += 1
                continue
            seen.add(did)
            kept.append(row)
            per_source[path.name] += 1

    train, val = stratified_split(kept, args.val_ratio, args.seed)
    if args.max_train and len(train) > args.max_train:
        train = train[: args.max_train]

    general_added = 0
    general_path: str | None = None
    if args.general_mix:
        gpath = Path(args.general_mix)
        general_path = str(gpath)
        if gpath.exists():
            pool = load_general_mix(gpath)
            n_mix = min(round(len(train) * args.mix_frac), len(pool))
            if n_mix:
                rng = random.Random(args.seed + 1)
                general = rng.sample(pool, n_mix)
                train.extend(general)
                rng.shuffle(train)
                general_added = len(general)
        else:
            print(f"[warn] --general-mix file not found, skipping: {gpath}", file=sys.stderr)

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    strata: Counter = Counter((row_language(r), row_mode(r)) for r in kept)
    train_strata: Counter = Counter((row_language(r), row_mode(r)) for r in train)
    val_strata: Counter = Counter((row_language(r), row_mode(r)) for r in val)
    stats = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": [str(Path(s)) for s in args.inputs],
        "rows_per_source": dict(sorted(per_source.items())),
        "total_read": total_read,
        "dropped_failed_verification": dropped_verification,
        "missing_id": no_id,
        "duplicates_removed": duplicates,
        "kept": len(kept),
        "train": len(train),
        "val": len(val),
        "general_mix_file": general_path,
        "general_mix_added": general_added,
        "val_ratio": args.val_ratio,
        "max_train": args.max_train,
        "seed": args.seed,
        "strata": [
            {
                "language": lang,
                "mode": mode,
                "total": strata[(lang, mode)],
                "train": train_strata[(lang, mode)],
                "val": val_strata[(lang, mode)],
            }
            for lang, mode in sorted(strata)
        ],
    }
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[build] read={total_read} kept={len(kept)} "
          f"(verification_dropped={dropped_verification} dup={duplicates} no_id={no_id})")
    print(f"[build] train={len(train)} (general_mix={general_added}) val={len(val)} -> {out_dir}")
    print(f"[build] strata (language x mode):")
    for s in stats["strata"]:
        print(f"    {s['language']:>3} x {s['mode']:<15} total={s['total']:>5} "
              f"train={s['train']:>5} val={s['val']:>3}")
    print(f"[build] stats -> {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
