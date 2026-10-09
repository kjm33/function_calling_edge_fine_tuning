"""LoRA SFT entrypoint for the 5 edge candidates (TRL SFTTrainer, single pinned GPU).

Bakes in the verified caveats from train/README_TRAINING.md + train/probe_trl.py:
  1. CUDA_VISIBLE_DEVICES pinned to ONE GPU (--gpu 0|1) before torch import —
     otherwise Trainer DataParallel-wraps both 3090s and crashes.
  2. SFTTrainer(processing_class=tokenizer) — AutoProcessor on qwen3_5 (unified-VL)
     pulls Qwen2VLImageProcessor and ImportErrors.
  3. Assistant-only loss via pre-masked labels + custom collator (Qwen templates
     lack {% generation %} markers, so TRL's assistant_only_loss is unusable).
  4. NO packing (qwen3_5 hybrid linear-attention packing unsafe; uniform off).

Usage:
  train/venv/bin/python train/train_sft.py --candidate qwen3-4b-2507 \
      --train data/final/train.jsonl --val data/final/val.jsonl --gpu 0
  train/venv/bin/python train/train_sft.py --candidate qwen3.5-4b \
      --train data/final/train.jsonl --val data/final/val.jsonl --gpu 1
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pathlib
import random
import shutil
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "train" / "configs"
RUNS_DIR = ROOT / "train" / "runs"


def parse_args() -> argparse.Namespace:
    candidate_keys = sorted(p.stem for p in CONFIGS_DIR.glob("*.json"))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate", required=True, choices=candidate_keys,
                   help=f"student key (config in train/configs/<key>.json): {candidate_keys}")
    p.add_argument("--train", required=True, help="train jsonl (neutral format)")
    p.add_argument("--val", required=True, help="val jsonl (neutral format)")
    p.add_argument("--out-dir", default=None,
                   help="default: train/runs/<candidate>")
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--tool-cap", type=int, default=1600,
                   help="max chars per tool-result payload in training render "
                        "(head+tail kept); 0 disables shrinking")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--batch-tokens", type=int, default=8192,
                   help="token budget per optimizer step; sets grad-accum when --grad-accum auto")
    p.add_argument("--micro-bs", type=int, default=1, help="per-device batch size")
    p.add_argument("--grad-accum", default="auto",
                   help="'auto' = max(1, batch-tokens/(micro-bs*seq-len)) or an explicit int")
    p.add_argument("--gpu", type=int, default=0, choices=[0, 1],
                   help="which physical GPU to pin via CUDA_VISIBLE_DEVICES")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--patience", type=int, default=3,
                   help="early-stopping patience on eval_loss")
    p.add_argument("--attn", default="auto",
                   choices=["auto", "flash_attention_2", "sdpa", "eager"],
                   help="auto = try config attn_candidates in order, silent fallback")
    p.add_argument("--min-keep-ratio", type=float, default=0.7,
                   help="drop rows whose truncation would cut more than (1-this) of train tokens")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="wipe a non-empty out-dir instead of erroring (resume still wins if checkpoints exist)")
    return p.parse_args()


def load_env_file(path: pathlib.Path) -> None:
    """Tiny KEY=VALUE parser (no python-dotenv dependency in the train venv).
    Only fills variables that are not already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("'\"")
        if k and v:
            os.environ.setdefault(k, v)


class Tee:
    """Redirect a stream to stdout/stderr AND a log file (tee-style)."""

    def __init__(self, stream, file_handle):
        self.stream = stream
        self.fh = file_handle

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)

    def flush(self):
        self.stream.flush()
        self.fh.flush()


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------------------
# Fallback renderer — vendored from train/smoke_lora.py (verified on both Qwens).
# FALLBACK ONLY: replace when src/render_templates.py lands; the entrypoint prefers the
# render_templates contract (render_segments(messages, tools, fmt, model_id)) and only
# uses this if that import fails. Produces the same segment shape:
# [{"text": str, "train": bool}], joined texts == full rendered conversation,
# train=True only on assistant tokens.
# --------------------------------------------------------------------------------------
FALLBACK_MARKER = "built-in fallback renderer (vendored from smoke_lora.py) — replace when src/render_templates.py lands"


def _args_to_dicts(messages):
    out = copy.deepcopy(messages)
    for msg in out:
        for call in msg.get("tool_calls") or []:
            args = call["function"].get("arguments")
            if isinstance(args, str):
                call["function"]["arguments"] = json.loads(args)
    return out


def render_segments_fallback(tokenizer, messages, tools):
    def rend(msgs):
        return tokenizer.apply_chat_template(
            msgs, tools=tools, tokenize=False, add_generation_prompt=False
        )

    def build(msgs):
        segments = []
        prev = ""
        started = False
        for k in range(1, len(msgs) + 1):
            try:
                r = rend(msgs[:k])
            except Exception:
                if msgs[k - 1]["role"] == "assistant":
                    return None
                continue  # unrenderable prefix (e.g. system-only for Qwen3.5)
            if not started:
                segments.append((r, msgs[k - 1]["role"] == "assistant"))
            elif r.startswith(prev):
                segments.append((r[len(prev):], msgs[k - 1]["role"] == "assistant"))
            else:
                # template regrouped earlier turns (consecutive tool responses):
                # trim the invalid tail of the previous segment (always a masked
                # tool turn here, else bail) and mask the common-prefix diff
                if msgs[k - 1]["role"] == "assistant":
                    return None
                cp = 0
                for a, b in zip(prev, r):
                    if a != b:
                        break
                    cp += 1
                drop = len(prev) - cp
                last_text = segments[-1][0]
                if drop > len(last_text):
                    return None
                segments[-1] = (last_text[: len(last_text) - drop], False)
                segments.append((r[cp:], False))
            prev = r
            started = True
        if not started or "".join(t for t, _ in segments) != rend(msgs):
            return None
        return segments

    segments = build(messages)
    if segments is None:
        segments = build(_args_to_dicts(messages))
    if segments is None:
        # final fallback: loss on the whole final assistant turn only
        try:
            r_prev = rend(messages[:-1])
            r_full = rend(messages)
        except Exception:
            r_prev, r_full = None, None
        if r_prev is None or not r_full.startswith(r_prev):
            m = _args_to_dicts(messages)
            r_prev, r_full = rend(m[:-1]), rend(m)
        segments = [(r_prev, False), (r_full[len(r_prev):], True)]
    return [{"text": t, "train": tr} for t, tr in segments]


def load_renderer(candidate_cfg, tokenizer):
    """Prefer the src/render_templates.py contract; fall back to the vendored
    smoke_lora renderer (qwen dict-args + regroup quirks handled)."""
    fmt, model_id = candidate_cfg["fmt"], candidate_cfg["model_id"]
    try:
        sys.path.insert(0, str(ROOT))
        from src.render_templates import FORMATS, render_segments  # noqa: PLC0415

        if fmt not in FORMATS:
            raise ValueError(f"fmt {fmt!r} not in FORMATS={sorted(FORMATS)}")
        path = "contract"

        def render(messages, tools):
            return render_segments(messages, tools, fmt=fmt, model_id=model_id)

    except Exception as e:  # ImportError, broken module, missing fmt — any of these
        log(f"[render] render_templates contract unavailable ({type(e).__name__}: "
            f"{str(e)[:200]}) — using {FALLBACK_MARKER}")
        path = "fallback"

        def render(messages, tools):
            return render_segments_fallback(tokenizer, messages, tools)

    return path, render


def tokenize_segments(tokenizer, segments, seq_len, eos_id):
    """Tokenize the FULL rendered text once and derive the label mask from
    character offsets (segment char spans). This is immune to segment
    boundaries splitting special tokens (phi-4 <|assistant|>, llama
    <|start_header_id|>): a token is trainable only if its char span lies
    wholly inside a trainable segment; straddling tokens are masked.
    The previous cumulative-prefix diff approach rejected rows whenever a
    boundary merge touched already-emitted train tokens (phi: 74% of rows,
    incl. EVERY functools-bearing one). Appends eos to ids if the render does
    not already end with it. Returns (ids, labels, info) or (None, reason, info)
    if the row is unusable."""
    full = "".join(s["text"] for s in segments)
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]

    ranges, pos = [], 0
    for s in segments:
        n = len(s["text"])
        if n:
            ranges.append((pos, pos + n, bool(s["train"])))
        pos += n

    labels: list[int] = []
    ri = 0
    for tid, (a, b) in zip(ids, offs):
        while ri < len(ranges) and ranges[ri][1] <= a:
            ri += 1
        trainable = False
        if ri < len(ranges):
            st, en, tr = ranges[ri]
            trainable = tr and st <= a and b <= en
        labels.append(tid if trainable else -100)

    n_train = sum(1 for x in labels if x != -100)
    if n_train == 0:
        return None, "row has no trainable (assistant) tokens", {}

    # per template conventions: append eos only if the last TRAIN segment does not
    # already end with the end-of-turn token (Qwen templates end with <|im_end|>\n;
    # gemma ends with <end_of_turn>\n so <eos> is appended after it)
    last_train_text = next((s["text"] for s in reversed(segments) if s["train"]), "")
    eos_str = tokenizer.eos_token or ""
    if eos_id is not None and eos_str and not last_train_text.rstrip().endswith(eos_str):
        ids.append(eos_id)
        labels.append(eos_id if segments[-1]["train"] else -100)
        n_train += 1 if segments[-1]["train"] else 0

    info = {"tokens": len(ids), "train_tokens": n_train, "truncated": False}
    if len(ids) > seq_len:
        kept_train = sum(1 for x in labels[:seq_len] if x != -100)
        if kept_train / n_train < args_min_keep_ratio[0]:
            return None, (f"truncation would cut train tokens to {kept_train}/{n_train} "
                          f"(< {args_min_keep_ratio[0]:.0%})"), info
        ids, labels = ids[:seq_len], labels[:seq_len]
        info.update(truncated=True, tokens=len(ids),
                    train_tokens=sum(1 for x in labels if x != -100))
    return ids, labels, info


# filled by main() so tokenize_segments can see --min-keep-ratio without threading it
args_min_keep_ratio = [0.7]


def ensure_weights(model_id: str) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        snap = snapshot_download(
            model_id, local_files_only=True,
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
        )
    except LocalEntryNotFoundError:
        sys.exit(
            f"FATAL: weights for {model_id} not found in HF cache "
            f"(HF_HOME={os.environ.get('HF_HOME')}).\n"
            f"Pre-download them first:\n"
            f"  HF_HOME={os.environ.get('HF_HOME')} train/venv/bin/huggingface-cli download {model_id}\n"
            f"(gated repos also need HF_TOKEN in .env)"
        )
    if not any(pathlib.Path(snap).glob("*.safetensors")):
        sys.exit(
            f"FATAL: cache for {model_id} has no *.safetensors (config-only snapshot).\n"
            f"Pre-download weights:\n"
            f"  HF_HOME={os.environ.get('HF_HOME')} train/venv/bin/huggingface-cli download {model_id}"
        )


def load_jsonl(path: pathlib.Path):
    rows = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"FATAL: bad jsonl in {path} line {i + 1}: {e}")
    if not rows:
        raise SystemExit(f"FATAL: empty dataset {path}")
    return rows


def _shrink_tool_results(messages, cap):
    """Compress oversized tool-result payloads (training-only, never trained
    tokens — context only) so long dialogs fit seq_len. Keeps head+tail."""
    if not cap:
        return messages, 0
    out, n_shrunk = [], 0
    for m in messages:
        c = m.get("content")
        if m.get("role") == "tool" and isinstance(c, str) and len(c) > cap:
            head, tail = int(cap * 0.7), int(cap * 0.25)
            cut = len(c) - head - tail
            m = {**m, "content": c[:head] + f"\n…[{cut} chars truncated for training]…\n" + c[-tail:]}
            n_shrunk += 1
        out.append(m)
    return out, n_shrunk


def build_dataset_rows(tokenizer, rows, render, seq_len, eos_id, split_name,
                       tool_cap=1600):
    feats, stats = [], {"rows": len(rows), "kept": 0, "skipped_render": 0,
                        "dropped_truncation": 0, "truncated": 0,
                        "tokens": 0, "train_tokens": 0, "shrunk_tool_msgs": 0}
    for i, row in enumerate(rows):
        msgs, n_shrunk = _shrink_tool_results(row["messages"], tool_cap)
        stats["shrunk_tool_msgs"] += n_shrunk
        try:
            segments = render(msgs, row.get("tools") or [])
        except Exception as e:
            log(f"[data:{split_name}] row {i} ({row.get('dialog_id', '?')}) render failed: "
                f"{type(e).__name__}: {str(e)[:160]} — skipping")
            stats["skipped_render"] += 1
            continue
        ids, labels, info = tokenize_segments(tokenizer, segments, seq_len, eos_id)
        if ids is None:
            log(f"[data:{split_name}] row {i} ({row.get('dialog_id', '?')}) dropped: {labels}")
            stats["dropped_truncation" if "truncation" in labels else "skipped_render"] += 1
            continue
        stats["kept"] += 1
        stats["tokens"] += info["tokens"]
        stats["train_tokens"] += info["train_tokens"]
        stats["truncated"] += int(info["truncated"])
        feats.append({"input_ids": ids, "labels": labels})
    log(f"[data:{split_name}] {stats['kept']}/{stats['rows']} rows kept "
        f"({stats['dropped_truncation']} dropped on truncation, "
        f"{stats['skipped_render']} render failures, "
        f"{stats['shrunk_tool_msgs']} tool payloads shrunk), "
        f"tokens={stats['tokens']}, train_tokens={stats['train_tokens']}")
    if not feats:
        raise SystemExit(f"FATAL: no usable rows in {split_name} split")
    return feats, stats


def find_latest_checkpoint(out_dir: pathlib.Path):
    ckpts = []
    for d in out_dir.glob("checkpoint-*"):
        try:
            ckpts.append((int(d.name.split("-")[-1]), d))
        except ValueError:
            continue
    return max(ckpts)[1] if ckpts else None


def main():
    args = parse_args()
    args_min_keep_ratio[0] = args.min_keep_ratio

    # --- environment BEFORE any torch/transformers import -----------------------
    load_env_file(ROOT / ".env")
    os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    out_dir = pathlib.Path(args.out_dir or (RUNS_DIR / args.candidate)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    log_fh = open(log_path, "a", buffering=1)
    sys.stdout = Tee(sys.stdout, log_fh)
    sys.stderr = Tee(sys.stderr, log_fh)

    t_start = time.time()
    log(f"[run] candidate={args.candidate} gpu={args.gpu} (CUDA_VISIBLE_DEVICES="
        f"{os.environ['CUDA_VISIBLE_DEVICES']}) out={out_dir}")
    log(f"[run] HF_HOME={os.environ['HF_HOME']}")

    cfg_path = CONFIGS_DIR / f"{args.candidate}.json"
    cfg = json.loads(cfg_path.read_text())
    model_id = cfg["model_id"]
    log(f"[cfg] {model_id} fmt={cfg['fmt']} attn_candidates={cfg.get('attn_candidates')}")

    grad_accum = (max(1, round(args.batch_tokens / (args.micro_bs * args.seq_len)))
                  if str(args.grad_accum) == "auto" else int(args.grad_accum))
    log(f"[bs] micro_bs={args.micro_bs} grad_accum={grad_accum} "
        f"(token budget/step ≈ {args.micro_bs * grad_accum * args.seq_len} of "
        f"{args.batch_tokens} requested)")

    # --- heavy imports (after CUDA_VISIBLE_DEVICES pin) -------------------------
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              EarlyStoppingCallback)
    from trl import SFTConfig, SFTTrainer

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    ensure_weights(model_id)

    # --- tokenizer + renderer ---------------------------------------------------
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=cfg.get("trust_remote_code", False))
    eos_id = tok.eos_token_id
    render_path, render = load_renderer(cfg, tok)
    log(f"[render] path={render_path}")

    # --- data -------------------------------------------------------------------
    train_rows = load_jsonl(pathlib.Path(args.train).resolve())
    val_rows = load_jsonl(pathlib.Path(args.val).resolve())
    train_feats, train_stats = build_dataset_rows(tok, train_rows, render, args.seq_len, eos_id, "train", args.tool_cap)
    val_feats, val_stats = build_dataset_rows(tok, val_rows, render, args.seq_len, eos_id, "val", args.tool_cap)
    train_ds = Dataset.from_list(train_feats)
    val_ds = Dataset.from_list(val_feats)

    # --- model ------------------------------------------------------------------
    candidates = [args.attn] if args.attn != "auto" else cfg.get("attn_candidates", ["sdpa"])
    if candidates[0] == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except Exception:
            log("[attn] flash-attn package not installed — using sdpa")
            candidates = ["sdpa"]

    model = None
    for i, attn in enumerate(candidates):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.bfloat16, attn_implementation=attn,
                trust_remote_code=cfg.get("trust_remote_code", False),
            )
            log(f"[attn] {attn} ok")
            break
        except Exception as e:
            log(f"[attn] {attn} failed ({type(e).__name__}: {str(e)[:200]})")
            if i + 1 < len(candidates):
                continue
            log("[attn] falling back to sdpa")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
                trust_remote_code=cfg.get("trust_remote_code", False),
            )
    if model is None:
        sys.exit(f"FATAL: could not load {model_id}")
    model.config.use_cache = False

    # --- LoRA + gradient checkpointing (probe_trl.py-verified recipe) ------------
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM, target_modules="all-linear",
    )
    try:
        model = get_peft_model(model, lora)
        lora_mode = "all-linear"
    except Exception as e:
        names = sorted({n.split(".")[-1] for n, m in model.named_modules()
                        if isinstance(m, torch.nn.Linear) and n.split(".")[-1] != "lm_head"})
        log(f"[lora] all-linear failed ({type(e).__name__}: {str(e)[:200]}); "
            f"naive targets: {names}")
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type=TaskType.CAUSAL_LM, target_modules=names))
        lora_mode = "naive-list"
    trainable, total = model.get_nb_trainable_parameters()
    log(f"[lora] {lora_mode} r={args.lora_r} a={args.lora_alpha} "
        f"trainable={trainable / 1e6:.1f}M / {total / 1e9:.2f}B ({100 * trainable / total:.2f}%)")
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    log("[grad-ckpt] enabled (use_reentrant=False)")

    # --- schedule ----------------------------------------------------------------
    if args.max_steps > 0:
        total_updates = args.max_steps
    else:
        steps_per_epoch = math.ceil(len(train_ds) / args.micro_bs)
        total_updates = math.ceil(steps_per_epoch / grad_accum) * math.ceil(args.epochs)
    warmup_steps = max(1, round(args.warmup_ratio * total_updates))
    save_steps = args.save_steps
    if save_steps % args.eval_steps != 0:
        save_steps = args.eval_steps * max(1, round(save_steps / args.eval_steps))
        log(f"[sched] save_steps realigned to {save_steps} (multiple of eval_steps, "
            f"required by load_best_model_at_end)")
    log(f"[sched] total_updates≈{total_updates} warmup_steps={warmup_steps} "
        f"eval_steps={args.eval_steps} save_steps={save_steps} patience={args.patience}")

    # --- resume / out-dir hygiene -------------------------------------------------
    resume_path = None
    if out_dir.exists():
        latest = find_latest_checkpoint(out_dir)
        if latest is not None and not args.force:
            resume_path = latest
            log(f"[resume] found checkpoint {latest.name} — resuming (use --force to restart)")
        elif any(p for p in out_dir.iterdir() if p.name != log_path.name) and not args.force:
            sys.exit(f"FATAL: {out_dir} exists and is non-empty without checkpoints. "
                     f"Pass --force to overwrite.")

    sft_args = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.micro_bs,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=args.micro_bs,
        max_length=args.seq_len,
        packing=False,  # qwen3_5 hybrid linear attention: packing unsafe — keep off
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=5,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim="adamw_torch",
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        disable_tqdm=True,
        report_to=[],
    )

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    class PadCollator:
        """Pads input_ids with pad token, labels with -100. No packing."""

        def __call__(self, features):
            maxlen = max(len(f["input_ids"]) for f in features)
            ids, labels, attn = [], [], []
            for f in features:
                n = maxlen - len(f["input_ids"])
                ids.append(list(f["input_ids"]) + [pad_id] * n)
                labels.append(list(f["labels"]) + [-100] * n)
                attn.append([1] * len(f["input_ids"]) + [0] * n)
            return {"input_ids": torch.tensor(ids),
                    "labels": torch.tensor(labels),
                    "attention_mask": torch.tensor(attn)}

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,  # NOT AutoProcessor: qwen3_5 unified-VL pulls an image processor
        data_collator=PadCollator(),
    )
    if total_updates >= args.eval_steps:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.patience))

    torch.cuda.reset_peak_memory_stats()
    try:
        trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
    except Exception as e:
        log(f"[FATAL] training crashed: {type(e).__name__}: {e}")
        raise

    # --- post-training -------------------------------------------------------------
    eval_entries = [{"step": h["step"], "eval_loss": h["eval_loss"]}
                    for h in trainer.state.log_history if "eval_loss" in h]
    train_entries = [{"step": h["step"], "loss": h["loss"]}
                     for h in trainer.state.log_history if "loss" in h]
    all_finite = all(math.isfinite(e["loss"]) for e in train_entries) and \
        all(math.isfinite(e["eval_loss"]) for e in eval_entries)
    best_ckpt = trainer.state.best_model_checkpoint

    trainer.save_model()  # best adapter (post load_best_model_at_end) → out_dir root
    tok.save_pretrained(str(out_dir))
    adapter_file = out_dir / "adapter_model.safetensors"
    adapter_mb = adapter_file.stat().st_size / 2**20 if adapter_file.exists() else 0.0
    ckpt_bytes = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())

    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    runtime = trainer.state.log_history[-1].get("train_runtime", time.time() - t_start)
    steps_per_s = trainer.state.global_step / runtime if runtime else 0.0

    def mtime(p):
        try:
            return round(p.stat().st_mtime, 3)
        except OSError:
            return None

    meta = {
        "candidate": args.candidate,
        "candidate_cfg": cfg,
        "args": {k: v for k, v in vars(args).items()},
        "derived": {"grad_accum": grad_accum, "warmup_steps": warmup_steps,
                    "save_steps": save_steps, "attn_used": getattr(model.config, "_attn_implementation", "?"),
                    "lora_mode": lora_mode, "trainable_params": trainable,
                    "render_path": render_path, "eos_appended_if_missing": True},
        "env": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                "trl": __import__("trl").__version__, "peft": __import__("peft").__version__,
                "datasets": __import__("datasets").__version__,
                "accelerate": __import__("accelerate").__version__,
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]},
        "version_stamp": {"train_sft.py.mtime": mtime(pathlib.Path(__file__)),
                          "config.mtime": mtime(cfg_path),
                          "render_templates.mtime": mtime(ROOT / "src" / "render_templates.py")},
        "data": {"train": train_stats, "val": val_stats,
                 "train_path": str(pathlib.Path(args.train).resolve()),
                 "val_path": str(pathlib.Path(args.val).resolve())},
        "final_step": trainer.state.global_step,
        "train_loss_curve": train_entries,
        "eval_loss_curve": eval_entries,
        "best_checkpoint": best_ckpt,
        "final_train_loss": train_entries[-1]["loss"] if train_entries else None,
        "best_eval_loss": min((e["eval_loss"] for e in eval_entries), default=None),
        "vram_peak_gib": round(peak, 2),
        "vram_reserved_gib": round(reserved, 2),
        "train_runtime_s": round(runtime, 1),
        "steps_per_sec": round(steps_per_s, 3),
        "adapter_mb": round(adapter_mb, 1),
        "out_dir_bytes": ckpt_bytes,
        "all_losses_finite": all_finite,
        "ok": bool(all_finite and eval_entries and adapter_file.exists()),
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    log(f"[final] step={trainer.state.global_step} train_loss={meta['final_train_loss']} "
        f"best_eval_loss={meta['best_eval_loss']} best_ckpt={best_ckpt}")
    log(f"[final] eval_curve={[(e['step'], round(e['eval_loss'], 4)) for e in eval_entries]}")
    log(f"[final] vram_peak={peak:.2f}GiB reserved={reserved:.2f}GiB "
        f"runtime={runtime:.0f}s steps/s={steps_per_s:.2f} adapter={adapter_mb:.1f}MB")
    log(f"TRAIN RESULT: {'OK' if meta['ok'] else 'MARGINAL'}")


if __name__ == "__main__":
    main()
