# Training environment — verified setup (D0 smoke test)

Date: 2026-10-02. Result: **both Qwen3-4B-Instruct-2507 and Qwen3.5-4B train with
LoRA (r=32, α=64, all-linear, bf16, gradient checkpointing) on a single RTX 3090.
No fallbacks were needed — release `transformers` already supports `qwen3_5`.**

## Environment spec (verified working)

Python 3.12.7, venv at `train/venv` (created with `uv venv train/venv --python 3.12`).

| package                 | version     |
|-------------------------|-------------|
| torch                   | 2.14.1+cu130 (default PyPI cu13 wheel, works on sm_86) |
| transformers            | **5.18.0** (release — `qwen3_5` supported, no git-main install needed) |
| peft                    | 0.21.2      |
| trl                     | 1.14.1      |
| accelerate              | 1.15.0      |
| datasets                | 5.0.1       |
| flash-linear-attention  | 0.5.2 (import name: `fla`) — optimized Gated DeltaNet kernels for Qwen3.5 |
| fla-core                | 0.5.2       |
| pillow                  | 12.3.0      |
| tokenizers              | 0.23.2      |

Driver 580.178.04 / CUDA 13.0, 2× RTX 3090 24GB.

Not installed: `causal-conv1d` (source build, needs nvcc, very slow to install).
Without it transformers falls back to the reference PyTorch conv1d impl — correct,
somewhat slower. `flash-linear-attention` (installed) covers the main DeltaNet kernels.

HF cache: `export HF_HOME=<repo>/hf_cache` (17 GB total: Qwen3-4B-Instruct-2507 ~7.5 GB,
Qwen3.5-4B ~9.5 GB, both full repos incl. vision weights for 3.5 — vision tensors are
inside the same safetensors shards, there are no separate vision files to exclude).
HF token comes from `../.env` (`set -a; source ../.env; set +a`).

## Model loading facts (empirical)

- **Qwen3-4B-Instruct-2507**: `AutoModelForCausalLM` → `Qwen3ForCausalLM`, `model_type=qwen3`,
  Hermes `<tool_call>{json}</tool_call>` template, `tools=` kwarg supported.
- **Qwen3.5-4B**: `AutoModelForCausalLM` → `Qwen3_5ForCausalLM` (**text-only class, 426
  tensors, 0 missing/unexpected keys** — transformers 5.18 remaps from the
  `Qwen3_5ForConditionalGeneration` checkpoint automatically; the vision tower is simply
  not loaded). Hybrid arch: 3× Gated DeltaNet (`linear_attention`) + 1× full attention,
  `model_type=qwen3_5_text`. Generates sanely ("The capital of France is Paris.").
  Template quirks (see smoke_lora.py `render_and_mask`):
  - `tool_calls[].function.arguments` must be a **dict**, not a JSON string
    (template iterates `arguments|items`; JSON string → `TemplateError`).
  - refuses system-only prefixes ("No user query found") — render from the first user turn.
  - tool-call format in rendered text is XML-ish `<function=name><parameter=...>` plus an
    empty `<think>\n\n</think>` block.
  Both Qwen templates regroup consecutive `tool` responses into one turn, which breaks
  naive prefix-diff masking; smoke_lora.py repairs this (masked common-prefix diff).

## How to run the smoke test

```bash
cd function_calling_edge_fine_tuning
set -a; source ../.env; set +a
export HF_HOME=$PWD/hf_cache

# one model per GPU, seq len 2048, 10 steps, LoRA r=32 α=64 dropout=0.05 all-linear
CUDA_VISIBLE_DEVICES=0 train/venv/bin/python train/smoke_lora.py \
    --model Qwen/Qwen3-4B-Instruct-2507 --device cuda:0
CUDA_VISIBLE_DEVICES=1 train/venv/bin/python train/smoke_lora.py \
    --model Qwen/Qwen3.5-4B --device cuda:1
```

`CUDA_VISIBLE_DEVICES` pinning matters: without it, any `Trainer`-based run wraps both
GPUs in `torch.nn.DataParallel` and crashes on device mismatch. The manual-loop smoke
script uses `device_map={"": device}` and is unaffected, but pin anyway.

Synthetic data: 8 hardcoded samples (geo/navigation theme), each
system → user → assistant(tool_calls) → tool result → assistant final; one sample has two
sequential tool calls. Loss on assistant turns only (manual label mask built from
incremental chat-template renders; falls back to final-turn-only → full-sequence if a
template isn't reconstructable). Fixed sample order, so steps 9-10 revisit samples 1-2 —
a like-for-like learning check.

## Per-model verdicts

### Qwen/Qwen3-4B-Instruct-2507 — WORKS, first try

- LoRA `all-linear`: 66.1M trainable / 4.09B (1.62%). Gradient checkpointing: on.
- Peak VRAM: **9.43 GiB allocated** (10.32 GiB reserved).
- Loss (lr 1e-4, bs 1): 1.42 → 0.27 → 0.56 → 0.19 → 0.04 → 0.02 → 0.06 → 0.02 →
  repeat-visits s0: 1.42→**0.27**, s1: 1.36→**0.12**. 2/2 repeat visits improved.

### Qwen/Qwen3.5-4B — WORKS, first try (with 3 documented TRL caveats)

- LoRA `all-linear` on the hybrid arch: works out of the box (peft discovers targets
  fine). 64.9M trainable / 4.27B (1.52%). Gradient checkpointing: on.
- Peak VRAM: **10.95 GiB allocated** (12.32 GiB reserved) at seq 2048.
- Loss: 0.27 → 0.32 → 0.08 → 0.02 → 0.004 → 0.0004 → 0.0002 → 0.007 → repeat-visits
  s0: 0.27→**0.13**, s1: 0.32→**0.20**. 2/2 repeat visits improved. (Model starts
  near-converged on this toy format — initial loss already ~0.1-0.3.)
- TRL SFTTrainer needed 3 workarounds (see `train/probe_trl.py`, 3-step probe passed,
  loss 0.23→0.06, token acc 0.92→0.97):
  1. `processing_class=tokenizer` **required** — TRL calls `AutoProcessor.from_pretrained`
     which, for unified-VL qwen3_5, pulls `Qwen2VLImageProcessor` →
     `ImportError: requires torchvision + pillow` (we installed pillow anyway).
  2. Assistant-only loss: TRL's `assistant_only_loss=True` needs `{% generation %}`
     template markers which Qwen templates lack → feed pre-tokenized
     `input_ids`/`labels` (masked) + custom collator instead.
  3. `CUDA_VISIBLE_DEVICES=<n>` pinning (DataParallel crash otherwise).

Blocked: nothing. Neither model needed the naive-target fallback, Trainer fallback,
or the no-gradient-checkpointing fallback.

## Resource usage

- `du -sh hf_cache` → **17 GB** (well under the 25 GB budget). Disk free after setup:
  ~51 GB on /home (venv ~7 GB + uv cache ~5.6 GB outside the repo).
- Only the 2 approved repos were downloaded; no stray snapshots (`hf_cache/hub/` has
  exactly the two `models--Qwen--*` entries).

## Recommended entrypoint for the real runs

Per-candidate, single GPU each (matches PLAN.md §3 recipe):

```bash
# candidate per 3090; lr 1e-4 cosine 3% warmup, 2-3 epochs, seq 8192, bf16 LoRA r32/α64 all-linear
CUDA_VISIBLE_DEVICES=0 train/venv/bin/python train/train_sft.py \
    --model Qwen/Qwen3-4B-Instruct-2507 --data data/final/train.jsonl --device cuda:0
CUDA_VISIBLE_DEVICES=1 train/venv/bin/python train/train_sft.py \
    --model Qwen/Qwen3.5-4B --data data/final/train.jsonl --device cuda:1
```

`train_sft.py` does not exist yet — build it from the verified pieces:

- Base it on **TRL SFTTrainer** (PLAN.md fallback framework) — it is verified working on
  BOTH models, provided the 3 Qwen3.5 workarounds above are baked in
  (copy `PadCollator` + `processing_class=` + masking from `train/probe_trl.py` /
  `train/smoke_lora.py`).
- Feed **pre-rendered `input_ids`/`labels`** (assistant-only mask via
  `smoke_lora.render_and_mask`) rather than raw text — this is the only verified
  assistant-only-loss path for both templates, and it transparently handles both the
  Hermes and qwen3_5 tool-call formats plus the dict-args/quk{} regroup quirks.
- Keep `attn_implementation="sdpa"`, `flash-linear-attention` installed; optionally try
  `causal-conv1d` before real runs (build takes a while; reference impl is correct but
  slower — throughput hit only, no correctness risk).
- Keep adapters only + `save_only_model`; prune `hf_cache` if disk tightens
  (51 GB free at time of writing).

## Logs

Raw logs: `train/smoke_qwen3_4b.log`, `train/smoke_qwen3_5_4b.log`,
`train/probe_trl_qwen3_4b.log`, `train/probe_trl_qwen3_5.log`.
