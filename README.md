# Function-Calling Edge Fine-Tuning

Fine-tune small (≤4B) local LLMs for **function calling over geo/map MCP-style tools**
(TomTom, HERE, OSRM/Nominatim, Open-Meteo) in 5 languages (en/de/pl/fr/es), for an
in-car navigation assistant. Five candidates are trained with LoRA, evaluated
zero-shot and fine-tuned on the same execution-based gold set, and compared in
`reports/RESULTS.md`.

## Pipeline

```
dataset_gen.py ──▶ raw_trajectories/ ──▶ verify.py ──▶ verified/ ──▶ build_dataset.py ──▶ final/ (train/val)
   (teacher LLMs +      multi-turn agent     (schema / exec-          (dedup + stratified
    live tool calls)     dialogs w/ tools     replay / judges)         lang×mode split)
                                                                              │
                                                              render_templates.py (5 formats)
                                                                              │
                                              train_sft.py (TRL LoRA) ──▶ runs/<candidate>
                                                                              │
                                              build_gold.py + harness.py (vLLM, judge-free,
                                              execution-based scoring) ──▶ eval/results/
```

- **Generation** (`src/dataset_gen.py`): APIGen-style multi-turn dialogs with live tool
  execution; 6 teachers via OpenRouter; 5 languages × multiple modes; weak dialogs re-rolled.
- **Verification** (`src/verify.py`): structural/schema checks, execution replay, secrets
  scan, language check, dual judge (qwen3.5-9b + deepseek-v4-flash, both ≥7.0 avg),
  MMR near-dup filtering. Judge pass ≠ truth — the execution-based gold eval decides.
- **Training** (`train/train_sft.py`): TRL + LoRA, one candidate per GPU, auto-resume,
  per-model chat/tool-call render formats.
- **Eval** (`eval/`): vLLM serving with per-candidate tool-call parsers; scores tool
  selection (strict/set), argument accuracy, and execution success against replay-verified
  reference calls. Zero-shot baseline + FT result on the same gold set, temp 0.

## Layout

| Path | Contents |
|---|---|
| `src/` | Pipeline: config, MCP client, tool registry (21 tools), dataset gen, verify, LLM client (dual-key round-robin), render templates, dataset build |
| `mcp/` | HERE + keyless REST tool wrappers (keeps `mcp/__init__.py` — fixes vLLM package collision) |
| `data/` | `raw_trajectories/`, `verified/`, `final/` (generated, gitignored); `final_stripped/` (portable, committed); gold test sets |
| `train/` | `train_sft.py`, `configs/<candidate>.json`, `runs/` (adapters, gitignored), smoke tests |
| `eval/` | `build_gold.py`, `harness.py`, `results/SUMMARY.md` (running results table) |
| `scripts/` | Runbooks, watchdogs, eval screen, `strip_dataset.py` |
| `reports/` | RESEARCH.md, VERIFY_PILOT.md, PILOT_QUALITY.md, RESULTS.md (final comparison) |
| `plans/PLAN.md` | Approved project plan |

## Candidates

| candidate | model_id | tool-call format | vLLM parser |
|---|---|---|---|
| qwen3.5-4b | Qwen/Qwen3.5-4B | native, DICT args, thinking disabled | `qwen3_coder` + `--reasoning-parser qwen3` |
| qwen3-4b-2507 | Qwen/Qwen3-4B-Instruct-2507 | native, JSON-string args | `hermes` |
| llama-3.2-3b | unsloth/Llama-3.2-3B-Instruct | native llama3, 1 call/turn max | `llama3_json` |
| phi-4-mini | microsoft/Phi-4-mini-instruct | tools in system msg, `functools[...]` | `phi4_mini_json` |
| gemma-3-4b | unsloth/gemma-3-4b-it | Hermes transplant (no native FC) | `hermes` |

## Results snapshot

Best fine-tuned runs on the 643-example gold set (full history in `eval/results/SUMMARY.md`,
final comparison in `reports/RESULTS.md`):

| candidate | FT run | exec_ok | sel_strict | sel_set | arg_acc |
|---|---|---|---|---|---|
| llama-3.2-3b | final2k | **0.895** | **0.304** | **0.463** | 0.443 |
| gemma-3-4b | final2k | 0.720 | 0.266 | 0.392 | **0.475** |

OOD generalization (297 unseen-tool examples): gemma-3-4b sel_strict 0.569 vs llama-3.2-3b 0.404.

## Portable dataset

`data/final_stripped/` contains the verified train/val dialogs with full tool schemas
replaced by `tool_names` lists (233MB → 42MB) — self-contained and reusable in other
projects. Regenerate with:

```bash
.venv/bin/python scripts/strip_dataset.py --in data/final/train.jsonl data/final/val.jsonl \
  --out-dir data/final_stripped
```

## Getting started

```bash
# 1. env: keys in .env (symlink ../.env); main venv has httpx only
cp .env.example .env  # or point the symlink at your keys file

# 2. generate raw dialogs (append-mode; background with setsid nohup)
PATH=$HOME/tools/node22/bin:$PATH PYTHONPATH=. .venv/bin/python -m src.dataset_gen \
  --n 6000 --out data/raw_trajectories/gen_main.jsonl --workers 12 \
  --teachers google/gemma-4-26b-a4b-it,qwen/qwen3.5-9b,z-ai/glm-5.3-flash,deepseek/deepseek-v4-flash,mistralai/mistral-nemo,openai/gpt-oss-20b

# 3. verify (dual judge; both >=7.0 avg, no dim <5)
PYTHONPATH=. .venv/bin/python -m src.verify --in data/raw_trajectories/gen_main.jsonl \
  --out data/verified/gen_main_verified.jsonl --summary data/verified/gen_main_summary.json

# 4. build train/val split
.venv/bin/python -m src.build_dataset --in data/verified/*_verified.jsonl --out-dir data/final

# 5. train (train/venv has torch+trl+peft; one candidate per GPU, auto-resume)
train/venv/bin/python train/train_sft.py --candidate qwen3.5-4b \
  --train data/final/train.jsonl --val data/final/val.jsonl --gpu 1

# 6. gold set + eval (eval/venv has vLLM nightly; execution-based, judge-free)
.venv/bin/python -m eval.build_gold --n 700 --out data/test_gold.jsonl
eval/venv/bin/python -m eval.harness --candidate qwen3.5-4b --tag both \
  --gold data/test_gold.jsonl --gpu 0
```

## Environments

- `.venv/` — main pipeline (httpx only)
- `train/venv/` — torch, transformers, trl, peft
- `eval/venv/` — vLLM nightly (serving + eval)
- TomTom MCP needs Node ≥22 (`PATH=$HOME/tools/node22/bin:$PATH`)
- HF cache at `/var/tmp/fcft_hf_cache` (symlink `hf_cache`); use `hf download` (not `huggingface-cli`)

## Notes & gotchas

- Data files are **append-mode**; never rewrite raw/verified files.
- HERE/TomTom API keys leak into error URLs — always run the verify secrets stage.
- `vllm serve` takes the BASE model id (LoRA is request-time `--enable-lora`);
  needs `PATH=eval/venv/bin:$PATH` (ninja) and `--max-model-len 32768`.
- Kill vLLM by PID (`pgrep VLLM::EngineCore`), not `pkill -f` patterns.
- Quality > quantity: ~4-5k verified dialogs for a focused 21-tool domain.
- vLLM batching is nondeterministic → expect ±5pp run variance on small n.

See `AGENTS.md` for the full agent guide, `train/README_TRAINING.md` and
`eval/README_EVAL.md` for component details, and `plans/PLAN.md` for the approved plan.
