# AGENTS.md — Guide for AI agents working in this repo

Fine-tune small (≤4B) local LLMs for **function calling over geo/map MCP-style tools**
(TomTom, HERE, OSRM/Nominatim, Open-Meteo) in 5 languages (en/de/pl/fr/es), for an
in-car navigation assistant. Compare 5 candidates, pick a winner, report in `reports/RESULTS.md`.

## Layout

- `src/` — pipeline: `config.py` (env, teachers, judges, students), `mcp_client.py` (stdio MCP),
  `tool_registry.py` (21 tools: TomTom MCP + HERE REST + keyless REST, arg-keyed response cache),
  `dataset_gen.py` (APIGen-style dialog generation w/ live tool calls), `verify.py`
  (structural/schema/exec-replay/secrets/language/dual-judge/MMR), `llm_client.py`
  (OpenAI + OpenRouter, dual-key round-robin, dead-key circuit breaker),
  `render_templates.py` (5 per-model formats), `build_dataset.py` (verified → train/val split)
- `mcp/` — HERE + keyless REST tool wrappers (`mcp/__init__.py` MUST stay: fixes vLLM `mcp` pkg collision)
- `data/` — `raw_trajectories/` (generated), `verified/`, `final/`, `test_gold*.jsonl`, `tool_response_cache.json`
- `train/` — `venv/` (torch+transformers 5.18+trl+peft), `train_sft.py` (TRL LoRA entrypoint),
  `configs/<candidate>.json`, `runs/<name>/`, `smoke_lora.py`
- `eval/` — `venv/` (vLLM nightly), `build_gold.py` (gold set w/ replay-verified reference calls),
  `harness.py` (serve + exec-based scoring), `results/SUMMARY.md` (running results table)
- `reports/` — RESEARCH.md, VERIFY_PILOT.md (overwritten by each verify run), PILOT_QUALITY.md, RESULTS.md (final)
- `plans/PLAN.md` — approved plan. `.env` → symlink to `../.env` (all API keys).

## Environments

- Main venv: `.venv/` (httpx only). Training: `train/venv/`. Eval/serving: `eval/venv/` (vLLM nightly).
- TomTom MCP needs Node ≥22: `PATH=$HOME/tools/node22/bin:$PATH` before any tool-using run.
- HF cache lives at `/var/tmp/fcft_hf_cache` (symlink `hf_cache` → it). Keep /home lean; adapters only.
- `huggingface-cli` is DEAD in new hub versions — use `hf download`.
- Model repos: `unsloth/Llama-3.2-3B-Instruct`, `unsloth/gemma-3-4b-it` (google+meta gated/denied),
  others official. Tokenizer-only downloads ~8MB via allow_patterns.

## Key commands

```bash
# generate raw dialogs (append-mode; background: setsid nohup ... > logs/x.log 2>&1 < /dev/null &)
PATH=$HOME/tools/node22/bin:$PATH PYTHONPATH=. .venv/bin/python -m src.dataset_gen \
  --n 6000 --out data/raw_trajectories/gen_main.jsonl --workers 12 \
  --teachers google/gemma-4-26b-a4b-it,qwen/qwen3.5-9b,z-ai/glm-5.3-flash,deepseek/deepseek-v4-flash,mistralai/mistral-nemo,openai/gpt-oss-20b

# verify (dual judge: qwen3.5-9b + deepseek-v4-flash; both >=7.0 avg, no dim <5)
PYTHONPATH=. .venv/bin/python -m src.verify --in data/raw_trajectories/gen_main.jsonl \
  --out data/verified/gen_main_verified.jsonl --summary data/verified/gen_main_summary.json

# build neutral dataset (train/val, stratified lang×mode)
.venv/bin/python -m src.build_dataset --in data/verified/*_verified.jsonl --out-dir data/final

# train (one candidate per GPU; auto-resume; --force wipes pre-created dirs — else FATAL)
train/venv/bin/python train/train_sft.py --candidate qwen3.5-4b --train data/final/train.jsonl \
  --val data/final/val.jsonl --gpu 1

# gold set + eval (judge-free, execution-based: this is the final arbiter)
.venv/bin/python -m eval.build_gold --n 700 --out data/test_gold.jsonl
eval/venv/bin/python -m eval.harness --candidate qwen3.5-4b --tag both --gold data/test_gold.jsonl --gpu 0
```

## Candidates & formats (see train/configs/, eval README)

| candidate | model_id | tool-call format | vLLM parser |
|---|---|---|---|
| qwen3.5-4b | Qwen/Qwen3.5-4B | native, DICT args, thinking disabled both train+serve | `qwen3_coder` + `--reasoning-parser qwen3` + `--language-model-only` |
| qwen3-4b-2507 | Qwen/Qwen3-4B-Instruct-2507 | native, JSON-string args | `hermes` |
| llama-3.2-3b | unsloth/Llama-3.2-3B-Instruct | native llama3, 1 call/turn max, DICT args in template | `llama3_json` |
| phi-4-mini | microsoft/Phi-4-mini-instruct | tools in system msg, `functools[...]` output | `phi4_mini_json` |
| gemma-3-4b | unsloth/gemma-3-4b-it | Hermes TRANSPLANT (no native FC; strict user/model alternation, tool results fold into user turns) | `hermes` |

## Gotchas (hard-won)

- **OpenAI credits exhausted on both keys** (circuit breaker handles it). Teachers/judges = OpenRouter only.
  glm-5.3-flash CANNOT be a judge (mandatory reasoning → 400). gemma-4 as judge = rubber stamp.
- eval harness: `vllm serve` takes the BASE model id (LoRA is request-time `--enable-lora` name);
  needs `PATH=eval/venv/bin:$PATH` (ninja for torch.compile); servers run `--max-model-len 32768`.
- qwen3.5 request bodies: tool_call arguments must be JSON STRINGS over the wire (harness normalizes).
- Qwen3.5 training: `processing_class=tokenizer` (AutoProcessor pulls image processor → ImportError),
  no packing (hybrid arch), pre-masked labels + custom collator (templates lack {% generation %}).
- Kill vLLM by PID (`pgrep VLLM::EngineCore` or port); `pkill -f` patterns can hang the shell.
- HERE APIs resolve ONLY under `*.hereapi.com`. HERE/TomTom keys leak into error URLs → always run
  verify.py secrets stage; `_redact()` already guards new gen output.
- Data files are APPEND-mode; never rewrite raw/verified files. `build_gold` dedups via `data/gold_seed_ids.json`.
- dataset_gen acceptance gate re-rolls weak dialogs (≤3 attempts); judge pass ≠ final truth —
  the **execution-based gold eval decides**.

## Conventions

- Background jobs: `setsid nohup ... </dev/null &`, logs in `logs/`, monitor with `tail`.
- Reports: append results rows to `eval/results/SUMMARY.md`; final comparison → `reports/RESULTS.md`.
- Quality>quantity (APIGen/ToolACE-R lesson): ~4-5k verified dialogs for a focused 21-tool domain.
- Zero-shot baseline per candidate + FT result, same gold set, temp 0. vLLM batching is
  nondeterministic → expect ±5pp run variance on small n.
