# Evaluation stack

Three pieces:

| piece | file | runs under |
|---|---|---|
| gold test-set builder | `eval/build_gold.py` | main `.venv` (httpx/dotenv) |
| serving + scoring harness | `eval/harness.py` | `eval/venv` (vLLM nightly, must NOT see train venv) |
| serving environment | `eval/venv/` | — |

## 1. Install the eval venv (vLLM nightly)

```bash
uv venv eval/venv --python 3.12
uv pip install --python eval/venv/bin/python --prerelease=allow "vllm>=0.0.0.dev0" \
    --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
uv pip install --python eval/venv/bin/python openai python-dotenv ninja
# ninja is REQUIRED: vLLM's torch.compile warmup hard-fails without it
# ("FileNotFoundError: 'ninja'"); the harness also prepends eval/venv/bin to the
# server's PATH so the server process finds the ninja binary.

eval/venv/bin/vllm --version          # e.g. 0.30.1rc1.dev593+g844211744
```

Run the harness with `python -m eval.harness` from the repo ROOT (the local `./mcp`
package must shadow the PyPI `mcp` SDK that vLLM installs; `mcp/__init__.py` +
sys.path ordering handle this).

The bundled transformers must know `qwen3_5` (Qwen/Qwen3.5-4B's model type). Verified on
the nightly above; if a future nightly regresses, pin the working version:

```bash
uv pip install --python eval/venv/bin/python \
    "vllm @ https://wheels.vllm.ai/nightly/<exact-wheel-name>.whl" --torch-backend=auto
```

## 2. Build the gold test set

Teacher-solved, tool-execution-verified items, DISJOINT from training seeds
(dedicated selection stream `random.Random(20260101)`; chosen seed ids recorded to
`data/gold_seed_ids.json` as a guard file for `build_dataset`).

```bash
# smoke (24 items, 5 langs, 4 no_tool) -> data/test_gold_smoke.jsonl
.venv/bin/python -m eval.build_gold --n 24 --out data/test_gold_smoke.jsonl

# full scale (~600 tool_use = 120/lang + ~100 no_tool = 20/lang) -> data/test_gold.jsonl
.venv/bin/python -m eval.build_gold --n 700 --out data/test_gold.jsonl
```

Key flags: `--langs en,de,pl,fr,es`, `--modes tool_use,no_tool`, `--no-tool-n N`,
`--workers 4`, `--teachers ...`. Tool executions reuse `data/tool_response_cache.json`
(cache-friendly w.r.t. TomTom quota). The report at the end prints per-lang/mode
counts, reference-call stats and a sample item.

## 3. Run the harness

Per candidate+tag it (1) checks the GPU is free (>2 GiB foreign use => refuse),
(2) launches a vLLM OpenAI server with flags from `train/configs/<key>.json`
(`--tool-call-parser`, optional `--reasoning-parser`, `--language-model-only` for
qwen3.5; LoRA via `--enable-lora --max-lora-rank 64 --lora-modules`), polls
`/health` (10 min timeout, log tail on failure), (3) replays every gold item as an
agent rollout (tool calls executed through `ToolRegistry` + cache, results fed back,
<= `--max-hops` rounds per user turn), (4) scores and writes
`eval/results/<candidate>_<tag>.json` + a row in `eval/results/SUMMARY.md`.

```bash
# zero-shot smoke (24 items) on GPU 0
eval/venv/bin/python -m eval.harness --candidate qwen3-4b-2507 --tag zeroshot \
    --gold data/test_gold_smoke.jsonl --gpu 0

# FT smoke: adapter from train/runs/<dir> (must contain adapter_config.json)
eval/venv/bin/python -m eval.harness --candidate qwen3-4b-2507 --tag ft \
    --adapter train/runs/eval-smoke --gold data/test_gold_smoke.jsonl --limit 8 --gpu 1
# ('--adapter auto' searches train/runs/<key>/{checkpoint-best,final,<root>})

# full runs, all candidates, both tags, judge on a sample
eval/venv/bin/python -m eval.harness --all --tag both \
    --gold data/test_gold.jsonl --gpu 0 --judge-sample 100
```

Ports default to `8120 + index` of the candidate in the sorted candidate list
(override `--port`). One candidate/GPU at a time; the server is always killed on
exit (atexit + finally, process-group SIGTERM then SIGKILL). Servers run with
`--max-model-len 32768` (14-tool schemas are ~7k tokens and multi-hop rollouts with
large tool payloads grow past 16k); on a context-length 400 the harness retries the
call with `max_tokens: 256`.

### Metrics

Per item: `parse_ok` (server parser OR harness `<tool_call>` fallback parsed calls),
`fallback_parse` (harness-side parses), `hops_used`,
`tool_selection_strict` (per-step ordered name sequence == reference) and
`tool_selection_set` (per-step name set == reference),
`arg_accuracy` (per reference call: fraction of key args matching — exact
strings/enums/ids, numbers within 2%, geocoords within 500 m haversine),
`execution_success`, `refusal_ok` (no_tool items: zero tool calls + direct answer),
`grounded` (`--judge-sample N` items judged by `JUDGE_MODELS[0]` against the item
rubric; pass = score >= 7).

Aggregates: overall + per-language + per-provider (`tomtom` / `here` / `keyless` =
osrm/nominatim/open-meteo prefixes) + `fallback_parse_rate`. Read them from the
`aggregate` block of `eval/results/<candidate>_<tag>.json` or the `SUMMARY.md` table.

### Candidate notes

- `qwen3.5-4b`: vLLM nightly + `--tool-call-parser qwen3_coder --reasoning-parser
  qwen3 --language-model-only`; requests send `chat_template_kwargs:
  {"enable_thinking": false}` (must match the training render).
- `gemma-3-4b`: served with the BASE template; the FT adapter teaches `<tool_call>`
  emission, the harness injects the Hermes tool instructions into the first user
  turn (via `render_templates.prepare_messages`) and regex-parses `<tool_call>`
  blocks when the server parser returns nothing (`fallback_parse` metric).
- `phi-4-mini`: `--trust-remote-code`; `llama-3.2-3b`: `llama3_json` parser;
  `qwen3-4b-2507`: `hermes`.

## 4. OOD generalization eval (unseen tools)

Probes overfitting to our 21 geo tools: gold items whose tools come from a
completely different corpus (NousResearch/hermes-function-calling-v1,
single-turn subset; xLAM is gated, ToolACE is reserved for the mixed-training
arm). Foreign tools are never executed — scoring is tool-selection + argument
match, plus two smoking-gun rates:

- `ood_hallucination_rate`: fraction of model calls whose name is NOT in the
  item's tool list
- `geo_hallucination_rate`: fraction of model calls that name one of OUR geo
  tools (specialization smell)

```bash
# build (297 items, stratified by category, deterministic seed)
.venv/bin/python -m eval.build_ood --n 300 --out data/test_gold_ood.jsonl

# run (same server/parser path per candidate; generic FC system, 1 emission)
eval/venv/bin/python -m eval.harness --candidate qwen3.5-4b --tag zeroshot \
    --gold data/test_gold_ood.jsonl --ood --gpu 0
```

Results land in a separate `## OOD generalization (unseen tools)` table in
`SUMMARY.md`. Compare zero-shot vs FT (and pure-geo vs ToolACE-mixed arms):
Δ ≈ 0 on sel_strict/arg_acc with low hallucination rates = benign
specialization; collapsed scores or geo-tool hallucinations = overfit.
Note: ~790 hermes "Information Extraction" rows are dropped at build time
(their reference calls mismatch their own schemas — gold must be
schema-consistent).

## Files

- `eval/results/<candidate>_<tag>.json` — meta (cmd, port, adapter, vLLM version,
  `adapter_loaded_log_lines` proving the LoRA was mounted), per-item rows, aggregates.
- `eval/results/SUMMARY.md` — one row per (candidate, tag) run.
- `eval/results/logs/serve_*.log` — raw vLLM server logs.
