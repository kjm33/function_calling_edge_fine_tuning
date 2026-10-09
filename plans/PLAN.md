# Plan: Fine-tuning small local models for MCP function calling (geo providers)

Approved: 2026-10-02. Task: fine-tune ≤4B edge models for function calling with MCP tools from
geo providers (HERE, TomTom, + keyless), generate verified training data from
`../instruction_extraction` assets, compare 5 candidates under a live MCP-exec eval.

## Resources

- Hardware: 2× RTX 3090 24GB (48GB), 62GB RAM, 32 cores. Disk /home ~74GB free → HF cache lean,
  keep LoRA adapters only.
- APIs (all in `../.env`): HERE_API_KEY, TOMTOM_API_KEY, OPENROUTER_API_KEY, HF_TOKEN,
  OPENAI_API_KEY_1 + OPENAI_API_KEY_2 (two accounts, Tier 1-2 → 2×2.5M tok/day on minis; use
  round-robin across both).
- Seeds: `../instruction_extraction/data/instructions_v3.jsonl` (21,959 nav instructions,
  5 langs), `dialogues.jsonl` (2,029 multi-turn), `scenarios_full.jsonl` (1,200 scenario cards),
  `json_seeds.json` (HERE Routing v8 schemas).

## 1. Tool pool (~25-30 tools)

- **TomTom official MCP** (`@tomtom-org/tomtom-mcp` via npx stdio; 15 tools incl. routing,
  geocode, POI, EV, reachable-range, along-route, traffic). `response_detail=compact` default.
  Free tier ~2.5k routing calls/day → arg-keyed response cache.
- **HERE**: community `heremaps-mcp-server` (6 tools) if workable, else custom thin FastMCP
  wrapper over HERE Routing v8 / Geocode / Search (schemas matching json_seeds). ~250k tx/month.
- **Keyless diversity**: Open-Meteo MCP (~weather tools), OSRM/OSM MCP (route, geocode, nearby).
  Relieves quota, adds provider variety.

## 2. Dataset (APIGen-style, verified)

- Pipeline: seed selection (instructions/dialogues/scenarios) → teacher generates multi-turn
  dialogs w/ tool calls → **live MCP execution** of every call → verification (JSON-schema check
  + execution success + dual-LLM judge ≥ threshold, select_v3.py pattern) → MMR 5-gram Jaccard
  dedup.
- Teachers: OpenAI minis (both keys, round-robin) primary; OpenRouter roster (qwen3.5-9b,
  glm-5.3-flash, deepseek-v4-flash …) for diversity/cost relief.
- Target ~15-20k verified multi-turn dialogs; balance 5 langs (EN/DE/PL/FR/ES) × provider ×
  call-count (1-4); include ~10% no-tool/irrelevance, ~10% error-recovery turns, 10-15% general
  multilingual chat mix (anti-forgetting).
- Neutral storage: `messages[{role,content,tool_calls,tool_call_id}]` + `tools[]`; render into
  per-model chat templates at train time.

## 3. Training

- Frameworks: LLaMA-Factory primary, Unsloth fast-path, TRL+PEFT fallback (Qwen3.5-4B new arch).
- Recipe: LoRA r=32 α=64 dropout=0.05 all-linear, lr 1e-4 cosine 3% warmup, 2-3 epochs, seq 8192,
  assistant-only loss, bf16, flash-attn2. 2 concurrent single-GPU runs (one candidate per 3090).
- Candidates (5): Qwen3.5-4B (BFCL-V4 50.3 prior; nightly vLLM; train-stack smoke test first),
  Qwen3-4B-Instruct-2507 (safe anchor), Llama-3.2-3B-Instruct, Phi-4-mini-instruct,
  gemma-3-4b-it (Hermes-format transplant).

## 4. Eval (live MCP-exec harness)

- Serve each candidate in vLLM with native parser: qwen3_coder (+reasoning-parser qwen3) for
  Qwen3.5-4B, hermes for Qwen3-4B & gemma-3, llama3_json, phi4_mini_json.
- Metrics: parse rate, tool-selection acc, per-arg acc, execution success (real API calls),
  multi-turn trajectory acc, irrelevance refusal; slices per language × provider.
- Baselines: zero-shot per candidate + teacher ceiling.
- Gold test ~700 items (120×5 langs + irrelevance), teacher-built, execution-verified,
  manually spot-checked.

## 5. Timeline (~10 days)

D0 scaffold + MCP smoke → D1-2 gen pipeline + pilot 500 → D3-6 full gen + gold set →
D5-8 fine-tune 5 candidates + eval → D9-10 reports/RESULTS.md + winner (+ optional GGUF).

## 6. Risks

- Qwen3.5-4B train-stack immaturity → D0 smoke test; anchor = Qwen3-4B-2507.
- TomTom quota 2.5k/day routing → response cache + OSRM substitution where equivalent.
- HERE community server quality → custom FastMCP wrapper fallback.
- Disk 74GB → adapters-only retention, prune HF cache.

## Repo layout

```
function_calling_edge_fine_tuning/
├── mcp/            # registry exports, HERE wrapper server
├── src/            # config, tool_registry, mcp_client, dataset_gen, verify,
│                   # build_dataset, render_templates
├── data/           # raw_trajectories/, verified/, final/, test_gold.jsonl
├── train/          # per-candidate configs
├── eval/           # harness, metrics
├── reports/        # RESEARCH.md, RESULTS.md
└── plans/          # this plan
```
