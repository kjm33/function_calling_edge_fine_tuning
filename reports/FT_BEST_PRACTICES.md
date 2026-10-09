# Fine-Tuning Best Practices for Small LLMs (Qwen3.5-4B) — In-Car / Edge Function Calling

Online research digest, 2026-10-03. Sources: Qwen3.5-4B official model card (HF),
arXiv:2604.22127 (+ GitHub correction notice), arXiv:2409.00608 (TinyAgent, EMNLP'24 demo),
Cerence/SiMa.ai + Arm Kleidi announcements (2025), vLLM Qwen3.5 recipe, community guides
(unsloth/TRL practice). Primary purpose: sanity-check and improve our train/eval pipeline for
the qwen3.5-4b candidate; secondary: deployment guidance for the in-car target.

## 1. Qwen3.5-4B — what the official model card says

**Architecture** (matters for training and edge deployment):

- Hybrid: 32 layers, hidden 2560, layout `8 × (3 × (GatedDeltaNet → FFN) → 1 × (GatedAttention → FFN))`
  — i.e. 3:1 linear-attention:full-attention, **sequential** per-block composition.
- 4B text LM (tied embeddings, 248k vocab) + vision encoder; ~5B params BF16 total.
  `--language-model-only` drops the vision path at serve time.
- Context 262,144 native (YaRN-extensible — irrelevant for us; keep context small on device).
- **MTP** (multi-token prediction) head trained-in → speculative decoding supported
  (`qwen3_next_mtp`, 2 speculative tokens in vLLM; NEXTN algo in SGLang).

**Agent/tool-calling scores** (small-size class): BFCL-V4 50.3, TAU2-Bench 79.9, IFEval 89.8,
DeepPlanning 17.6 — strongest tool-use tier for ≤4B, which is why it's our lead candidate.

**Thinking mode** — critical for us:

- Thinking is ON by default (`<think>...</think>` before response).
- **No `/think` `/no_think` soft switch** (removed vs Qwen3). Disable only via
  `chat_template_kwargs={"enable_thinking": False}` (or tokenizer `enable_thinking=False`).
- Our stack already trains AND serves with thinking disabled — this train/serve consistency is
  exactly right; do not drift.
- Multi-turn best practice: strip thinking content from history (the Jinja template does it;
  relevant only if we ever hand-roll prompts).

**Serving flags** (confirm ours match — they do): `--reasoning-parser qwen3 --tool-call-parser
qwen3_coder --enable-auto-tool-choice`, `--language-model-only` for text-only (frees vision
memory for KV cache).

**Sampling** (official recommendations):

| Mode | temp | top_p | top_k | presence_penalty |
|---|---|---|---|---|
| Thinking, general | 1.0 | 0.95 | 20 | 1.5 |
| Thinking, precise coding | 0.6 | 0.95 | 20 | 0.0 |
| **Instruct (non-thinking), general** | 0.7 | 0.8 | 20 | 1.5 |
| Instruct, reasoning | 1.0 | 1.0 | 40 | 2.0 |

Our temp-0 exec-based gold eval is correct for the arbiter (determinism). If we ever serve
non-greedy (e.g. user-facing pilot), keep presence_penalty ≥1 to suppress repetition loops in
instruct mode.

**Ecosystem maturity**: 683 LoRA adapters, 479 quantizations, 838 finetunes already on HF —
low risk of undocumented stack breakage.

## 2. LoRA on hybrid architectures — placement

Only direct study: **arXiv:2604.22127 "Where Should LoRA Go? Component-Type Placement in
Hybrid Language Models"** (Apr 2026; Qwen3.5-0.8B sequential GDN + Falcon-H1-0.5B parallel
Mamba-2; r16 α32 dropout 0.05 lr 2e-4 cosine, 3 epochs, eff. batch 16, bf16, 8-bit Adam).

**Read the v2 correction, not the abstract of v1**: the headline claims (attention-only LoRA
wins with 5–10× fewer params; recurrent-backbone adaptation destructive in sequential hybrids,
−14.8pp GSM8K) were **withdrawn** after two eval-harness defects (GSM8K answer extraction;
HumanEval never executed, NaN→0). Corrected, statistically surviving findings:

1. **No placement reliably maximizes on-target gain** — and the design was underpowered
   (sub-1B models, 2k examples, single seed; GSM8K subsets of 128–256 vs power analysis
   needing 635–1764 items for 3–5pp). Treat "where to put LoRA for best task gain" as open.
2. **Broad placements cause off-target damage; narrow ones don't.** After general-instruct FT
   on Falcon-H1, `all_eligible`, `attn+mlp`, `mlp_only` significantly lost HellaSwag;
   `attention_only` / `ssm_only` did not. Placement governs collateral damage, not target gain.
3. Topologies order single-component placements differently (no universal rule).

**Implications for us (qwen3.5-4b)**:

- Our all-linear-modules LoRA remains a reasonable default (no evidence it's worse on-target).
- If gold-eval or spot checks show regression in general/multilingual behavior after FT, the
  documented mitigation is a **narrower target set** (attention projections only) — cheap A/B
  on the existing exec harness.
- The paper's config doubles as a sane reference point; ours (see train/configs) is in the
  same family.

## 3. SFT for function calling at small scale

- **TinyAgent** (arXiv:2409.00608): curated high-quality FC dataset + SFT lets 1.1B/7B models
  match GPT-4-Turbo function-calling on a focused toolset — direct validation of our
  APIGen-style "quality > quantity, ~4-5k verified dialogs" premise. Two transferable tricks:
  - **Tool RAG**: retrieve the relevant tool subset per turn instead of stuffing all schemas.
    We always inject all 21 tools; prompt bloat disproportionately hurts small models. If
    zero-shot baselines show tool-confusion, this is the first lever.
  - **Quantization at deploy for decode speed** (see §5).
- **Execution-based eval as the final arbiter** (our `eval/harness.py` replay design) is the
  BFCL-style standard — LLM judges are a filter only, never ground truth. Our dual-judge +
  exec-replay split matches best practice.
- **Loss masking**: assistant/tool-call tokens only (we pre-mask labels — correct). Never
  compute loss on tool/system/user turns.
- **No packing for hybrid arch** — packing breaks DeltaNet recurrence boundaries; already a
  hard rule in our train stack. (Also: `processing_class=tokenizer`, not AutoProcessor.)
- **Hyperparameter consensus** (community + study configs) for 3–4B LoRA SFT:
  - rank: 8–32; r=8 often within ~0.2–0.5pp of r=32 ("How Small Can You Go?" study) — rank is
    rarely the bottleneck; data quality is.
  - α = 2r, dropout 0–0.05.
  - lr 1e-4–2e-4, cosine, warmup 3–5%, 2–3 epochs for 4–5k verified samples, early-stop on
    val loss + task metric.
  - QLoRA saves memory, not time (dequant-on-forward); we have VRAM headroom → plain LoRA.
- **Catastrophic forgetting**: no directly cited multilingual-replay study surfaced, but the
  corrected placement paper (§2) + general practice suggest: if multilingual/general quality
  degrades post-FT, mix in a small replay slice of general/multilingual instruct data (~10%),
  or narrow the LoRA targets.

## 4. Edge / automotive constraints and precedents

- **Precedent**: Cerence **CaLLM Edge** — automotive-grade embedded SLM shipping on
  SiMa.ai Modalix MLSoC and Arm (Kleidi) — establishes the pattern: dedicated in-car SLM for
  offline/deterministic intent + tool routing, cloud LLM as fallback for open-ended queries.
  Our architecture (FT SLM does tool calls; tools hit REST/MCP) fits the same envelope.
- **In-car hard requirements** (from on-device AI literature + practice): offline-first,
  sub-second first-token, shared SoC memory/thermal budget, temp-0 determinism, strict schema
  validation of every tool call (no hallucinated arguments reaching APIs), 10+ year lifecycle
  (pin runtime + artifact versions).
- **Qwen3.5's hybrid arch is an edge win**: only 8 of 32 layers are full attention → KV cache
  ~4× smaller than a same-depth pure transformer. Longer car-session context fits in less
  memory; DeltaNet layers are O(N) decode.
- **MTP speculative decoding** = biggest free latency lever at serve time for Qwen3.5
  (draft-verify with the trained MTP head).

## 5. Quantization for deployment

- Q4_K_M GGUF ≈ 2.5–3 GB for the 4B; llama.cpp runs the Qwen3.5 family natively (community
  runs 35B-A3B at 2-bit on RPi5 ≈ 3 tok/s; 4B is comfortable on laptop/embedded-class CPUs).
  WebGPU (browser) demos exist for Qwen3.5-4B.
- **Tool calling is quantization-sensitive**: structured JSON/argument fidelity degrades more
  than chat quality at INT4. Standard practice: **eval the deployed artifact, not the BF16
  adapter** — merge LoRA → quantize (Q4_K_M/Q8 GGUF, or AWQ/W4A16 for GPU runtimes) → re-run
  the same gold eval on the quantized model. Our harness is vLLM-based; a llama.cpp side-check
  of the final winner at Q4/Q8 would de-risk the deployment story.
- Quantization-aware alternatives (QLoRA-FT on 4-bit base) train the adapter on the quantized
  base so serve-time mismatch shrinks — an option only if the merged+quant eval shows drift.

## 6. Status vs. our pipeline

| Practice | Status in repo |
|---|---|
| Thinking disabled consistently (train + serve) | ✅ done (keep) |
| `qwen3_coder` tool parser + `qwen3` reasoning parser + `--language-model-only` | ✅ done |
| Pre-masked labels (assistant-only loss), no packing, tokenizer-as-processor | ✅ done |
| Execution-based gold eval as final arbiter | ✅ done |
| Dual-judge as filter only | ✅ done |
| LoRA hyperparams in consensus band (r/α/lr/epochs) | ✅ (per train/configs) |
| Tool RAG / prompt slimming (21 tools always in prompt) | ⬜ optional — first lever if tool confusion |
| Narrow-vs-broad LoRA target A/B | ⬜ only if general/multilingual regression observed |
| Replay mix against forgetting | ⬜ only if regression observed |
| Merged+quantized (Q4/Q8) artifact eval of winner | ⬜ recommended before any deployment claim |
| MTP speculative decode latency benchmark | ⬜ optional latency work |

## Sources

- Qwen/Qwen3.5-4B model card — https://huggingface.co/Qwen/Qwen3.5-4B (arch, sampling,
  thinking-mode handling, serving flags, MTP)
- arXiv:2604.22127 + github.com/hecboar/lora-placement-hybrid (LoRA placement in hybrids;
  **v2 correction** — v1 claims withdrawn)
- arXiv:2409.00608 TinyAgent: Function Calling at the Edge (EMNLP'24 demo)
- Cerence AI × SiMa.ai (Sep 2025), Cerence × Arm Kleidi (Jun 2025) — CaLLM Edge announcements
- vLLM Qwen3.5 recipe (docs.vllm.ai), community quantization reports (llama.cpp/RPi5)
- "How Small Can You Go? LoRA Fine-Tuning 270M…" (2026) — rank sensitivity
- Community SFT practice: unsloth/TRL guides, QLoRA paper ("saves memory not time")
