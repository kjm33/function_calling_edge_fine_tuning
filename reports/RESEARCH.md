# Candidate Research Summary

## Selected candidates (5)

| # | Model | HF id | Params | vLLM tool parser | FC format | Multilingual | Notes |
|---|-------|-------|--------|------------------|-----------|--------------|-------|
| 1 | Qwen3.5-4B | Qwen/Qwen3.5-4B | ~4.3B | qwen3_coder (+reasoning qwen3, --language-model-only, nightly) | native, dict-args | 201 langs | hybrid GDN arch, train-stack verified |
| 2 | Qwen3-4B-Instruct-2507 | Qwen/Qwen3-4B-Instruct-2507 | 4B | hermes | native | 100+ | mature anchor |
| 3 | Llama-3.2-3B-Instruct | unsloth/Llama-3.2-3B-Instruct (meta gated → mirror) | 3B | llama3_json | native (1 call/turn) | no PL | weakest PL coverage of the five |
| 4 | Phi-4-mini-instruct | microsoft/Phi-4-mini-instruct | 3.8B | phi4_mini_json | functools[...] JSON | partial PL | MIT |
| 5 | gemma-3-4b-it | google/gemma-3-4b-it | 4B | hermes (transplant) | none native → Hermes transplant | 140+ langs incl PL | template transplant experiment |

All five verified end-to-end at the infra level: rendering (reports/RENDER_EXAMPLES.md), training (train/README_TRAINING.md), serving (eval/README_EVAL.md).

## Evaluated and rejected

### Microsoft BitNet (bitnet-b1.58-2B4T) — SKIP (2026-10-02)
Checked at user request. Findings (researcher, primary sources — several claims weakly sourced, flagged):
- **Language coverage: FAIL.** Base is English-only (SmolLM-Corpus/dclm/open-web-math; tech report arXiv:2504.12285 lists multilingual as future work). Our hard requirement is EN/DE/PL/FR/ES with Polish critical; a 2.4B English-only base cannot acquire 5 languages reliably from a ~15-20k domain dataset.
- **vLLM serving: FAIL.** Official vLLM BitNet PR closed unmerged; HF repo has missing/custom modeling files. Proposed workarounds (community C++ engines, bitnet.cpp + custom OpenAI shim) are unverified, partly hallucination-prone, and would break eval comparability (all other candidates run the same vLLM OpenAI stack).
- **Training stack: UNVERIFIED.** Published LoRA paths pin an old transformers commit + trust_remote_code examples repo; compatibility with our transformers 5.18 / TRL 1.14 / PEFT 0.21 stack is unconfirmed.
- **Genuine upside** (why it stays on the radar): ~0.4GB non-embedding memory, ~29ms CPU latency/token, 0.028 J/token — best-in-class edge efficiency. If a future project targets CPU/on-device with English-only (or if vLLM lands native support + a multilingual base), revisit.
- Decision: not added as candidate 6; would cost a parallel serving+training path for a model that fails the Polish requirement.

## Storage note (2026-10-02)
User freed disk + approved /var/tmp for models. hf_cache/ moved to /var/tmp/fcft_hf_cache (symlinked); /home 116GB free; remaining candidate weights (~23GB) will auto-land there via the symlink.
