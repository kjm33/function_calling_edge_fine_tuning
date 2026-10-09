#!/usr/bin/env bash
# Screen-round evals: ft tag only (zs rows already banked in SUMMARY.md).
# GPU0: qwen3.5-4b -> llama-3.2-3b -> gemma-3-4b | GPU1: qwen3-4b-2507 -> phi-4-mini
set -u
cd "$(dirname "$0")/.."
export PATH=$PWD/eval/venv/bin:$PATH
export HF_HOME=/var/tmp/fcft_hf_cache
run() {  # $1=candidate $2=gpu
  eval/venv/bin/python -m eval.harness --candidate "$1" --tag ft \
    --gold data/test_gold_merged.jsonl --gpu "$2" \
    > "logs/eval_screen_$1.log" 2>&1 \
    && echo "[$(date '+%H:%M:%S')] EVAL OK: $1" \
    || echo "[$(date '+%H:%M:%S')] EVAL FAILED: $1 (logs/eval_screen_$1.log)"
}
( for c in qwen3.5-4b llama-3.2-3b gemma-3-4b; do run "$c" 0; done; echo "GPU0 chain done" ) &
( for c in qwen3-4b-2507 phi-4-mini;      do run "$c" 1; done; echo "GPU1 chain done" ) &
wait
echo "SCREEN_EVALS_DONE — rank and pick top 2"
