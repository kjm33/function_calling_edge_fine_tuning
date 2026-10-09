#!/usr/bin/env bash
# Auto-chain: wait for both finalist ft evals -> launch OOD evals (both finalists).
# Marker on success: OOD_DONE in logs/auto_ood.log.
set -u
cd "$(dirname "$0")/.."
LOG()  { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
FATAL(){ echo "[$(date '+%m-%d %H:%M:%S')] FATAL: $*"; exit 1; }
START=$(date +%s)

LOG "PHASE A: waiting for llama+gemma finalist ft evals"
while :; do
  if ! pgrep -f "[e]val.harness" >/dev/null; then
    for f in eval/results/llama-3.2-3b_ft.json eval/results/gemma-3-4b_ft.json; do
      MTIME=$(stat -c %Y "$f" 2>/dev/null || echo 0)
      [ "$MTIME" -ge "$START" ] || FATAL "$f not refreshed — check logs/eval_final_*.log"
    done
    LOG "both ft evals finished"
    break
  fi
  sleep 120
done

LOG "PHASE B: launching OOD evals (finalists, ft adapters, test_gold_ood)"
PATH=eval/venv/bin:$PATH HF_HOME=/var/tmp/fcft_hf_cache setsid nohup \
  eval/venv/bin/python -m eval.harness --candidate llama-3.2-3b --tag ft --ood \
  --adapter train/runs/llama-3.2-3b_final2k --gold data/test_gold_ood.jsonl --gpu 0 \
  > logs/eval_ood_llama.log 2>&1 < /dev/null &
PATH=eval/venv/bin:$PATH HF_HOME=/var/tmp/fcft_hf_cache setsid nohup \
  eval/venv/bin/python -m eval.harness --candidate gemma-3-4b --tag ft --ood \
  --adapter train/runs/gemma-3-4b_final2k --gold data/test_gold_ood.jsonl --gpu 1 \
  > logs/eval_ood_gemma.log 2>&1 < /dev/null &
sleep 120
N=$(pgrep -fc "[e]val.harness" || true)
[ "${N:-0}" -ge 2 ] || FATAL "expected 2 harness procs 2min after OOD launch, got ${N:-0}"
LOG "OOD evals running ($N procs); logs/eval_ood_*.log"
LOG "OOD_DONE"
