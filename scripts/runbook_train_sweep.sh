#!/usr/bin/env bash
# Training sweep: 5 candidates, one per GPU, sequential pairs, independent failure.
#   pair 1: qwen3.5-4b (GPU0) + qwen3-4b-2507 (GPU1)
#   pair 2: llama-3.2-3b (GPU0) + phi-4-mini  (GPU1)
#   pair 3: gemma-3-4b (GPU0)
# Ends with SWEEP COMPLETE banner; adapters land in train/runs/<candidate>/.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/var/tmp/fcft_hf_cache}
LOG() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
TP=train/venv/bin/python

train_one() {  # $1=candidate $2=gpu -> logs to logs/train_<cand>.log
  local c=$1 g=$2 rc
  LOG "TRAIN START: $c on GPU$g"
  $TP train/train_sft.py --candidate "$c" \
      --train data/final/train_500.jsonl --val data/final/val.jsonl \
      --seq-len 16384 --epochs 1 --eval-steps 125 --save-steps 250 --patience 2 --force \
      --gpu "$g" > "logs/train_$c.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then LOG "TRAIN OK: $c"; else LOG "TRAIN FAILED (rc=$rc): $c — see logs/train_$c.log"; fi
  return $rc
}

run_pair() {  # $1=gpu0 cand (required)  $2=gpu1 cand (optional)
  local pids=() rc_all=0 c
  train_one "$1" 0 & pids+=($!)
  if [ -n "${2:-}" ]; then train_one "$2" 1 & pids+=($!); fi
  for p in "${pids[@]}"; do wait "$p" || rc_all=1; done
  [ $rc_all -eq 0 ] || LOG "NOTE: pair had failure(s) — sweep continues"
}

LOG "SWEEP BEGIN (SCREEN 500x1ep) — data: $(wc -l < data/final/train_500.jsonl) train / $(wc -l < data/final/val.jsonl) val, seq_len 16384"

LOG "=== pair 1/3: qwen3.5-4b + qwen3-4b-2507 ==="
run_pair qwen3.5-4b qwen3-4b-2507
LOG "=== pair 2/3: llama-3.2-3b + phi-4-mini ==="
run_pair llama-3.2-3b phi-4-mini
LOG "=== pair 3/3: gemma-3-4b (solo) ==="
run_pair gemma-3-4b ""

LOG "sweep finished — adapter inventory:"
for c in qwen3.5-4b qwen3-4b-2507 llama-3.2-3b phi-4-mini gemma-3-4b; do
  a="train/runs/$c"
  if compgen -G "$a/checkpoint-*/adapter_model.safetensors" > /dev/null; then
    LOG "  $c: OK ($(ls -d "$a"/checkpoint-* | sort -V | tail -1))"
  elif [ -f "$a/adapter_model.safetensors" ]; then
    LOG "  $c: OK (final adapter in run root)"
  else
    LOG "  $c: MISSING ADAPTER"
  fi
done
LOG "SWEEP COMPLETE — next: eval harness on data/test_gold_merged.jsonl (+ OOD)"
