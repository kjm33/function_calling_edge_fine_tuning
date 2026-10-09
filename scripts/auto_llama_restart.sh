#!/usr/bin/env bash
# Wait for GPU0 to be free (foreign jobs finish on their own), then restart the
# llama-3.2-3b finalist training. Polite: NEVER kills foreign processes.
# Marker on success: LLAMA_RESTARTED in logs/auto_llama_restart.log.
set -u
cd "$(dirname "$0")/.."
LOG()  { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
FATAL(){ echo "[$(date '+%m-%d %H:%M:%S')] FATAL: $*"; exit 1; }
GPU0_UUID="GPU-dd54e124-cd86-f586-07fc-0b991c76b00b"

LOG "waiting for GPU0 to be free (foreign training holds it)"
for i in $(seq 1 1440); do  # up to 24h, 1-min polls
  procs=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader \
          | grep "$GPU0_UUID" | cut -d, -f1)
  [ -z "$procs" ] && { LOG "GPU0 free after ${i}min"; break; }
  [ $((i % 30)) -eq 0 ] && LOG "still waiting ($((i / 60))h$((i % 60))m): pids $procs"
  [ "$i" -eq 1440 ] && FATAL "GPU0 busy for 24h — gave up waiting"
  sleep 60
done

LOG "relaunching llama-3.2-3b finalist (2k x 2ep @ 16k)"
HF_HOME=/var/tmp/fcft_hf_cache setsid nohup train/venv/bin/python train/train_sft.py \
  --candidate llama-3.2-3b --train data/final/train_2k.jsonl --val data/final/val.jsonl \
  --out-dir train/runs/llama-3.2-3b_final2k --seq-len 16384 --epochs 2 \
  --eval-steps 250 --save-steps 500 --patience 3 \
  --gpu 0 > logs/train_llama-3.2-3b_final2k.log 2>&1 < /dev/null &
sleep 90
pgrep -f "[t]rain_sft.*llama" >/dev/null || FATAL "llama train_sft not alive 90s after launch"
LOG "LLAMA_RESTARTED — monitor logs/train_llama-3.2-3b_final2k.log"
