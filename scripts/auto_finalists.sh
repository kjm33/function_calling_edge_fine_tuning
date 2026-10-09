#!/usr/bin/env bash
# Auto-chain: wait for phi ft eval -> final 5-way ranking -> launch top-2
# finalist training (train_2k x 2ep, one per GPU) with zero idle time.
# Marker on success: FINALISTS_DONE in logs/auto_finalists.log.
set -u
cd "$(dirname "$0")/.."
LOG()  { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
FATAL(){ echo "[$(date '+%m-%d %H:%M:%S')] FATAL: $*"; exit 1; }
GPU0_UUID="GPU-dd54e124-cd86-f586-07fc-0b991c76b00b"
GPU1_UUID="GPU-6c9a3adb-4bca-99b7-6d92-4cdb8dbe060a"
PHI_JSON=eval/results/phi-4-mini_ft.json

# ------------------------------------------------ phase A: wait for phi eval
LOG "PHASE A: waiting for phi-4-mini ft eval to finish"
START=$(date +%s)
while :; do
  if ! pgrep -f "[e]val.harness" >/dev/null; then
    MTIME=$(stat -c %Y "$PHI_JSON" 2>/dev/null || echo 0)
    if [ "$MTIME" -ge "$START" ]; then
      LOG "phi eval finished ($PHI_JSON refreshed)"
      break
    fi
    FATAL "eval.harness exited but $PHI_JSON was not refreshed — check logs/eval_screen_phi_fix2.log"
  fi
  sleep 120
done

# ------------------------------------------------ phase B: rank + pick top 2
LOG "PHASE B: final screen ranking (exec, tiebreak arg_accuracy)"
PICK=$(PYTHONPATH=. .venv/bin/python - <<'PY'
import json, sys
cands = ["qwen3.5-4b", "qwen3-4b-2507", "llama-3.2-3b", "phi-4-mini", "gemma-3-4b"]
rows = []
for c in cands:
    try:
        a = json.load(open(f"eval/results/{c}_ft.json"))["aggregate"]
    except Exception as e:
        print(f"ERR {c}: {e}", file=sys.stderr); sys.exit(3)
    rows.append((c, a))
rows.sort(key=lambda r: (-(r[1].get("execution_success_rate") or 0.0),
                         -(r[1].get("arg_accuracy") or 0.0)))
lines = ["| rank | candidate | exec | arg | strict | parse |",
         "|---|---|---|---|---|---|"]
for i, (c, a) in enumerate(rows, 1):
    f = lambda k: (a.get(k) or 0.0) * 100
    lines.append(f"| {i} | {c} | {f('execution_success_rate'):.1f} | "
                 f"{f('arg_accuracy'):.1f} | {f('tool_selection_strict'):.1f} | "
                 f"{f('parse_ok_rate'):.1f} |")
hdr = "# Screen ranking (500x1ep adapters, ft tag, gold n=643)\n\n" + "\n".join(lines) + "\n"
open("eval/results/SCREEN_RANKING.md", "w").write(hdr)
print(rows[0][0], rows[1][0])
PY
) || FATAL "ranking failed"
A=$(echo $PICK | awk '{print $1}'); B=$(echo $PICK | awk '{print $2}')
LOG "finalists: $A (GPU0), $B (GPU1) — table in eval/results/SCREEN_RANKING.md"

# ------------------------------------------------ phase C: drain GPUs, train
# Polite-first drain: foreign jobs (e.g. smoke tests from other sessions) get a
# 40-min grace window to finish on their own; only then hard cleanup.
wait_gpu_free() {  # $1 uuid, $2 label
  local i procs
  for i in $(seq 1 480); do
    procs=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader \
            | grep "$1" | cut -d, -f1)
    [ -z "$procs" ] && { LOG "$2 drained"; return 0; }
    [ $((i % 12)) -eq 0 ] && LOG "$2 still busy ($((i * 5 / 60)))min): pids $procs"
    [ $((i % 96)) -eq 0 ] && kill -9 $procs 2>/dev/null
    sleep 5
  done
  FATAL "$2 never drained"
}
train_final() {  # $1 candidate, $2 gpu
  LOG "launching finalist $1 on GPU$2 (2k x 2ep @ 16k ctx)"
  HF_HOME=/var/tmp/fcft_hf_cache setsid nohup train/venv/bin/python train/train_sft.py \
    --candidate "$1" --train data/final/train_2k.jsonl --val data/final/val.jsonl \
    --out-dir "train/runs/$1_final2k" --seq-len 16384 --epochs 2 \
    --eval-steps 250 --save-steps 500 --patience 3 --force \
    --gpu "$2" > "logs/train_$1_final2k.log" 2>&1 < /dev/null &
}

# GPU1 frees as soon as the phi eval server exits -> start finalist B there at once;
# GPU0 may host a foreign smoke test -> wait politely, then start finalist A.
(
  wait_gpu_free "$GPU1_UUID" GPU1
  train_final "$B" 1
) &
PAIR_B=$!
wait_gpu_free "$GPU0_UUID" GPU0
train_final "$A" 0
wait $PAIR_B
sleep 90
N=$(pgrep -fc "[t]rain_sft" || true)
[ "${N:-0}" -ge 2 ] || FATAL "expected 2 train_sft procs 90s after launch, got ${N:-0}"
LOG "finalist training running ($N procs); logs/train_<cand>_final2k.log + train/runs/<cand>_final2k/train.log"
LOG "FINALISTS_DONE"
