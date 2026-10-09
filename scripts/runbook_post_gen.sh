#!/usr/bin/env bash
# Post-generation runbook: fires when gen_local hits 7000 rows.
# Phases: [0] wait  [1] swap GPU0->gpt-oss, qwen:8001 seq-cap 16  [1.5] fix endpoints
#         [2] verify all raw  [3] build train/val  [4] gold rebuild+merge  [5] stop servers
# Fail-fast: any phase error exits 1 with a FATAL banner; servers left for inspection.
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD
LOG()  { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
FATAL(){ echo "[$(date '+%m-%d %H:%M:%S')] FATAL: $*"; exit 1; }
GPU0_UUID="GPU-dd54e124-cd86-f586-07fc-0b991c76b00b"
GPU1_UUID="GPU-6c9a3adb-4bca-99b7-6d92-4cdb8dbe060a"

gpu_procs() { nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep "$1" | cut -d, -f1; }
kill_gpu_engine() {  # $1 = gpu uuid
  for p in $(gpu_procs "$1"); do kill "$p" 2>/dev/null; done
}
wait_gpu_free() {  # $1 = gpu uuid — SIGKILL escalation if memory won't drain
  for i in $(seq 1 36); do
    [ -z "$(gpu_procs "$1")" ] && { LOG "GPU $1 drained"; return 0; }
    [ $((i % 6)) -eq 0 ] && { LOG "GPU $1 still busy — SIGKILL"; for p in $(gpu_procs "$1"); do kill -9 "$p" 2>/dev/null; done; }
    sleep 5
  done
  FATAL "GPU $1 never drained after kill attempts"
}
gpu_state() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; }
wait_healthy() {  # $1 = port, $2 = model id fragment
  for i in $(seq 1 80); do
    r=$(curl -s -m 3 "localhost:$1/v1/models" 2>/dev/null)
    case "$r" in *"$2"*) LOG "port $1 healthy ($2)"; return 0;; esac
    sleep 15
  done
  FATAL "port $1 never became healthy"
}

# ---------------------------------------------------------------- phase 0
LOG "PHASE 0: waiting for dataset_gen to finish (floor 6900 rows)"
while pgrep -f "[d]ataset_gen" >/dev/null; do sleep 300; done
R=$(wc -l < data/raw_trajectories/gen_local.jsonl)
[ "$R" -ge 6900 ] || FATAL "gen exited with only $R/6900 rows — manual inspection needed"
LOG "generation complete: $R rows"
pkill -f "[d]ataset_gen" 2>/dev/null; sleep 5

# ---------------------------------------------------------------- phase 1
LOG "PHASE 1: swap GPU0 to gpt-oss-20b; restart qwen:8001 with seq cap 16"
fuser -k 8000/tcp 2>/dev/null; fuser -k 8001/tcp 2>/dev/null; sleep 10
kill_gpu_engine "$GPU0_UUID"; kill_gpu_engine "$GPU1_UUID"
wait_gpu_free "$GPU0_UUID"; wait_gpu_free "$GPU1_UUID"
gpu_state | LOG

setsid nohup env CUDA_VISIBLE_DEVICES=0 PATH=$ROOT/eval/venv/bin:$PATH \
  HF_HOME=/var/tmp/fcft_hf_cache \
  eval/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-20b --served-model-name gpt-oss-20b --port 8000 \
  --gpu-memory-utilization 0.9 --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss \
  > logs/vllm_gptoss_phase1.log 2>&1 < /dev/null &
setsid nohup env CUDA_VISIBLE_DEVICES=1 PATH=$ROOT/eval/venv/bin:$PATH \
  HF_HOME=/var/tmp/fcft_hf_cache \
  eval/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-9B --served-model-name qwen3.5-9b --port 8001 \
  --gpu-memory-utilization 0.93 --max-model-len 32768 --max-num-seqs 16 \
  --language-model-only --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  > logs/vllm_qwen9b_gpu1_s16.log 2>&1 < /dev/null &
wait_healthy 8000 gpt-oss-20b
wait_healthy 8001 qwen3.5-9b

# ---------------------------------------------------------------- phase 1.5
LOG "PHASE 1.5: pin qwen routing to :8001 only (8000 is gpt-oss now)"
.venv/bin/python - <<'PY' || FATAL "endpoint rewrite failed"
import re, pathlib
p = pathlib.Path("src/llm_client.py"); s = p.read_text()
new = '    "local/qwen3.5-9b": [("http://127.0.0.1:8001/v1", "qwen3.5-9b")],'
s2 = re.sub(r'    "local/qwen3\.5-9b": \[.*?\],', new, s, count=1, flags=re.S)
assert s2 != s, "pattern not found"
p.write_text(s2)
PY
PYTHONPATH=. .venv/bin/python -c "
from src.llm_client import _next_local
assert all(':8001' in _next_local('local/qwen3.5-9b')[0] for _ in range(10))
assert ':8000' in _next_local('local/gpt-oss-20b')[0]
print('routing OK: qwen->8001 only, gpt-oss->8000')" || FATAL "routing check failed"

# ---------------------------------------------------------------- phase 2
LOG "PHASE 2: verify gen_local(7000) + gen_openai(192), dual local judges"
PATH=$HOME/tools/node22/bin:$PATH PYTHONPATH=. .venv/bin/python -u -m src.verify \
  --in data/raw_trajectories/gen_local.jsonl data/raw_trajectories/gen_openai.jsonl \
  --out data/verified/gen_final_verified.jsonl \
  --summary data/verified/gen_final_summary.json \
  --workers 16 > logs/verify_final.log 2>&1 || FATAL "verify failed — see logs/verify_final.log"
V=$(wc -l < data/verified/gen_final_verified.jsonl)
LOG "verify done: $V verified rows"
[ "$V" -lt 3500 ] && LOG "WARNING: only $V verified (<3500) — check summary before training"

# ---------------------------------------------------------------- phase 3
LOG "PHASE 3: build neutral train/val from all verified files"
.venv/bin/python -m src.build_dataset --in data/verified/*_verified.jsonl \
  --out-dir data/final > logs/build_final.log 2>&1 || FATAL "build_dataset failed"
LOG "train/val built:"; LOG "$(wc -l data/final/*.jsonl)"

# ---------------------------------------------------------------- phase 4
LOG "PHASE 4: gold rebuild with local teachers (75 old items kept as dedup base)"
if [ -r /proc/316709/cmdline ] && tr '\0' ' ' < /proc/316709/cmdline | grep -q build_gold; then
  kill -9 316709 && LOG "parked gold builder (316709) killed"
else
  LOG "NOTE: parked gold builder 316709 not found (already gone?)"
fi
PATH=$HOME/tools/node22/bin:$PATH PYTHONPATH=. .venv/bin/python -u -m eval.build_gold \
  --n 700 --out data/test_gold_rebuild.jsonl \
  --teachers local/qwen3.5-9b,local/gpt-oss-20b --workers 8 \
  > logs/gold_rebuild.log 2>&1 || FATAL "gold rebuild failed — see logs/gold_rebuild.log"
cat data/test_gold.jsonl data/test_gold_rebuild.jsonl > data/test_gold_merged.jsonl
G=$(wc -l < data/test_gold_merged.jsonl)
LOG "gold merged: $G items (75 old + rebuild)"
[ "$G" -lt 600 ] && LOG "WARNING: only $G gold items (<600) — eval n will be smaller"

# ---------------------------------------------------------------- phase 5
LOG "PHASE 5: stopping vLLM servers, freeing GPUs for training"
fuser -k 8000/tcp 2>/dev/null; fuser -k 8001/tcp 2>/dev/null; sleep 10
kill_gpu_engine "$GPU0_UUID"; kill_gpu_engine "$GPU1_UUID"; sleep 8
gpu_state | LOG
LOG "RUNBOOK COMPLETE — GPUs free. Next: train sweeps (see AGENTS.md), then gold+OOD eval."
