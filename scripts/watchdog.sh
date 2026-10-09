#!/usr/bin/env bash
# Periodic health monitor for the gen/verify/gold pipeline + vLLM servers.
# Detects: stalled stages, early runbook exit, orphan vLLM holding VRAM,
#          server-needing stages with a dead port.
# Artifacts: logs/watchdog_status.txt (live snapshot), logs/watchdog.log (alerts),
#            logs/.watchdog_state (progress bookkeeping), logs/.watchdog_alert (mtime = last alert)
# Usage: scripts/watchdog.sh [--once]   (interval: FCFT_WATCHDOG_INTERVAL sec, default 300)
cd "$(dirname "$0")/.."
INTERVAL=${FCFT_WATCHDOG_INTERVAL:-300}
STALL_SEC=$((25 * 60))
ALERT_REPEAT_SEC=$((30 * 60))
STATE=logs/.watchdog_state
STATUS=logs/watchdog_status.txt
ALOG=logs/watchdog.log
touch "$STATE"

ALERT() {  # $1 = alert key, $2 = message; throttled per key
  local key=$1 msg=$2 now last
  now=$(date +%s)
  last=$(grep "^$key|" "$STATE" 2>/dev/null | cut -d'|' -f3)
  if [ -n "$last" ] && [ $((now - last)) -lt $ALERT_REPEAT_SEC ]; then return; fi
  echo "[$(date '+%m-%d %H:%M:%S')] ALERT $key: $msg" | tee -a "$ALOG"
  sed -i "/^$key|/d" "$STATE"; echo "$key|alert|$now" >> "$STATE"
  touch logs/.watchdog_alert
}

progress() {  # $1 = metric key, $2 = current value -> stall detection
  local key=$1 val=$2 now ent oval epoch
  now=$(date +%s)
  ent=$(grep "^$key|" "$STATE" 2>/dev/null)
  oval=${ent%%|*}; oval=${ent#*|}; oval=${oval%%|*}; epoch=${ent##*|}
  if [ "$val" != "$oval" ] || [ -z "$epoch" ]; then
    sed -i "/^$key|/d" "$STATE"; echo "$key|$val|$now" >> "$STATE"; return
  fi
  if [ $((now - epoch)) -gt "$STALL_SEC" ]; then
    ALERT "stall_$key" "no progress for $(( (now - epoch) / 60 ))min (metric=$val)"
  fi
}

active() {  # $1 = pgrep pattern -> PIDs excluding stopped (T) procs
  for p in $(pgrep -f "$1" 2>/dev/null); do
    case $(ps -o stat= -p "$p" 2>/dev/null) in T*) ;; *) echo "$p";; esac
  done
}

cycle() {
  local gen ver gold rb engines h0 h1 g0 g1 done_line rb_last tr sw tbytes sw_last au au_last
  gen=$(active  "[d]ataset_gen"   | wc -l)
  ver=$(active  "[s]rc.verify"    | wc -l)
  gold=$(active "[b]uild_gold"    | wc -l)
  rb=$(active   "[r]unbook_post_gen" | wc -l)
  tr=$(active   "[t]rain_sft"     | wc -l)
  sw=$(active   "[r]unbook_train_sweep" | wc -l)
  au=$(active   "[a]uto_finalists" | wc -l)
  al=$(active   "[a]uto_llama_restart" | wc -l)
  ao=$(active   "[a]uto_ood" | wc -l)
  engines=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  g0=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i 0)
  g1=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i 1)
  h0=$(curl -s -m 3 localhost:8000/v1/models 2>/dev/null | grep -o "gpt-oss-20b" | head -1); [ -n "$h0" ] || h0=DOWN
  h1=$(curl -s -m 3 localhost:8001/v1/models 2>/dev/null | grep -o "qwen3.5-9b"  | head -1); [ -n "$h1" ] || h1=DOWN
  rb_last=$(tail -1 logs/runbook.log 2>/dev/null)
  sw_last=$(tail -1 logs/train_sweep.log 2>/dev/null)
  au_last=$(tail -1 logs/auto_finalists.log 2>/dev/null)
  al_last=$(tail -1 logs/auto_llama_restart.log 2>/dev/null)

  # --- progress metrics for active stages ---
  [ "$gen"  -gt 0 ] && progress gen   "$(wc -l < data/raw_trajectories/gen_local.jsonl 2>/dev/null || echo 0)"
  [ "$ver"  -gt 0 ] && progress verif "$(stat -c %s logs/verify_final.log 2>/dev/null || echo 0)"
  [ "$gold" -gt 0 ] && progress goldb "$(stat -c %s logs/gold_rebuild.log 2>/dev/null || echo 0)"
  tbytes=$(du -cb logs/train_*.log 2>/dev/null | tail -1 | cut -f1); [ -n "$tbytes" ] || tbytes=0
  [ "$tr"   -gt 0 ] && progress train "$tbytes"

  # --- rules ---
  if [ "$rb" -eq 0 ] && [ -f logs/runbook.log ] && ! grep -q "RUNBOOK COMPLETE" logs/runbook.log; then
    ALERT runbook_dead "runbook exited without COMPLETE — last: $rb_last"
  elif grep -q "RUNBOOK COMPLETE" logs/runbook.log 2>/dev/null; then
    :  # pipeline finished; watchdog is just idle now
  elif [ "$engines" -gt 0 ] && [ "$rb" -eq 0 ] && [ "$gen" -eq 0 ] && [ "$ver" -eq 0 ] && [ "$gold" -eq 0 ] \
       && [ "$tr" -eq 0 ] && [ "$sw" -eq 0 ] && [ "$au" -eq 0 ]; then
    ALERT orphan_vram "$engines CUDA proc(s) holding VRAM but nothing is running (runbook alive=$rb)"
  fi
  if [ "$sw" -eq 0 ] && [ -f logs/train_sweep.log ] && ! grep -q "SWEEP COMPLETE" logs/train_sweep.log; then
    ALERT sweep_dead "train sweep exited without COMPLETE — last: $sw_last"
  fi
  if [ "$au" -eq 0 ] && [ -f logs/auto_finalists.log ] \
     && ! grep -q "FINALISTS_DONE" logs/auto_finalists.log; then
    ALERT auto_dead "auto_finalists exited without FINALISTS_DONE — last: $au_last"
  fi
  if [ "$ao" -eq 0 ] && [ -f logs/auto_ood.log ] && ! grep -q "OOD_DONE" logs/auto_ood.log; then
    ALERT ood_watch_dead "auto_ood exited without OOD_DONE — last: $(tail -1 logs/auto_ood.log 2>/dev/null)"
  fi
  if [ "$al" -eq 0 ] && [ -f logs/auto_llama_restart.log ] \
     && ! grep -q "LLAMA_RESTARTED" logs/auto_llama_restart.log; then
    ALERT llama_watch_dead "auto_llama_restart exited without LLAMA_RESTARTED — last: $al_last"
  fi
  if { [ "$ver" -gt 0 ] || [ "$gold" -gt 0 ]; } && { [ "$h0" = DOWN ] || [ "$h1" = DOWN ]; }; then
    ALERT server_down "active stage needs both servers — :8000=$h0 :8001=$h1"
  fi
  if [ "$gen" -gt 0 ] && [ "$h0" = DOWN ] && [ "$h1" = DOWN ]; then
    ALERT gen_no_servers "dataset_gen running but BOTH servers down"
  fi

  # --- snapshot ---
  {
    echo "=== watchdog @ $(date '+%m-%d %H:%M:%S') ==="
    echo "procs: runbook=$rb gen=$gen verify=$ver gold=$gold train=$tr sweep=$sw auto=$au llamar=$al  cuda_procs=$engines"
    echo "gpu0: $g0 | gpu1: $g1"
    echo ":8000=$h0 :8001=$h1"
    echo "runbook: ${rb_last:-<none>}"
    echo "sweep: ${sw_last:-<none>}"
    echo "auto: ${au_last:-<none>}"
    wcl() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
    echo "rows: gen_local=$(wcl data/raw_trajectories/gen_local.jsonl) verified_final=$(wcl data/verified/gen_final_verified.jsonl) gold_rebuild=$(wcl data/test_gold_rebuild.jsonl)"
    echo "alerts: $(grep -c ALERT "$ALOG" 2>/dev/null || echo 0) total (see logs/watchdog.log)"
  } > "$STATUS"
}

while :; do
  cycle
  [ "${1:-}" = "--once" ] && break
  sleep "$INTERVAL"
done
