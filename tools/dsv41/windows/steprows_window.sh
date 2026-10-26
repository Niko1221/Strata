#!/usr/bin/env bash
# The per-row chunk path (STRATA_DS4_STEPROWS=1) against the contiguous one: the same 300-token prompt, small chunks.
set -u
LOG=/data/works/steprows_window.log
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
CD=/data/works/Strata

restore() {
  say "restoring production"
  pkill -f 'serve.server' 2>/dev/null || true
  pkill -f 'serve/server.py' 2>/dev/null || true
  pkill -f 'strata-dsv41.batch' 2>/dev/null || true
  sleep 6
  cd "$CD" || exit 1
  setsid nohup sh run-deepseek-v4.1-flash-ist-das-2.25bpw.sh > /data/works/serve_restore.log 2>&1 < /dev/null &
  code=000
  for i in $(seq 1 120); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8080/health)
    [ "$code" = "200" ] && break
    sleep 5
  done
  say "prod health=$code"
  bash /data/works/services/boot.sh
  sleep 3
  say "supervisor: $(pgrep -f 'services/supervise.sh' | head -1)"
}
trap restore EXIT

say "stopping production"
pkill -f 'services/supervise.sh' 2>/dev/null || true
sleep 1
pkill -f 'serve/server.py' 2>/dev/null || true
sleep 5
for i in $(seq 1 50); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$used" -lt 2000 ] && break
  sleep 5
done
cd "$CD" || exit 1
say "the per-row chunk path against the contiguous one (300-token prompt, --prefill-chunk 8)"
timeout 1500 .venv/bin/python tools/dsv41/steprows_check.py --exe "$CD/build/strata-dsv41.batch" \
  --pack /data/works/Strata-data/packs/deepseek-v4.1-flash-ist-das-2.25bpw --tokens 12 --timeout 1800 --steprows 2 >> "$LOG" 2>&1
say "steprows rc=$?"
tail -6 /data/works/steprows_engine.log >> "$LOG" 2>/dev/null
pkill -f 'strata-dsv41.batch' 2>/dev/null || true
sleep 6
say "window done"
