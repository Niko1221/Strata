#!/usr/bin/env bash
# The measured batch geometry (M5) on this box: the same build at two contexts, the INFO line its own numbers.
set -u
LOG=/data/works/batch_plan_window.log
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
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64
PACK=/data/works/Strata-data/packs/deepseek-v4.1-flash-ist-das-2.25bpw
for ctx in 8192 1048576; do
  say "context $ctx, --batch 16"
  timeout 420 ./build/strata-dsv41.batch --serve --pack "$PACK" --max-context "$ctx" --eos 1 --dspark auto \
    --tp 2 --batch 16 --slot-burst 8 --read-slice 1024 < /dev/null > "/data/works/plan_$ctx.log" 2>&1
  grep -a 'INFO batch_' "/data/works/plan_$ctx.log" >> "$LOG"
  grep -a 'free' "/data/works/plan_$ctx.log" | tail -4 >> "$LOG"
done
say "plan window done"
