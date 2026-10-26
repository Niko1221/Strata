#!/usr/bin/env bash
set -u
LOG=/data/works/batch_turn_window.log
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
CD=/data/works/Strata
ENGINE=/data/works/Strata/build/strata-dsv41.batch

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

pkill -f -- '--port 8099' 2>/dev/null || true
sleep 2
if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8099/health; then
  say "port 8099 is still taken: aborting"
  exit 1
fi

say "stopping production"
pkill -f 'services/supervise.sh' 2>/dev/null || true
sleep 1
pkill -f 'serve/server.py' 2>/dev/null || true
sleep 5
used=99999
for i in $(seq 1 40); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$used" -lt 2500 ] && break
  sleep 5
done
say "gpu max used=$used MiB"

cd "$CD" || exit 1
.venv/bin/python /data/works/mk_batch_cfg.py 2 8 "$ENGINE" 1024 >> "$LOG" 2>&1
: > /data/works/serve_batch1.log
: > /data/works/Strata/strata-batch1.log
STRATA_DS4_SLOT_TRACE=1 setsid nohup .venv/bin/python serve/server.py --engine strata \
  --config /data/works/Strata/strata-batch1.json --port 8099 --host 127.0.0.1 > /data/works/serve_batch1.log 2>&1 < /dev/null &
code=000
for i in $(seq 1 100); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8099/health)
  [ "$code" = "200" ] && break
  sleep 5
done
say "server health=$code"

timeout 900 .venv/bin/python /data/works/batch_turn_check.py check >> "$LOG" 2>&1
say "turn check rc=$?"
say "--- the engine's prompt lines (reused / read) ---"
grep -a -m 6 'serve: prompt ' /data/works/Strata/strata-batch1.log >> "$LOG" 2>&1
timeout 900 .venv/bin/python /data/works/batch_http_check.py check >> "$LOG" 2>&1
say "http check rc=$?"
say "--- conversations set aside / taken back ---"
grep -a -m 6 'saved a .*conversation\|took the conversation\|take' /data/works/Strata/strata-batch1.log >> "$LOG" 2>&1
say "window done"