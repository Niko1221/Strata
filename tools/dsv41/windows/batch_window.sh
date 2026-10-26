#!/usr/bin/env bash
set -u
LOG=/data/works/batch_window.log
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

restore() {
  say "restoring production"
  pkill -f 'strata-dsv41.batch' 2>/dev/null || true
  pkill -f 'serve/server.py' 2>/dev/null || true
  sleep 6
  cd /data/works/Strata || exit 1
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
used=99999
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$used" -lt 2500 ] && break
  sleep 5
done
say "gpu max used=$used MiB"

cd /data/works/Strata || exit 1
say "1/2 engine protocol probe (one load)"
timeout 1200 .venv/bin/python tools/dsv41/batch_probe.py --exe /data/works/Strata/build/strata-dsv41.batch \
  --pack /data/works/Strata-data/packs/deepseek-v4.1-flash-ist-das-2.25bpw --tokens 12 --slots 2 --timeout 1800 >> "$LOG" 2>&1
probe_rc=$?
say "probe rc=$probe_rc"
pkill -f 'strata-dsv41.batch' 2>/dev/null || true
sleep 6

if [ "$probe_rc" = "0" ]; then
  say "2/2 server check (second load)"
  .venv/bin/python /data/works/mk_batch_cfg.py >> "$LOG" 2>&1
  STRATA_BATCH_TRACE=1 setsid nohup .venv/bin/python serve/server.py --engine strata \
    --config /data/works/Strata/strata-batch1.json --port 8099 --host 127.0.0.1 > /data/works/serve_batch1.log 2>&1 < /dev/null &
  code=000
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8099/health)
    [ "$code" = "200" ] && break
    sleep 5
  done
  say "batch server health=$code"
  timeout 1200 .venv/bin/python /data/works/batch_http_check.py check >> "$LOG" 2>&1
  say "http check rc=$?"
  say "--- the batch lines the server sent ---"
  grep -a -c 'BGEN' /data/works/serve_batch1.log >> "$LOG" 2>&1
  grep -a -c 'BADM' /data/works/serve_batch1.log >> "$LOG" 2>&1
  grep -a -c 'BT 0 ' /data/works/serve_batch1.log >> "$LOG" 2>&1
  grep -a -m 4 'BADM' /data/works/serve_batch1.log >> "$LOG" 2>&1
  say "--- server log tail ---"
  tail -12 /data/works/serve_batch1.log >> "$LOG" 2>&1
else
  say "probe failed: the server check is skipped this window"
fi
say "window done"