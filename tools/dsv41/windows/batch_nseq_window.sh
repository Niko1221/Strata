#!/usr/bin/env bash
set -u
LOG=/data/works/batch_nseq_window.log
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
say "1/2 probe with two window rings (STRATA_DS4_NSEQ=2)"
STRATA_DS4_NSEQ=2 timeout 1200 .venv/bin/python tools/dsv41/batch_probe.py --exe "$ENGINE" \
  --pack /data/works/Strata-data/packs/deepseek-v4.1-flash-ist-das-2.25bpw --tokens 12 --slots 2 --timeout 1800 >> "$LOG" 2>&1
say "probe rc=$?"
pkill -f 'strata-dsv41.batch' 2>/dev/null || true
sleep 6

if grep -q 'batch_probe: all checks passed' "$LOG"; then
  say "2/2 server with two window rings, three clients"
  .venv/bin/python /data/works/mk_batch_cfg.py 2 8 "$ENGINE" 1024 >> "$LOG" 2>&1
  : > /data/works/serve_batch1.log
  : > /data/works/Strata/strata-batch1.log
  STRATA_DS4_NSEQ=2 setsid nohup .venv/bin/python serve/server.py --engine strata \
    --config /data/works/Strata/strata-batch1.json --port 8099 --host 127.0.0.1 > /data/works/serve_batch1.log 2>&1 < /dev/null &
  code=000
  for i in $(seq 1 100); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8099/health)
    [ "$code" = "200" ] && break
    sleep 5
  done
  say "server health=$code"
  timeout 900 .venv/bin/python /data/works/batch_http_check.py check >> "$LOG" 2>&1
  say "http check rc=$?"
  say "engine ERR lines: $(grep -a -c 'ERR ' /data/works/Strata/strata-batch1.log)"
else
  say "probe failed: the server check is skipped"
fi
say "window done"
