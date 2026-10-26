#!/usr/bin/env bash
# One sequence on its own answers what sequence 0 does (M2's regions): the parity check in tools/dsv41/, two engine loads
# (sequence 0, then STRATA_DS4_VSEQ=1), production stopped for the duration and restored by the trap.
set -u
LOG=/data/works/batch_seq_window.log
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
CD=/data/works/Strata
ENGINE=/data/works/Strata/build/strata-dsv41.batch
PACK=/data/works/Strata-data/packs/deepseek-v4.1-flash-ist-das-2.25bpw

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
say "the parity check: sequence 0, then sequence 1, on the same prompts"
timeout 2400 .venv/bin/python tools/dsv41/batch_seq_parity.py --exe "$ENGINE" --pack "$PACK" \
  --tokens 12 --long 140 --timeout 1800 >> "$LOG" 2>&1
say "parity rc=$?"
pkill -f 'strata-dsv41.batch' 2>/dev/null || true
sleep 6
say "window done"
