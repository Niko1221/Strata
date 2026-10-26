#!/usr/bin/env bash
# Is a single tensor-parallel group viable for M2's step?  The engine at --tp 4 (one group of four cards: no source is
# read from another group, so the step's cross-group hand-off is not needed), the probe with two slots and a 3-client run.
set -u
LOG=/data/works/tp4_window.log
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
CD=/data/works/Strata
CFG=$CD/batch_tp4.json

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
say "gpu max used=$used MiB"
cd "$CD" || exit 1
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64

say "1/2 the probe at --tp 4 (one group), two slots"
timeout 1500 .venv/bin/python tools/dsv41/batch_probe.py --exe "$CD/build/strata-dsv41.batch" \
  --pack /data/works/Strata-data/packs/deepseek-v4.1-flash-ist-das-2.25bpw --tokens 12 --slots 2 --timeout 1800 \
  --extra "--tp 4" >> "$LOG" 2>&1
say "probe rc=$?"
pkill -f 'strata-dsv41.batch' 2>/dev/null || true
sleep 8

if grep -q 'batch_probe: all checks passed' "$LOG"; then
  say "2/2 three clients over HTTP at --tp 4"
  python3 - <<'PY'
import json
p = "/data/works/Strata/strata-deepseek-v4.1-flash-ist-das-2.25bpw.json"
d = json.load(open(p))
args = [x for x in d["args"]]
i = args.index("--max-context"); args[i + 1] = "8192"
i = args.index("--tp"); args[i + 1] = "4"
for k in ("--batch", "--slot-burst", "--read-slice"):
    while k in args:
        j = args.index(k)
        args[j:j + 2] = []
args += ["--batch", "2", "--slot-burst", "8", "--read-slice", "1024"]
d["exe"] = "/data/works/Strata/build/strata-dsv41.batch"
d["args"] = args
d["log"] = "/data/works/Strata/tp4_engine.log"
d["port"] = 8099
json.dump(d, open("/data/works/Strata/batch_tp4.json", "w"), indent=1)
PY
  STRATA_DS4_BATCH_LIVE=1 setsid nohup .venv/bin/python serve/server.py --engine strata --config "$CFG" --port 8099 \
    --host 127.0.0.1 > /data/works/serve_tp4.log 2>&1 < /dev/null &
  code=000
  for i in $(seq 1 110); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8099/health)
    [ "$code" = "200" ] && break
    sleep 4
  done
  say "tp4 server health=$code"
  if [ "$code" = "200" ]; then
    timeout 900 .venv/bin/python tools/dsv41/batch_conc_check.py --port 8099 --clients 3 --tokens 100 --max-new 4096 >> "$LOG" 2>&1
    say "3-client rc=$?"
  fi
else
  say "the probe failed at --tp 4"
fi
pkill -f 'serve/server.py' 2>/dev/null || true
pkill -f 'strata-dsv41.batch' 2>/dev/null || true
sleep 6
say "tp4 window done"
