#!/usr/bin/env bash
# The burst from the measured switch (M5): two slots, live, the trace shows what it picks; three clients over HTTP.
set -u
LOG=/data/works/burst_window.log
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
CD=/data/works/Strata
CFG=$CD/batch_burst.json

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
python3 - <<'PY'
import json
p = "/data/works/Strata/strata-deepseek-v4.1-flash-ist-das-2.25bpw.json"
d = json.load(open(p))
args = [x for x in d["args"]]
i = args.index("--max-context"); args[i + 1] = "8192"
for k in ("--batch", "--slot-burst", "--read-slice"):
    while k in args:
        j = args.index(k)
        args[j:j + 2] = []
args += ["--batch", "2", "--read-slice", "1024"]     # (no --slot-burst: the engine measures it)
d["exe"] = "/data/works/Strata/build/strata-dsv41.batch"
d["args"] = args
d["log"] = "/data/works/Strata/burst_engine.log"
d["port"] = 8099
json.dump(d, open("/data/works/Strata/batch_burst.json", "w"), indent=1)
PY
say "two slots, burst measured (SLOT_TRACE on), three clients"
STRATA_DS4_BATCH_LIVE=1 STRATA_DS4_SLOT_TRACE=1 setsid nohup .venv/bin/python serve/server.py --engine strata \
  --config "$CFG" --port 8099 --host 127.0.0.1 > /data/works/serve_burst.log 2>&1 < /dev/null &
code=000
for i in $(seq 1 110); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8099/health)
  [ "$code" = "200" ] && break
  sleep 4
done
say "health=$code"
if [ "$code" = "200" ]; then
  timeout 900 .venv/bin/python tools/dsv41/batch_conc_check.py --port 8099 --clients 3 --tokens 100 --max-new 4096 >> "$LOG" 2>&1
  say "rc=$?"
  grep -a 'burst ' /data/works/serve_burst.log | tail -8 >> "$LOG"
fi
pkill -f 'serve/server.py' 2>/dev/null || true
pkill -f 'strata-dsv41.batch' 2>/dev/null || true
sleep 6
say "burst window done"
