#!/usr/bin/env bash
# 16 slots at the production 1M context: where the time goes.  Three variants against the 31.08 s baseline:
#   a) the drafter off for batch slots (STRATA_DS4_BATCH_NOSPEC=1)
#   b) a burst of 32 instead of 8 (a slot keeps its experts hot longer)
#   c) --experts 5 (17% less expert traffic a token)
set -u
LOG=/data/works/batch_opt_window.log
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
CD=/data/works/Strata
CFG=$CD/batch_opt.json

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

mkcfg() {   # $1 extra args
  python3 - "$1" "$CFG" <<'PY'
import json, sys
extra, cfg = sys.argv[1], sys.argv[2]
p = "/data/works/Strata/strata-deepseek-v4.1-flash-ist-das-2.25bpw.json"
d = json.load(open(p))
d["exe"] = "/data/works/Strata/build/strata-dsv41.batch"
args = [x for x in d["args"]]
for k in ("--batch", "--slot-burst", "--read-slice", "--experts"):
    while k in args:
        i = args.index(k)
        args[i:i + 2] = []
args += [x for x in extra.split() if x]
d["args"] = args
d["log"] = "/data/works/Strata/opt_engine.log"
d["port"] = 8099
json.dump(d, open(cfg, "w"), indent=1)
PY
}

run_case() {   # $1 label  $2 extra args  $3 env
  local label=$1 extra=$2 envv=$3
  mkcfg "$extra"
  : > "$CD/opt_engine.log"
  say "case: $label | args: $extra | env: $envv"
  env $envv setsid nohup .venv/bin/python serve/server.py --engine strata --config "$CFG" --port 8099 \
    --host 127.0.0.1 > /data/works/serve_opt.log 2>&1 < /dev/null &
  local code=000
  for i in $(seq 1 110); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8099/health)
    [ "$code" = "200" ] && break
    sleep 4
  done
  say "$label health=$code"
  if [ "$code" = "200" ]; then
    timeout 1200 .venv/bin/python tools/dsv41/batch_conc_check.py --port 8099 --clients 16 --tokens 100 \
      --max-new 262144 >> "$LOG" 2>&1
    say "$label check rc=$?"
    grep -a 'leaves over' "$CD/opt_engine.log" | tail -1 >> "$LOG" 2>/dev/null
  fi
  pkill -f 'serve/server.py' 2>/dev/null || true
  pkill -f 'strata-dsv41.batch' 2>/dev/null || true
  sleep 8
}

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

run_case "nopsec (drafter off in the batch)" "--batch 16 --slot-burst 8 --read-slice 1024" "STRATA_DS4_BATCH_LIVE=1 STRATA_DS4_BATCH_NOSPEC=1"
run_case "burst32" "--batch 16 --slot-burst 32 --read-slice 1024" "STRATA_DS4_BATCH_LIVE=1"
run_case "experts5" "--batch 16 --slot-burst 8 --read-slice 1024 --experts 5" "STRATA_DS4_BATCH_LIVE=1"
say "opt window done"
