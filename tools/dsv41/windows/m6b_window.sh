#!/usr/bin/env bash
set -u
LOG=/data/works/m6b_window.log
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
CD=/data/works/Strata
restore() {
  say "restoring production"
  pkill -f 'serve.server' 2>/dev/null || true; pkill -f 'serve/server.py' 2>/dev/null || true; pkill -f 'strata-dsv41.batch' 2>/dev/null || true
  sleep 6; cd "$CD" || exit 1
  setsid nohup sh run-deepseek-v4.1-flash-ist-das-2.25bpw.sh > /data/works/serve_restore.log 2>&1 < /dev/null &
  code=000
  for i in $(seq 1 120); do code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8080/health); [ "$code" = "200" ] && break; sleep 5; done
  say "prod health=$code"; bash /data/works/services/boot.sh; sleep 3; say "supervisor: $(pgrep -f 'services/supervise.sh' | head -1)"
}
trap restore EXIT
say "stopping production"
pkill -f 'services/supervise.sh' 2>/dev/null || true; sleep 1; pkill -f 'serve/server.py' 2>/dev/null || true; sleep 5
for i in $(seq 1 50); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$used" -lt 2000 ] && break; sleep 5; done
cd "$CD" || exit 1
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64
say "batch engine (two slots, live) + the server with --api-monitor"
STRATA_DS4_BATCH_LIVE=1 setsid nohup .venv/bin/python serve/server.py --engine strata --config batch_m6.json --port 8099 --host 127.0.0.1 --api-monitor > /data/works/serve_m6b.log 2>&1 < /dev/null &
code=000
for i in $(seq 1 110); do code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8099/health); [ "$code" = "200" ] && break; sleep 4; done
say "health=$code"
if [ "$code" != "200" ]; then exit 1; fi
say "the monitor page and its panel"
printf 'api-monitor %s\n' "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://127.0.0.1:8099/api-monitor)" >> "$LOG"
curl -s --max-time 4 http://127.0.0.1:8099/api-monitor | grep -o 'id="parallel"\|id="slots"\|id="chart"\|id="p-inflight"\|id="p-busy"\|id="p-waiting"\|id="p-peak"' | sort -u >> "$LOG"
say "two clients while the status is sampled"
( .venv/bin/python tools/dsv41/batch_conc_check.py --port 8099 --clients 2 --tokens 100 --max-new 4096 > /data/works/m6b_clients.log 2>&1 ) &
clients=$!
for i in $(seq 1 10); do
  curl -s --max-time 2 http://127.0.0.1:8099/v1/status | python3 -c 'import json,sys
d = json.load(sys.stdin); c = d.get("concurrency") or {}
print("in_flight=%s slots=%s busy=%s serving=%s" % ((d.get("activity") or {}).get("in_flight"), "".join("B" if s.get("busy") else "." for s in (c.get("slots") or [])), c.get("busy"), c.get("serving")))' >> "$LOG"
  sleep 1
done
wait $clients; say "clients rc=$?"
tail -4 /data/works/m6b_clients.log >> "$LOG"
say "m6b done"