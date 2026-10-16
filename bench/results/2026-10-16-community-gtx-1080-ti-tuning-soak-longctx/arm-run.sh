#!/bin/bash
# arm-run.sh CFG LABEL - ONE Strata instance (NUMA node 1), warm, quality probe, bench.py, teardown, verify clean.
# Refuses to start if any engine is already running (single-instance rule).
set -u
CFG=$1; LABEL=$2
DIR=~/Strata/ab-dense-f32
PORT=18099
PY=~/Strata/.venv/bin/python
BENCH=~/github/Strata/bench/results/2026-10-12-p40-sm61-kernels/bench.py
LOG=$(python3 -c "import json;print(json.load(open('$CFG'))['log'])")
touch "$LOG"   # bench.py counts lines before each request; the engine appends here
WARM_MAX=${WARM_MAX:-600}
READY_MAX=${READY_MAX:-600}
snap(){ nvidia-smi --query-gpu=temperature.gpu,clocks.gr,memory.used,power.draw --format=csv,noheader; }

engine_busy(){ pgrep -x strata >/dev/null 2>&1 || nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -q "strata"; }
if engine_busy; then
  echo "!! $LABEL: another ENGINE is running - waiting up to 5 min (single-instance rule)"
  for i in $(seq 1 60); do engine_busy || break; sleep 5; done
  if engine_busy; then echo "!! $LABEL: engine still running, aborting"; pgrep -ax strata; exit 1; fi
  echo "== $LABEL: the other engine is gone, continuing"
fi
if ss -tln | grep -q ":$PORT "; then echo "!! port $PORT busy"; exit 1; fi
for i in $(seq 1 48); do T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "$T" -le 58 ] && break; sleep 5; done
echo "== $LABEL start $(date '+%T') start-temp $T C: $(snap)"

numactl --preferred=1 --cpunodebind=1 $PY ~/Strata/serve/server.py \
  --engine strata --config "$CFG" --port $PORT > $DIR/logs/server-$LABEL.out 2>&1 &
SPID=$!
ready=0
for i in $(seq 1 $((READY_MAX/3))); do
  sleep 3
  curl -sf -m 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { ready=1; break; }
  kill -0 $SPID 2>/dev/null || { echo "!! $LABEL SERVER DIED"; tail -6 $DIR/logs/server-$LABEL.out; exit 1; }
done
[ $ready = 1 ] || { echo "!! $LABEL not ready in $READY_MAX s (hang? bug #2126?)"; kill -TERM $SPID 2>/dev/null; sleep 5; kill -9 $SPID 2>/dev/null; exit 1; }
echo "== $LABEL ready ~$((i*3))s $(date '+%T')"

curl -s -m $WARM_MAX http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d \
  "{\"model\":\"qwen3.8-flash-next-iq3_s\",\"messages\":[{\"role\":\"user\",\"content\":\"warmup $RANDOM$RANDOM\"}],\"max_tokens\":4,\"temperature\":0,\"reasoning_effort\":\"none\"}" -o /dev/null || true
echo "== $LABEL warm $(date '+%T'): $(snap)"

# fixed quality probe: 17*23 + 41*19 = 1170 (temperature 0, 48 tokens)
curl -s -m 120 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d \
  '{"model":"qwen3.8-flash-next-iq3_s","messages":[{"role":"user","content":"Compute 17*23 + 41*19. Answer with just the number."}],"max_tokens":48,"temperature":0,"reasoning_effort":"none"}' \
  | python3 -c 'import json,sys; print("QUALITY:", json.load(sys.stdin)["choices"][0]["message"]["content"].strip()[:80])' > $DIR/logs/quality-$LABEL.txt 2>&1
cat $DIR/logs/quality-$LABEL.txt

if [ "${USE_BURST:-0}" = "1" ]; then
  BENCH_PORT=$PORT taskset -c 22-43,66-87 $PY $DIR/burst.py 2
elif [ "${USE_SOAK:-0}" = "1" ]; then
  BENCH_PORT=$PORT taskset -c 22-43,66-87 $PY $DIR/soak.py "${SOAK_MIN:-60}"
elif [ "${USE_LONG:-0}" = "1" ]; then
  BENCH_LOG="$LOG" BENCH_PORT=$PORT taskset -c 22-43,66-87 $PY $DIR/longbench.py "$LABEL" ${LONG_RUNS:-2 2 1}
  if [ "${NEEDLES:-0}" = "1" ]; then
    $PY ~/github/Strata/tools/needle_bench.py --url http://127.0.0.1:$PORT --lengths 128k,250k --depths 50 --out $DIR/logs/needles-$LABEL.json
  fi
else
  BENCH_PORT=$PORT taskset -c 22-43,66-87 $PY $BENCH $LOG "$LABEL" --runs 3
fi

echo "== $LABEL done $(date '+%T') end: $(snap)"
grep -m1 -oE "pcie[_ -]?(frac|probe)[^,;]*" $LOG | head -1
kill -TERM $SPID 2>/dev/null
for i in $(seq 1 90); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
kill -0 $SPID 2>/dev/null && { kill -9 $SPID; sleep 3; }
sleep 4
if engine_busy; then echo "!! $LABEL LEFTOVER ENGINE"; pgrep -ax strata; exit 1; fi
echo "== $LABEL clean"
