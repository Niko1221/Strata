#!/bin/bash
# ab-arm.sh ARM ROUND — one A/B arm: start ONE Strata server (NUMA node 1, GPU-local), bench, stop, verify GPU free.
# ARM = a|b (a: STRATA_DENSE_F32=0, b: =1), ROUND = 1|2. Never two instances at once.
set -u
ARM=$1; ROUND=$2; U=${ARM^^}
DIR=~/Strata/ab-dense-f32
PORT=18099
PY=~/Strata/.venv/bin/python
BENCH=~/github/Strata/bench/results/2026-10-12-p40-sm61-kernels/bench.py
snap(){ nvidia-smi --query-gpu=temperature.gpu,clocks.gr,memory.used,power.draw --format=csv,noheader; }

# cool to idle temp first (P40 lesson: same starting temp per arm; capped at 4 min)
for i in $(seq 1 48); do T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "$T" -le 58 ] && break; sleep 5; done
echo "== arm $U round $ROUND start $(date '+%F %T') start-temp $T C: $(snap)"

numactl --preferred=1 --cpunodebind=1 $PY ~/Strata/serve/server.py \
  --engine strata --config $DIR/config-$ARM.json --port $PORT > $DIR/server-$ARM$ROUND.out 2>&1 &
SPID=$!
ready=0
for i in $(seq 1 300); do
  sleep 3
  curl -sf -m 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { ready=1; break; }
  kill -0 $SPID 2>/dev/null || { echo "SERVER DIED"; tail -8 $DIR/server-$ARM$ROUND.out; exit 1; }
done
[ $ready = 1 ] || { echo "NOT READY after 15 min"; kill -TERM $SPID; exit 1; }
echo "== ready after ~$((i*3))s $(date '+%T'): $(snap)"

# warm-up: starts the engine + loads the model (lazy) and creates the log, pays the PCIe probe,
# so no bench request carries one-time work
curl -s -m 900 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d \
  '{"model":"qwen3.8-flash-next-iq3_s","messages":[{"role":"user","content":"warmup '$RANDOM$RANDOM'"}],"max_tokens":4,"temperature":0,"reasoning_effort":"none"}' -o /dev/null
echo "== warm (model loaded) $(date '+%T'): $(snap)"

BENCH_PORT=$PORT taskset -c 22-43,66-87 $PY $BENCH $DIR/ab-$ARM.log "${U}${ROUND}" --runs 3
RC=$?
echo "== arm $U round $ROUND bench done rc=$RC $(date '+%T') end: $(snap)"

kill -TERM $SPID 2>/dev/null
for i in $(seq 1 90); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
kill -0 $SPID 2>/dev/null && { kill -9 $SPID; sleep 3; }
sleep 4
echo "== gpu after stop: $(snap)"
LEFT=$(pgrep -af "engine-cuda12/strata|build-cuda/strata|serve/server.py" | grep -v grep)
[ -n "$LEFT" ] && { echo "LEFTOVER: $LEFT"; exit 1; }
echo "== arm $U round $ROUND clean"
