#!/usr/bin/env bash
# strata_env_ab.sh OUTROOT CHECKOUT NAME=ENVSPEC ...  e.g. ring0='STRATA_RING_BYTES=0'  : one server of the given Strata checkout per environment setting (-c 32768), one iteration of
# strata_community.py (short / long cold / follow-up) each, to find which setting restores the prefill speed of 0.1.33 on gfx1100 (0.1.40 "prompt chunk auto: 8192 tokens, a 96-slot ring").
OUT=$1; SD=$2; shift 2
B=/mnt/f/programming/llm/qwen3.8-27b_rx7900xtx_inference/bench/decode_investigation
for A in "$@"; do
  NAME=${A%%=*}; ENVS=${A#*=}
  D=$OUT/$NAME; mkdir -p $D
  pkill -x strata 2>/dev/null
  for p in $(ps -eo pid,args | grep '[s]erve.server --engine strata' | awk '{print $1}'); do kill $p; done
  for w in $(seq 1 30); do pgrep -x strata >/dev/null || break; sleep 2; done; sleep 2
  rm -f /tmp/strata-serve.out /tmp/strata-coder.log
  env $ENVS STRATA_DIR=$SD bash $B/strata_start.sh 32768 8081 >/dev/null
  for i in $(seq 1 120); do grep -q "^ready" /tmp/strata-serve.out 2>/dev/null && break; sleep 3; done
  if ! grep -q "^ready" /tmp/strata-serve.out; then echo "$NAME: server did not start" > $D/FAILED; continue; fi
  echo "== $NAME ($ENVS): $(grep -h 'prompt chunk' /tmp/strata-coder.log | head -1)"
  /usr/bin/python3 $B/strata_community.py 8081 $D 1 256 2>&1 | grep -E "long|followup" | cut -c1-200
  cp /tmp/strata-coder.log $D/engine.log
  pkill -x strata; sleep 6
done
for p in $(ps -eo pid,args | grep '[s]erve.server --engine strata' | awk '{print $1}'); do kill $p; done
echo ENV_AB_DONE
