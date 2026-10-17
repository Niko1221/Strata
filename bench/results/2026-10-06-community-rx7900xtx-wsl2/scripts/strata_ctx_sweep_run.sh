#!/usr/bin/env bash
# strata_ctx_sweep_run.sh OUTROOT VER:CHECKOUT ...: per Strata checkout, start the server with -c 262144 (strata_start.sh), RSS sampled every 2 s, run strata_ctx_sweep.py, keep the engine log.
# Env passes through (STRATA_PREFILL_STREAM_MIN=65536 for the third configuration). Waits for a previous server to exit first.
OUT=$1; shift
B=/mnt/f/programming/llm/qwen3.8-27b_rx7900xtx_inference/bench/decode_investigation
for VC in "$@"; do
  IFS=: read VER SD <<< "$VC"
  D=$OUT/$VER; mkdir -p $D
  pkill -x strata 2>/dev/null; pkill -x llama-server 2>/dev/null
  for p in $(ps -eo pid,args | grep '[s]erve.server --engine strata' | awk '{print $1}'); do kill $p; done
  sleep 2
  for w in $(seq 1 30); do pgrep -x strata >/dev/null || pgrep -x llama-server >/dev/null || break; sleep 2; done
  rm -f /tmp/strata-serve.out /tmp/strata-coder.log
  STRATA_DIR=$SD bash $B/strata_start.sh 262144 8081 >/dev/null
  for i in $(seq 1 120); do grep -q "^ready" /tmp/strata-serve.out 2>/dev/null && break; sleep 3; done
  if ! grep -q "^ready" /tmp/strata-serve.out; then echo "$VER: server did not start at -c 262144" | tee $D/FAILED; tail -8 /tmp/strata-serve.out /tmp/strata-coder.log >> $D/FAILED 2>/dev/null; continue; fi
  ( echo "t,rss_mib,mem_used_mib"; while pgrep -x strata >/dev/null; do echo "$(date +%s),$(ps -o rss= -C strata | sort -n | tail -1 | awk '{print int($1/1024)}'),$(free -m | awk '/Mem:/{print $3}')"; sleep 2; done ) > $D/rss.csv &
  SAMP=$!
  SRC_DIR=$SD /usr/bin/python3 $B/strata_ctx_sweep.py 8081 $D 262144 ${SWEEP_N:-2} 256 2>&1 | tee $D/sweep.log
  cp /tmp/strata-coder.log $D/engine.log; cp /tmp/strata-serve.out $D/serve.out
  grep -h "VRAM free\|expert cache\|filling" /tmp/strata-serve.out /tmp/strata-coder.log | head -5 > $D/vram_line.txt
  pkill -x strata; sleep 8; kill $SAMP 2>/dev/null
done
echo SWEEP_RUN_DONE $(date +%H:%M)
