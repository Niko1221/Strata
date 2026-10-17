#!/usr/bin/env bash
# strata_community_run.sh OUTROOT VER[:CHECKOUT] ...   e.g. OUTROOT/strata-0.1.40 ~/ref/Strata-0.1.40 : the measurements of Strata's docs/COMMUNITY_BENCHMARKS.md for each Strata checkout.
# Per version: server `-c 131072` (strata_start.sh, STRATA_DIR=checkout), engine-log timing lines, RSS of the engine sampled every 2 s (rss.csv), strata_community.py (3 iterations of
# short / long / follow-up), needle_bench.py (32k + 128k at depth 50), engine log and the free-VRAM line copied. Waits for a previous server to exit first.
OUT=$1; shift
B=/mnt/f/programming/llm/qwen3.8-27b_rx7900xtx_inference/bench/decode_investigation
for VC in "$@"; do
  IFS=: read VER SD <<< "$VC"
  D=$OUT/$VER; mkdir -p $D
  pkill -x strata 2>/dev/null; pkill -x llama-server 2>/dev/null
  for p in $(ps -eo pid,args | grep '[s]erve.server --engine strata' | awk '{print $1}'); do kill $p; done   # stale python front ends hold port 8081
  sleep 2
  for w in $(seq 1 30); do pgrep -x strata >/dev/null || pgrep -x llama-server >/dev/null || break; sleep 2; done
  rm -f /tmp/strata-serve.out /tmp/strata-coder.log
  STRATA_DIR=$SD bash $B/strata_start.sh 131072 8081 >/dev/null
  for i in $(seq 1 120); do grep -q "^ready" /tmp/strata-serve.out 2>/dev/null && break; sleep 3; done
  if ! grep -q "^ready" /tmp/strata-serve.out; then echo "$VER: server did not start" | tee $D/FAILED; tail -5 /tmp/strata-serve.out >> $D/FAILED; continue; fi
  ( echo "t,rss_mib,mem_used_mib"; while pgrep -x strata >/dev/null; do echo "$(date +%s),$(ps -o rss= -C strata | sort -n | tail -1 | awk '{print int($1/1024)}'),$(free -m | awk '/Mem:/{print $3}')"; sleep 2; done ) > $D/rss.csv &
  SAMP=$!
  /usr/bin/python3 $B/strata_community.py 8081 $D 3 256 2>&1 | tee $D/community.log
  [ -z "$SKIP_NEEDLE" ] && ( cd $SD && $HOME/ref/Strata/.venv/bin/python tools/needle_bench.py --url http://127.0.0.1:8081 --lengths 32k,128k --depths 50 --out $D/needles.json > $D/needle.log 2>&1 )
  cp /tmp/strata-coder.log $D/engine.log; cp /tmp/strata-serve.out $D/serve.out
  grep -h "VRAM free\|expert cache\|filling" /tmp/strata-serve.out /tmp/strata-coder.log | head -5 > $D/vram_line.txt
  pkill -x strata; sleep 8; kill $SAMP 2>/dev/null
done
echo COMMUNITY_RUN_DONE $(date +%H:%M)
