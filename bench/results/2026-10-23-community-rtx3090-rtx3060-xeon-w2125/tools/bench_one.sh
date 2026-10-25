#!/bin/bash
# usage: bench_one.sh OUTDIR LABEL RUNSCRIPT PORT ENGINE_DIR
# Starts a Strata server from RUNSCRIPT, waits for health, runs the existing bench.py (depths 1K warmup + 150K/200K/250K, no needles),
# records version/config/VRAM/log findings, stops the server. Never prints or stores the API key.
OUT=$1; LABEL=$2; RUN=$3; PORT=$4; ED=$5
KEYFILE=~/strata-src/strata-api-key
mkdir -p "$OUT"; cd "$OUT"
{ date -Is; EX=$ED/engine/strata; [ -x "$EX" ] || EX=$ED/build/strata; echo "engine: $($EX --version 2>&1 | head -1) ($EX)"; git -C "$ED" log -1 --format='commit %H %ad' --date=iso; git -C "$ED" describe --tags --always; } > version.txt 2>&1
CFG=$(grep -o -- '--config" "[^"]*' "$RUN" | sed 's/.*"//'); echo "run script: $RUN" >> version.txt; echo "config: $CFG" >> version.txt
python3 -I -c "
import json,sys
c=json.load(open('$CFG'))
c.pop('api_key',None)
json.dump(c,open('config_redacted.json','w'),indent=1)"
free -g > free_before_start.txt; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv > vram_before_start.txt
SWAP0=$(awk '/^pswpout/{print $2}' /proc/vmstat); echo "pswpout_start $SWAP0" > swap_counters.txt
setsid nohup "$RUN" > server.log 2>&1 < /dev/null &
for i in $(seq 1 360); do curl -s -m 3 http://127.0.0.1:$PORT/health | grep -q '"loaded": true' && break; sleep 5; done
curl -s -m 3 http://127.0.0.1:$PORT/health > health.txt; echo >> health.txt
if ! grep -q '"loaded": true' health.txt; then echo "SERVER DID NOT BECOME HEALTHY" > FAILED; exit 1; fi
sleep 20; free -g > free_after_load.txt; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > vram_after_load.txt
python3 -I ~/strata-bench/tools/bench.py --base-url http://127.0.0.1:$PORT --api-key-file $KEYFILE --model "$(python3 -I -c "import json;print(json.load(open('$CFG')).get('model_name','strata'))")" \
  --depths 1000,150000,200000,250000 --cold-runs 3 --ckpt-interval 16384 --label "$LABEL" --out-dir "$OUT" > bench.log 2>&1
echo "bench_exit $?" >> bench.log
echo "pswpout_end $(awk '/^pswpout/{print $2}' /proc/vmstat)" >> swap_counters.txt
free -g > free_after_bench.txt; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > vram_after_bench.txt
grep -inE 'swap|oom|out of memory|stopped unexpectedly|restart|killed|cuda error|traceback|fatal|abort' server.log "$(python3 -I -c "import json;print(json.load(open('$CFG')).get('log',''))")" > log_findings.txt 2>&1
sudo -n dmesg | grep -iE 'out of memory|oom-kill|killed process' > dmesg_oom.txt 2>&1
pkill -f "serve/server.py --engine strata --config $CFG" ; for i in $(seq 40); do pgrep -f "engine/strata --serve" >/dev/null || break; sleep 3; done
touch DONE
