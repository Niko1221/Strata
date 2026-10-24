#!/bin/sh
# community benchmark for a GitHub report: bench.sh --long (starts its own server from the production config) and the
# needle recall test on a fresh server, both inside the launcher's memory caps; production server restarted at the end
S=<strata>
OUT=<strata>/bench/results/.report-20261021
CAPS="-p MemoryHigh=23G -p MemoryMax=25G -p MemorySwapMax=2G"
cd $S || exit 1
# stop the production server
P=$(ps -eo pid,args | grep "[s]erver.py --engine strata" | awk '{print $1}' | head -1); E=$(pgrep -x strata | head -1)
[ -n "$P" ] && kill $P; for i in $(seq 1 60); do kill -0 $P 2>/dev/null || kill -0 $E 2>/dev/null || break; sleep 1; done
rm -rf $OUT; mkdir -p $OUT
echo "== bench --long $(date +%T)"
systemd-run --user --scope --quiet $CAPS ./bench.sh --config strata-iq2_xs.json --long --out $OUT < /dev/null
echo "== bench exit $? $(date +%T)"
sleep 5
echo "== needle: starting server $(date +%T)"
systemd-run --user --scope --quiet $CAPS .venv/bin/python serve/server.py --engine strata --config <strata>/strata-iq2_xs.json --port 8085 > $OUT/needle-server.out 2>&1 &
for i in $(seq 1 150); do curl -s -m 3 http://127.0.0.1:8085/health | grep -q '"loaded": true' && break; sleep 2; done
DIR=$(ls -d $OUT/*/ | head -1)
.venv/bin/python tools/needle_bench.py --url http://127.0.0.1:8085 --lengths 8k,32k,56k --depths 10,50,90 --timeout 900 --out $DIR/needles.json
# (the measured run passed the folder itself to --out, so needle_bench could not write it; needles.json was transcribed
#  from the console output of that run)
echo "== needle exit $? $(date +%T)"
P=$(ps -eo pid,args | grep "[s]erver.py --engine strata" | awk '{print $1}' | head -1); E=$(pgrep -x strata | head -1)
kill $P; for i in $(seq 1 60); do kill -0 $P 2>/dev/null || kill -0 $E 2>/dev/null || break; sleep 1; done
<restart the normal server>
echo "== done $(date +%T)"
