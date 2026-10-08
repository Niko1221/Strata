#!/bin/bash
# uso: run-ver-test.sh <versao>  (16 GB simulado, ctx 262144, prompt ~250K)
V=$1; D=<strata-dir>-v$V; O=$D/ver-test.out; T=$D/ver-temps.csv
export XDG_RUNTIME_DIR=/run/user/UID DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/UID/bus
cd $D; : > $O; : > $T
mkdir -p engine; cp build-906/strata build-906/strata-device engine/ 2>/dev/null
echo "{\"version\":\"$V\"}" > engine/BUILD.json
python3 - <<PY
import json
c=json.load(open("<strata-dir>/strata-iq2_xs-16gb-ctx262144.json"))
s=json.dumps(c).replace("/IA/Strata-gfx906/","/IA/Strata-gfx906-v$V/")
c=json.loads(s); c["log"]="$D/engine-16gb-ctx262144.log"
json.dump(c,open("$D/cfg-16gb-ctx262144.json","w"),indent=1)
PY
cp <strata-dir>/data/expert-profile*.bin data/ 2>/dev/null
cool() { while [ "$(/opt/rocm/bin/rocm-smi --showtemp | grep junction | grep -o "[0-9.]*$" | cut -d. -f1)" -gt 50 ]; do sleep 15; done; }
cool
systemd-run --user --scope -q --unit=strata-ver-test -p MemoryHigh=52G -p MemoryMax=58G -p MemorySwapMax=0 choom -n 1000 -- \
  <strata-dir>/.venv/bin/python $D/serve/server.py --engine strata --config $D/cfg-16gb-ctx262144.json --host 127.0.0.1 --port 8080 > $D/server.log 2>&1 &
for i in $(seq 1 400); do curl -sf localhost:8080/health >/dev/null && break; sleep 2; done
echo "versao $V pronto apos ~$((i*2))s; health=$(curl -s localhost:8080/health)" >> $O
touch run.flag
( hot=0; while [ -f run.flag ]; do t=$(/opt/rocm/bin/rocm-smi --showtemp | grep junction | grep -o '[0-9.]*$' | cut -d. -f1); echo "$(date +%T) $t" >> $T
  if [ -n "$t" ] && [ "$t" -ge 107 ]; then hot=$((hot+1)); else hot=0; fi
  if [ $hot -ge 2 ]; then echo "ABORTADO junction $t" >> $O; rm -f run.flag; systemctl --user stop strata-ver-test.scope; fi; sleep 10; done ) &
for seed in 23 11; do
  cool; echo "=== seed $seed ===" >> $O
  python3 <strata-dir>/bench-ctx.py 250000 $seed >> $O 2>&1
  grep -a "strata serve: prompt\|decode expert cache hit rate\|rror\|nan\|NaN" $D/engine-16gb-ctx262144.log | tail -4 | cut -c1-230 >> $O
done
echo TEST_DONE >> $O
rm -f run.flag
systemctl --user stop strata-ver-test.scope; sleep 15
