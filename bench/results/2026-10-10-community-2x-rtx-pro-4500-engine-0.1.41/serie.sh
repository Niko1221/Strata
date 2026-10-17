#!/bin/bash
# Strata 0.1.41 auf 2x RTX PRO 4500 (Linux): Block X (#1760 Kartenreihenfolge + #1353 Prefill), Block Y (8 Clients, batch-groups),
# Block Z (Community-Benchmark 0.1.41 gegen 0.1.40.1, ABA). Dienst stoppt, trap stellt Produktion (0.1.40.1 solo) wieder her.
MESS=/home/qni/messungen/2026-10-10-strata-0.1.41-tests
cd $MESS
PY=/home/qni/ai/Strata-0.1.36/.venv/bin/python
V0=/home/qni/ai/Strata-0.1.40.1
V1=/home/qni/ai/Strata-0.1.41
exec >> $MESS/lauf-alles.log 2>&1
stoppen() {
  pkill -TERM -f "[s]erve/server.py --engine strata"; sleep 3
  pkill -TERM -f "[e]ngine/strata --serve"; pkill -TERM -f "[e]ngine/strata-vision"; pkill -TERM -f "[e]ngine/strata "
  for i in $(seq 1 60); do
    if ! ss -ltn | grep -q ":8080 " && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 2000 ]; then return 0; fi
    sleep 2
  done
  pkill -KILL -f "[e]ngine/strata --serve"; pkill -KILL -f "[e]ngine/strata "; pkill -KILL -f "[s]erve/server.py --engine strata"; sleep 5
}
laeuft() { curl -s -m 5 http://127.0.0.1:8080/v1/models 2>/dev/null | grep -q loaded; }
restore() {
  echo "== RESTORE $(date +%T)"
  kill $TAKTPID 2>/dev/null
  stoppen
  $V0/umschalten.sh solo
  sleep 5
  if ! laeuft; then
    echo "== RESTORE 1 FEHLGESCHLAGEN $(date +%T) - zweiter Versuch von Hand"
    stoppen
    (cd $V0; setsid nohup sh run-swift-iq3_xxs.sh > $MESS/restore-run.out 2>&1 < /dev/null &)
    for i in $(seq 1 120); do laeuft && break; sleep 3; done
  fi
  laeuft && echo "== RESTORE OK $(date +%T)" || echo "== RESTORE FEHLGESCHLAGEN (Dienst steht!) $(date +%T)"
}
trap restore EXIT
mkcfg() {  # name dir split order(-|as_given) vision(0|1) batch(-|N) groups(-|auto|1|2) reserve(-|MiB)
  $PY - "$@" <<'PYEOF'
import json, sys
name, dirp, split, order, vis, batch, groups, reserve = sys.argv[1:9]
MESS = "/home/qni/messungen/2026-10-10-strata-0.1.41-tests"
c = json.load(open("/home/qni/ai/Strata-0.1.40.1/strata-swift-iq3_xxs.json"))
c = json.loads(json.dumps(c).replace("/home/qni/ai/Strata-0.1.40.1", dirp))
a = c["args"]
c["layer_split"] = split
if order != "-":
    c["gpu_order"] = order
if vis == "0":
    a[:] = [x for x in a if x != "--vision"]
    c.pop("vision", None)
if reserve != "-":
    i = a.index("--vram-reserve-mib"); a[i + 1] = reserve
if batch != "-":
    a += ["--batch", batch, "--trim-stage-weights"]
    if groups != "-":
        a += ["--batch-groups", groups]
c["sampling"]["presence_penalty"] = 0.0
c["log"] = MESS + "/server-%s.log" % name
json.dump(c, open(MESS + "/config-%s.json" % name, "w"), indent=1)
PYEOF
}
start() {  # name dir split order vision batch groups reserve -> 0 ok
  echo "== ARM $1 (dir $(basename $2) split $3 order $4 vision $5 batch $6 groups $7 reserve $8) $(date +%T)"
  mkcfg "$@"
  : > $MESS/server-$1.log
  setsid nohup $PY $2/serve/server.py --engine strata --config $MESS/config-$1.json --port 8080 > $MESS/serverout-$1.txt 2>&1 < /dev/null &
  for i in $(seq 1 400); do laeuft && break; sleep 3; done
  if ! laeuft; then
    echo "START FEHLER $1"; tail -8 $MESS/server-$1.log | cut -c1-240; tail -5 $MESS/serverout-$1.txt | cut -c1-240; stoppen; return 1
  fi
  grep -h -i "card order\|gpu_order\|as_given\|speed score\|layer split:\|batch\|groups\|resident" $MESS/serverout-$1.txt $MESS/server-$1.log | cut -c1-220 | sort -u | head -16
  timeout 900 python3 $MESS/parlast.py $MESS/warm-$1.json 1 1 8000 300 > /dev/null 2>&1
  free -m | sed -n 2p
  return 0
}
takt() { nvidia-smi --query-gpu=clocks.mem,power.draw --format=csv,noheader -l 1 > $MESS/takt-$1.csv 2>&1 & TAKTPID=$!; }
decode_lang() {  # name: ein langer Decode, GPU-Auslastung, tok/s aus dem Log
  curl -s -m 300 http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"swift-1.5-iq3_xxs","messages":[{"role":"user","content":"Write a very long and detailed Python package for text statistics with many modules, docstrings and tests. Give all code."}],"max_tokens":2500,"temperature":0}' > /dev/null &
  CURLPID=$!
  sleep 8
  echo "   Decode lang, GPU: $(nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader | tr '\n' ';')"
  wait $CURLPID
  grep "generated in" $MESS/server-$1.log | tail -n 1 | cut -c1-200
}
armX() {  # name dir split order
  start $1 $2 $3 $4 1 - - - || return 1
  takt $1
  decode_lang $1
  BENCH_LENGTHS=1k,32k,128k python3 $MESS/bench.py $MESS/bench-$1 $MESS/server-$1.log > $MESS/bench-$1.log 2>&1; grep -E "^\| (1k|32k|128k) " $MESS/bench-$1.log | cut -c1-120
  kill $TAKTPID 2>/dev/null
  stoppen
}
armY() {  # name dir groups
  for R in 700 1500 2600; do
    if start $1 $2 25 - 0 8 $3 $R; then
      echo "   (Reserve $R MiB)"
      takt $1
      timeout 3000 python3 $V1/tools/serve_load.py http://127.0.0.1:8080 --clients 1,2,4,8 --rounds 3 --max-tokens 256 --json $MESS/load-$1.json > $MESS/load-$1.log 2>&1
      tail -n 14 $MESS/load-$1.log | cut -c1-170
      kill $TAKTPID 2>/dev/null
      stoppen
      return 0
    fi
  done
  echo "Y-ARM $1: keine Reserve (700/1500/2600) hat gestartet"
  return 1
}
armZ() {  # name dir mit_nadel_und_tools(0|1)
  start $1 $2 25 - 1 - - - || return 1
  takt $1
  BENCH_LENGTHS=4k,32k,128k,256k python3 $MESS/bench.py $MESS/bench-$1 $MESS/server-$1.log > $MESS/bench-$1.log 2>&1; grep -E "^\| (4k|32k|128k|256k) " $MESS/bench-$1.log | cut -c1-120
  grep "generated in" $MESS/server-$1.log | tail -n 12 | cut -c1-230 > $MESS/timing-$1.txt
  if [ "$3" = 1 ]; then
    echo "-- Tool-Aufrufe:"
    python3 $MESS/quality.py $MESS/q-$1 tools > $MESS/q-$1.log 2>&1; tail -n 9 $MESS/q-$1.log | cut -c1-160
    echo "-- Nadeltest:"
    (cd $2 && timeout 2400 $PY tools/needle_bench.py --lengths 32k,128k,262k --depths 10,50,90 --url http://127.0.0.1:8080 --out $MESS/needle-$1.json) 2>&1 | tail -n 12 | cut -c1-200
  fi
  kill $TAKTPID 2>/dev/null
  stoppen
}
echo "== START $(date)"
stoppen
echo "######## BLOCK X: #1760 + #1353"
armX X1-041-auto      $V1 auto -
armX X2-041-auto-asgiven $V1 auto as_given
echo "######## BLOCK Y: 8 Clients"
armY Y1-041-b8-auto   $V1 auto
armY Y2-041-b8-g1     $V1 1
armY Y3-0401-b8-g2    $V0 2
echo "######## BLOCK Z: Community-Benchmark"
armZ Z1-041           $V1 1
armZ Z0-0401          $V0 0
armZ Z2-041-wdh       $V1 0
echo "== FERTIG serie $(date +%T)"
