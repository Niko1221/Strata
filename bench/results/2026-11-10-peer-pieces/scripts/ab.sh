#!/bin/bash
# A/B: STRATA_PF_PEER_PIECES off/on, je Arm ein frischer Serverstart, Reihenfolge wechselt pro Runde
S=$(dirname "$(readlink -f "$0")")
WT=<strata>
OUT=$S/runs.jsonl
ROUNDS=${ROUNDS:-4}
start_arm() {  # $1 = arm
  local arm=$1 log=$S/srv-$1-$2.log
  ( [ "$arm" = on ] && export STRATA_PF_PEER_PIECES=1 || unset STRATA_PF_PEER_PIECES
    STRATA_DIR=$WT BUILD_DIR=$WT/build-rb VISION=none FORWARD=0 HOST=127.0.0.1 PORT=8081 CONV_CACHE_MIB=0 \
    LOG_FILE=$log setsid <run-script> >/dev/null 2>&1 & echo $! > $S/pid )
  for i in $(seq 1 600); do
    curl -sf http://127.0.0.1:8081/v1/models >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "start $arm timeout"; return 1
}
stop_arm() {
  local pid=$(cat $S/pid); kill -TERM -- -$pid 2>/dev/null
  for i in $(seq 1 60); do pgrep -f "build-rb/strata" >/dev/null || break; sleep 1; done
  pkill -9 -f "build-rb/strata" 2>/dev/null; sleep 3
}
for r in $(seq 1 $ROUNDS); do
  if [ $((r % 2)) = 1 ]; then arms="off on"; else arms="on off"; fi
  for a in $arms; do
    start_arm $a $r || { stop_arm; exit 1; }
    <venv>/bin/python -I $S/abpr.py pieces_$a $r $OUT
    cp $S/srv-$a-$r.log.engine $S/engine-$a-$r.txt 2>/dev/null
    stop_arm
  done
done
echo done
