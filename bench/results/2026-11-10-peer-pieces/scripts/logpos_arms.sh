#!/bin/bash
# Logpos-Check: je Arm ein Serverstart mit STRATA_LOGPOS, dann lp.py
# Arme: name:build:pieces:p2p
S=$(dirname "$(readlink -f "$0")")
WT=<strata>
ARMS=${ARMS:-"main:<strata-main>/build:0:0 off:$WT/build-rb:0:0 on:$WT/build-rb:1:0 mainP2P:<strata-main>/build:0:1 onP2P:$WT/build-rb:1:1"}
for spec in $ARMS; do
  IFS=: read name bd pieces p2p <<< "$spec"
  rm -f $S/$name-lp.logpos
  ( [ "$pieces" = 1 ] && export STRATA_PF_PEER_PIECES=1 || unset STRATA_PF_PEER_PIECES
    [ "$p2p" = 1 ] && export STRATA_PEER_NO_P2P= || export STRATA_PEER_NO_P2P=1
    export STRATA_LOGPOS=$S/$name-lp.logpos STRATA_LOGPOS_TOPK=20
    STRATA_DIR=$WT BUILD_DIR=$bd VISION=none FORWARD=0 HOST=127.0.0.1 PORT=8081 CONV_CACHE_MIB=0 \
    LOG_FILE=$S/lpsrv-$name.log setsid <run-script> >/dev/null 2>&1 & echo $! > $S/pid )
  ok=0; for i in $(seq 1 600); do curl -sf http://127.0.0.1:8081/v1/models >/dev/null 2>&1 && { ok=1; break; }; sleep 1; done
  [ $ok = 1 ] && <venv>/bin/python -I $S/lp.py | sed "s/^/$name /" || echo "$name start timeout"
  grep -iE 'peer.*(p2p|host route)|PIECES' $S/lpsrv-$name.log.engine | head -3 | sed "s/^/$name log: /"
  pid=$(cat $S/pid); kill -TERM -- -$pid 2>/dev/null
  for i in $(seq 1 60); do pgrep -f -- "--expert-profile" >/dev/null || break; sleep 1; done
  pkill -9 -f -- "--expert-profile" 2>/dev/null; sleep 3
done
echo done
