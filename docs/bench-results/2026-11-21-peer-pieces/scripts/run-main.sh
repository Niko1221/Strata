#!/bin/bash
# A/B pieces off/on auf dem rebased Branch (3142b66b); eigener Serverstart pro Arm, nur eigene Prozessgruppe beenden
S=$(dirname "$(readlink -f "$0")")
WT=<strata>
MAIN=<strata-main>
PY=$MAIN/.venv/bin/python
EXE=<strata-main>/build/strata
ROUNDS=${ROUNDS:-4}
G=<models>/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
P=<data>/packs/gsq-iq3_s
export STRATA_PEER_NO_P2P=1 STRATA_GR_DOWN_MAX4=1 STRATA_PEER_HOT_AT=11283 STRATA_PEER_HOT=0.45 STRATA_PREFILL_RING=96 \
       STRATA_NO_LARGEPAGES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1
start() {  # $1 name, $2 pieces 0/1, $3 logpos file or ""
  if ss -ltn | grep -q ':8081 '; then echo "port 8081 busy"; return 1; fi
  local cfg=$S/cfg-$1.json
  $PY - "$cfg" "$S/srv-$1.log" <<PY
import json, sys
json.dump({"exe": "$EXE", "args": ["--pack", "$P", "--native", "$G", "--expert-profile", "$MAIN/data/expert-profile.bin",
  "--expert-cache", "auto", "--prefill", "auto:16384", "--spec", "2", "--spec-min-p", "0.5", "--mtp", "<data>/mtp/rt",
  "--max-context", "262144", "--kv", "int8", "--ple-io", "ram", "--peer-device", "1", "--conversation-cache-mib", "0"],
  "cwd": "$WT", "tokenizer": "$P/tokenizer", "model_name": "qwen3.8-flash-next", "log": sys.argv[2] + ".engine",
  "lib_dirs": ["/usr/local/cuda-12.8/bin", "/usr/local/cuda-12.8/lib64"], "port": 8081, "host": "127.0.0.1",
  "gpu": [0, 1], "gpus_asked": True}, open(sys.argv[1], "w"), indent=1)
PY
  ( [ "$2" = 1 ] && export STRATA_PF_PEER_PIECES=1 || unset STRATA_PF_PEER_PIECES
    if [ -n "$3" ]; then export STRATA_LOGPOS=$3 STRATA_LOGPOS_TOPK=20; rm -f $3; fi
    setsid $PY -I $WT/terminal/serve/server.py --engine strata --config $cfg --port 8081 > $S/srv-$1.log 2>&1 & echo $! > $S/pid )
  for i in $(seq 1 900); do
    curl -sf http://127.0.0.1:8081/v1/models >/dev/null 2>&1 && return 0
    kill -0 $(cat $S/pid) 2>/dev/null || { echo "server $1 died"; return 1; }
    sleep 1
  done
  echo "start $1 timeout"; return 1
}
stop() {
  local pid=$(cat $S/pid); kill -TERM -- -$pid 2>/dev/null
  for i in $(seq 1 90); do pgrep -f "^$EXE" >/dev/null || break; sleep 1; done
  pkill -9 -f "^$EXE" 2>/dev/null; kill -9 -- -$pid 2>/dev/null; sleep 3
}
start out-main 0 $S/main.logpos || { stop; exit 1; }
$PY -I $S/lp.py | sed "s/^/main lp /"
$PY -I $S/gen.py | sed "s/^/main /" | tee -a $S/greedy.txt
stop
$PY -I $S/logpos_identical.py $S/main.logpos $S/../ab2/off.logpos $S/../ab2/on.logpos
echo done
