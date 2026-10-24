#!/bin/bash
# usage: ab_run.sh LABEL MAXNEW ROUNDS CONFIG ARM [ARM ...]   (ab_engine.py with the GPU pre-flight and a state snapshot)
label=$1; maxnew=$2; rounds=$3; cfg=$4; shift 4
R=<STRATA_ROOT>; O=<BENCH>/$label; mkdir -p "$O"
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
running=$(curl -s -m 5 http://127.0.0.1:8090/running)
if [ "$used" -gt 2000 ] || [ "$running" != '{"running":[]}' ]; then echo "ABORT before $label: gpu used ${used} MiB, llama-swap says $running"; exit 2; fi
{ echo "label: $label  max-new: $maxnew  rounds: $rounds"; echo "start: $(date -Is)"; echo "config: $cfg"; echo "arms: $*"
  echo "git HEAD: $(git -C $R rev-parse HEAD)  engine: $(tr -d '\n ' < $R/engine/BUILD.json)"
  nvidia-smi --query-gpu=memory.used,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv
  echo "loadavg: $(cat /proc/loadavg)"; free -m | sed -n 1,2p; } > "$O/pre-run-state.txt" 2>&1
cd $R && .venv/bin/python tools/ab_engine.py "$cfg" "$@" --rounds "$rounds" --port 18234 --max-new "$maxnew" --log "$O/ab_engine.jsonl" > "$O/ab-console.txt" 2>&1
echo "exit=$?" >> "$O/ab-console.txt"; echo "end: $(date -Is)" >> "$O/pre-run-state.txt"; tail -12 "$O/ab-console.txt"
