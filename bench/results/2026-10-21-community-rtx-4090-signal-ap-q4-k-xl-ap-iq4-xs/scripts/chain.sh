#!/bin/bash
# usage: chain.sh LABEL:CONFIG [LABEL:CONFIG ...]   runs run_one.sh for each, aborting if the GPU is not idle
S=<SCRIPTS>
for pair in "$@"; do
  label=${pair%%:*}; cfg=${pair#*:}
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  running=$(curl -s -m 5 http://127.0.0.1:8090/running)
  if [ "$used" -gt 2000 ] || [ "$running" != '{"running":[]}' ]; then echo "ABORT before $label: gpu used ${used} MiB, llama-swap says $running"; exit 2; fi
  echo "== $label $(date -Is)"; $S/run_one.sh "$label" "$cfg"
done
echo CHAIN-DONE
