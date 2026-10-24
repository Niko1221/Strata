#!/bin/bash
# usage: run_one.sh LABEL CONFIG.json [extra strata_bench args]   (run from anywhere; never touches the repo's files)
set -u
label=$1; cfg=$2; shift 2
R=<STRATA_ROOT>
O=<BENCH>/$label
mkdir -p "$O"
{
  echo "label: $label"; echo "start: $(date -Is)"; echo "config: $cfg"
  echo "git HEAD: $(git -C $R rev-parse HEAD)  engine: $(cat $R/engine/BUILD.json | tr -d '\n ')"
  echo "--- gpu"; nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
  echo "--- llama-swap /running: $(curl -s -m 5 http://127.0.0.1:8090/running)"
  echo "--- memory"; free -m | sed -n 1,2p
  echo "--- loadavg: $(cat /proc/loadavg)   governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
  echo "--- top cpu"; ps -eo pid,pcpu,rss,comm --sort=-pcpu | head -6
  echo "--- config args"; python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['exe']); print(' '.join(map(str,d['args']))); print({k:v for k,v in d.items() if k not in ('args','exe','sampling')})" "$cfg"
} > "$O/pre-run-state.txt" 2>&1
cd $R && .venv/bin/python tools/strata_bench.py --config "$cfg" --out "$O" "$@" > "$O/bench-console.txt" 2>&1
echo "exit=$?" >> "$O/bench-console.txt"
{ echo "end: $(date -Is)"; nvidia-smi --query-gpu=memory.used,clocks.sm,power.draw,temperature.gpu --format=csv; echo "llama-swap /running: $(curl -s -m 5 http://127.0.0.1:8090/running)"; } > "$O/post-run-state.txt" 2>&1
tail -3 "$O/bench-console.txt"
