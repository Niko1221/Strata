#!/bin/bash
# every 10 s: time, engine RSS (MiB, process named strata), GPU memory/clock/power/temp/util   -> one CSV line
out=<BENCH>/sampler.csv
echo "time,engine_rss_mib,gpu_mem_mib,sm_mhz,power_w,temp_c,util_pct,mem_avail_mib" > "$out"
while true; do
  rss=$(ps -C strata -o rss= 2>/dev/null | awk '{s+=$1} END {printf "%d", s/1024}')
  g=$(nvidia-smi --query-gpu=memory.used,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  av=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
  echo "$(date +%T),${rss:-0},${g:-,,,,},$av" >> "$out"
  sleep 10
done
