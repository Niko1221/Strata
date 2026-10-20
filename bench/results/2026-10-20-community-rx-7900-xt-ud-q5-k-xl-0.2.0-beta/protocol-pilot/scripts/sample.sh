#!/bin/sh
export PATH=/run/current-system/sw/bin:$PATH
# usage: sample.sh OUT.csv  -- one row per 2 s: GPU VRAM used, sclk/mclk, power, temp, PCIe link, RAM used, load
out="$1"
d=/sys/class/drm/card1/device
h=$(ls -d $d/hwmon/hwmon* | head -1)
echo "t_utc,vram_used_b,sclk_mhz,mclk_mhz,power_w,temp_c,pcie_speed,pcie_width,mem_used_kib,mem_avail_kib,load1" > "$out"
while true; do
  sclk=$(grep '\*' $d/pp_dpm_sclk | head -1 | sed 's/.*: *\([0-9]*\)Mhz.*/\1/')
  mclk=$(grep '\*' $d/pp_dpm_mclk | head -1 | sed 's/.*: *\([0-9]*\)Mhz.*/\1/')
  pw=$(cat $h/power1_average 2>/dev/null | awk '{printf "%.1f", $1/1000000}')
  tp=$(cat $h/temp1_input 2>/dev/null | awk '{printf "%.0f", $1/1000}')
  tot=$(awk '/MemTotal/{print $2}' /proc/meminfo); av=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  echo "$(date -u +%FT%TZ),$(cat $d/mem_info_vram_used),$sclk,$mclk,$pw,$tp,$(cat $d/current_link_speed | tr ' ' '_'),$(cat $d/current_link_width),$((tot-av)),$av,$(cut -d' ' -f1 /proc/loadavg)" >> "$out"
  sleep 2
done
