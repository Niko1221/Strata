#!/bin/bash
# One benchmark session as in this report: start the server (loopback, no key, no browser), monitor + benchmark,
# needle checks, stop.  Also records a system snapshot, the enforced GPU power limit and the session's engine log.
# usage (from anywhere): run-session.sh CONFIG OUTDIR TARGETS NEEDLE_LENGTHS
#   e.g. run-session.sh strata-iq3_s-128k.json ~/bench-out 4096,32768,128000 32k,128k
set -u
cd "$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)" || exit 1
CFG=$(readlink -f "$1"); OUT=$2; TARGETS=$3; NEEDLES=$4
B=bench/results/2026-09-30-community-rtx-5090
LOG=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['log'])" "$CFG")
GPU=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | cut -c5- | tr 'A-F' 'a-f')
mkdir -p "$OUT"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/session.log"; }
gpu_temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits; }

{
  echo "date: $(date -Is)"; uname -a; echo "uptime: $(uptime)"
  echo "cmdline: $(cat /proc/cmdline)"
  echo "GPU IOMMU group type: $(cat /sys/bus/pci/devices/$GPU/iommu_group/type 2>/dev/null || echo none)"
  echo "commit: $(git rev-parse HEAD) ($(git describe --tags))"
  cat engine/BUILD.json; echo
  echo "engine sha256: $(sha256sum engine/strata | cut -d' ' -f1)"
  echo "expert-profile sha256: $(sha256sum data/expert-profile.bin | cut -d' ' -f1)"
  echo "platform_profile: $(cat /sys/firmware/acpi/platform_profile 2>/dev/null)  ppd: $(powerprofilesctl get 2>/dev/null)"
  echo "nvidia-powerd: $(pgrep -a nvidia-powerd)"
  for d in /sys/class/firmware-attributes/*/attributes/*/; do echo "firmware $(basename "$d")=$(cat "$d/current_value" 2>/dev/null)"; done
  echo "cpu governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)  epp: $(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null)"
  echo "THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
  free -b; swapon --show
  nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.max,pcie.link.width.max,memory.total,memory.used --format=csv
  nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
  nvidia-smi -q -d POWER | head -12
  nvcc --version 2>/dev/null | tail -2 || /opt/cuda/bin/nvcc --version | tail -2; gcc --version | head -1; .venv/bin/python --version
  findmnt -no FSTYPE,OPTIONS -T ../Strata-data
  pgrep -a 'kwin|plasmashell|Xwayland|firefox' || echo "no desktop or browser processes"
} > "$OUT/sysinfo.txt" 2>&1
cp "$CFG" "$OUT/"
LOGSTART=$(( $(wc -l < "$LOG") + 1 ))

for _ in $(seq 1 60); do [ "$(gpu_temp)" -le 45 ] && break; sleep 5; done   # start cool, at most 5 min wait
log "GPU temperature at start: $(gpu_temp) C"
log "starting server: $CFG"
.venv/bin/python serve/server.py --engine strata --config "$CFG" --port 8080 > "$OUT/server.out" 2>&1 &
SRV=$!
n=0
until curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/v1/models 2>/dev/null | grep -q 200; do
  kill -0 $SRV 2>/dev/null || { log "server exited before it was ready"; exit 1; }
  n=$((n + 1)); [ $n -gt 120 ] && { log "server not ready after 10 min"; kill $SRV; exit 1; }
  sleep 5
done
log "server ready"

nvidia-smi --query-gpu=timestamp,enforced.power.limit,power.draw,utilization.gpu,clocks.sm,temperature.gpu,pstate,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown \
  --format=csv,nounits -l 2 > "$OUT/power.csv" 2>&1 &
PW=$!
.venv/bin/python $B/monitor.py "$OUT/telemetry.jsonl" &
M=$!
log "benchmark --targets $TARGETS"
.venv/bin/python $B/benchmark.py --root . --pack ../Strata-data/packs/iq3_s --url http://127.0.0.1:8080 \
  --out "$OUT" --targets "$TARGETS" > "$OUT/benchmark.out" 2>&1
rc=$?
kill $M
log "benchmark exit $rc"
log "needles --lengths $NEEDLES"
.venv/bin/python tools/needle_bench.py --url http://127.0.0.1:8080 --lengths "$NEEDLES" --depths 10,50,90 \
  --out "$OUT/needles.json" > "$OUT/needles.out" 2>&1
nrc=$?
log "needle exit $nrc"
kill $PW
kill $SRV; wait $SRV 2>/dev/null
sed -n "${LOGSTART},\$p" "$LOG" > "$OUT/engine.log"
log "session done (benchmark $rc, needles $nrc)"
[ $rc -eq 0 ] && [ $nrc -eq 0 ]
