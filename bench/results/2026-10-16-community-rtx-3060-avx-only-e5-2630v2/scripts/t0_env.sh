#!/usr/bin/env bash
# T0: record the machine, the software and the files before testing (COMMUNITY_BENCHMARKS.md "What to record").
#   ./t0_env.sh          machine facts + quick file check (sizes)            ~1 min
#   ./t0_env.sh --hash   also SHA-256 of every model/pack/engine file        ~15 min more
set -u
. "$(dirname "$0")/common.sh"
D=$(newdir T0-env)
say "T0 -> $D"
{
  echo "## date";            date -Is
  echo "## uname";           uname -a
  echo "## os-release";      cat /etc/os-release
  echo "## lscpu";           lscpu
  echo "## cpu flags (avx*, fma, f16c)"; grep -m1 -o -w -E 'avx[0-9a-z_]*|fma|f16c' /proc/cpuinfo | sort -u | tr '\n' ' '; echo
  echo "## numa nodes";      for n in /sys/devices/system/node/node*; do echo "$(basename $n): cpus $(cat $n/cpulist)"; grep -E 'MemTotal|MemFree' $n/meminfo; done
  echo "## free -b";         free -b
  echo "## loadavg";         cat /proc/loadavg
  echo "## top processes by CPU"; ps -eo pid,pcpu,pmem,rss,comm --sort=-pcpu | head -15
  echo "## nvidia-smi";      nvidia-smi
  echo "## nvidia-smi query"; nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,driver_version,vbios_version,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,power.limit,power.default_limit,power.max_limit,temperature.gpu,clocks.max.graphics,clocks.max.memory --format=csv 2>/dev/null \
                             || nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max,power.limit --format=csv
  echo "## gpu compute apps"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  echo "## PCIe (sysfs) GPU and its root port"
  for dev in $(ls /sys/bus/pci/devices); do
    if [ "$(cat /sys/bus/pci/devices/$dev/vendor)" = 0x10de ] && [ "$(cat /sys/bus/pci/devices/$dev/class | cut -c1-6)" = 0x0300 ]; then
      p=$(readlink -f /sys/bus/pci/devices/$dev)
      for x in "$p" "$(dirname "$p")"; do
        echo "$(basename $x): max $(cat $x/max_link_speed 2>/dev/null) x$(cat $x/max_link_width 2>/dev/null), current $(cat $x/current_link_speed 2>/dev/null) x$(cat $x/current_link_width 2>/dev/null), numa_node $(cat $x/numa_node 2>/dev/null)"
      done
    fi
  done
  # until the 2026-10-16 reseat, the SSDs' PCIe switch card raised ~140,000 correctable errors/s (runbook 1.1.1)
  echo "## PCIe errors this boot (should be 0)"; echo "PCIe Bus Error lines: $(journalctl -k -b --no-pager 2>/dev/null | grep -c 'PCIe Bus Error')"
  for dev in $(ls /sys/bus/pci/devices); do
    c=$(grep -h TOTAL_ERR /sys/bus/pci/devices/$dev/aer_dev_correctable /sys/bus/pci/devices/$dev/aer_dev_nonfatal /sys/bus/pci/devices/$dev/aer_dev_fatal 2>/dev/null | awk '{s+=$2} END {print s+0}')
    [ "$c" != 0 ] && echo "$dev AER total $c: $(lspci -s $dev 2>/dev/null | cut -d' ' -f2-)"
  done
  echo "## NVMe link (SSDs)"
  for n in /sys/class/nvme/nvme*; do x=$(readlink -f $n/device); echo "$(basename $n) $(cat $n/model | xargs) $(basename $x): current $(cat $x/current_link_speed) x$(cat $x/current_link_width)"; done
  echo "## network (should be cut)"; ip -br addr; ip route
  echo "## disk";            df -h /mnt/ssd
  echo "## compilers";       gcc-14 --version | head -1; g++-14 --version | head -1
  echo "## nvcc";            $TOP/toolchain/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc --version | tail -2
  echo "## python";          "$PYS" --version; "$PYE" --version
  echo "## strata source";   git -C $TOP/sources/Strata log -1 --format='%H %ci' 2>/dev/null
  echo "## engine BUILD.json"; cat $S/engine/BUILD.json; echo
  echo "## engine sha256";   sha256sum $S/engine/strata $S/engine/libstrata_kq.so
  echo "## test-suite build"; ls -l $T/build-tests/strata 2>/dev/null
  echo "## base config ($BASE_CFG)"; cat "$BASE_CFG"
} > "$D/environment.txt" 2>&1
"$PYS" -m pip freeze > "$D/pip-freeze-setup-venv.txt" 2>/dev/null
"$PYE" -m pip freeze > "$D/pip-freeze-eval-venv.txt" 2>/dev/null

# Quick check: every reference file exists with the size it had when the reference hashes were made
python3 - "$T/manifest/sha256-reference.txt" "$TOP" > "$D/files-check.txt" <<'EOF'
import os, sys
ref, top = sys.argv[1], sys.argv[2]
bad = 0
for line in open(ref):
    h, name = line.rstrip("\n").split("  ", 1)
    p = os.path.join(top, name)
    ok = os.path.exists(p)
    print(("ok      " if ok else "MISSING ") + name + ("  %d bytes" % os.path.getsize(p) if ok else ""))
    bad += not ok
print("missing files:", bad)
EOF
tail -1 "$D/files-check.txt"

if [ "${1:-}" = --hash ]; then
  say "hashing (about 15 minutes) ..."
  ( cd $TOP && sha256sum -c "$T/manifest/sha256-reference.txt" ) > "$D/sha256-check.txt" 2>&1
  echo "sha256: $(grep -c ': OK$' "$D/sha256-check.txt") OK, $(grep -vc ': OK$' "$D/sha256-check.txt") not OK"
fi
say "T0 done: $D/environment.txt"
