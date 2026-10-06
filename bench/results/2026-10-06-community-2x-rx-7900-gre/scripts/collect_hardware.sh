#!/usr/bin/env bash
# Everything the report's data/hardware.txt needs: GPU, CPU, RAM, PCIe link, OS, driver.
# Needs no root: the PCIe link attributes in sysfs are world-readable on this kernel.
set -u
OUT=${1:?usage: collect_hardware.sh <output file>}

{
  echo "=== collected $(date -Is) by collect_hardware.sh ==="
  echo
  echo "=== uname -a ==="
  uname -a
  echo
  echo "=== /etc/os-release ==="
  cat /etc/os-release
  echo
  echo "=== rocminfo (agents) ==="
  rocminfo 2>/dev/null | grep -E "^\s*(Name|Marketing Name|Device Type|Vendor ID|Device ID|gfx|Supported Target|Workgroup|Wavefront|L[123] Cache|Uuid|Feature|Board Name|Driver Version|gfx_target_version|Minor Max Mode)" | sed 's/^ *//'
  echo
  echo "=== rocm-smi ==="
  rocm-smi --showid --showmeminfo vram vram_barc --showtemp --showpower --showclock --showfw 2>&1 | grep -v "libdrm_amdgpu\|^$"
  echo
  echo "=== amdgpu kernel module ==="
  cat /sys/module/amdgpu/version 2>/dev/null || echo "not exposed"
  echo
  echo "=== drm cards ==="
  for c in /sys/class/drm/card[0-9]; do
    [ -e "$c/device/uevent" ] || continue
    echo "--- $c"
    cat "$c/device/uevent"
    echo "boot  : $(cat "$c/device/boot_vga" 2>/dev/null)"
    vtot=$(cat "$c/device/mem_info_vram_total" 2>/dev/null)
    vuse=$(cat "$c/device/mem_info_vram_used" 2>/dev/null)
    echo "vram  : $vtot bytes total, $vuse used"
    echo "busy  : $(cat "$c/device/gpu_busy_percent" 2>/dev/null) %"
  done
  echo
  echo "=== PCIe link speed and width (sysfs, no root needed) ==="
  for d in /sys/bus/pci/devices/*/; do
    [ -e "$d/vendor" ] || continue
    case "$(cat "$d/class" 2>/dev/null)" in 0x030000|0x060400) ;; *) continue ;; esac
    echo "$(basename "$d") class=$(cat "$d/class") vendor=$(cat "$d/vendor") device=$(cat "$d/device")"
    echo "    current_link_speed=$(cat "$d/current_link_speed" 2>/dev/null)"
    echo "    current_link_width=$(cat "$d/current_link_width" 2>/dev/null)"
    echo "    max_link_speed    =$(cat "$d/max_link_speed" 2>/dev/null)"
    echo "    max_link_width    =$(cat "$d/max_link_width" 2>/dev/null)"
    [ -r "$d/uevent" ] && grep PCI_SLOT "$d/uevent" | sed 's/^/    /'
  done
  echo "(the extended Link Control/Status registers need root: lspci is not installed and sudo needs a password)"
  echo
  echo "=== CPU ==="
  grep -m1 "model name" /proc/cpuinfo
  grep -m1 "^cpu MHz" /proc/cpuinfo
  echo "cores/threads: $(nproc)"
  grep -m1 "^flags" /proc/cpuinfo | tr ' ' '\n' | grep -E "^(avx|avx2|avx512f|amx_tile)$" | tr '\n' ' '; echo
  echo
  echo "=== RAM ==="
  grep -E "MemTotal|MemAvailable|SwapTotal|HugePages_Total|Hugepagesize" /proc/meminfo
  echo
  echo "=== storage ==="
  lsblk -d -o NAME,ROTA,SIZE,MODEL 2>/dev/null
  echo
  echo "=== ROCm / HIP ==="
  sed 's/^/rocm tree version: /' "$HOME/.local/opt/rocm/.info/version" 2>/dev/null || echo "no rocm tree version file"
  hipcc --version 2>&1 | head -4
  rocm-smi --version 2>&1 | head -4
  modinfo amdgpu 2>/dev/null | grep -E "^(filename|version)" || echo "modinfo unavailable"
  echo
  echo "=== strata engine ==="
  ls -l engine/strata
  strings engine/strata 2>/dev/null | grep -m3 -E "^0\.1\.[0-9]+$" || true
} > "$OUT" 2>&1
echo "wrote $OUT"
