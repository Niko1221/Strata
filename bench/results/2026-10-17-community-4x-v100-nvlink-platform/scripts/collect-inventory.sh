#!/usr/bin/env bash
# collect-inventory.sh — Этап I: снятие аппаратной и программной инвентаризации.
# Только чтение: ничего не меняет в системе. Результаты — в raw/inventory и raw/topology.
# Использование: ./scripts/collect-inventory.sh [каталог_результатов]
set -u

BASE="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
STAMP="$(date -u +%Y-%m-%dT%H%M%SZ)"
OUT="$BASE/raw/inventory/$STAMP"
TOPO="$BASE/raw/topology/$STAMP"
mkdir -p "$OUT" "$TOPO"

run() { # run <file> <cmd...>
  local f="$OUT/$1"; shift
  { echo "# $* (UTC $(date -u +%FT%TZ))"; "$@" 2>&1; echo "# exit=$?"; } > "$f"
}
runtopo() { # runtopo <file> <cmd...>
  local f="$TOPO/$1"; shift
  { echo "# $* (UTC $(date -u +%FT%TZ))"; "$@" 2>&1; echo "# exit=$?"; } > "$f"
}

echo "OUT=$OUT"
echo "TOPO=$TOPO"

# ---------- Паспорт самого замера ----------
{
  echo "capture_utc:  $(date -u +%FT%TZ)"
  echo "host:         $(hostname)"
  echo "script:       $0"
  echo "script_sha:   $(sha256sum "$0" | cut -d' ' -f1)"
  echo "user:         $(id -un)"
  echo "purpose:      Этап I — инвентаризация (только чтение, ничего не изменяется)"
  echo "disk:"; df -h "$HOME" | tail -2
  echo "optional tools:"; for t in dcgmi nsys ncu compute-sanitizer git; do printf '  %-16s %s\n' "$t" "$(command -v "$t" || echo NOT-FOUND)"; done
} > "$OUT/capture-info.txt"

# ---------- ОС / ядро ----------
run os-release.txt cat /etc/os-release
run uname-a.txt uname -a
run lsb-release.txt lsb_release -a
run kernel-cmdline.txt cat /proc/cmdline
run uptime.txt uptime
run hostname.txt hostnamectl

# ---------- CPU / RAM / NUMA ----------
run lscpu.txt lscpu -a -e
run lscpu-full.txt lscpu
run free-h.txt free -h
run meminfo-key.txt grep -E '^(MemTotal|MemAvailable|SwapTotal|HugePages_Total|Hugepagesize)' /proc/meminfo
if command -v numactl >/dev/null 2>&1; then run numactl-hw.txt numactl --hardware; fi
run nodes.txt ls -1 /sys/devices/system/node
for n in /sys/devices/system/node/node*/cpulist; do echo "$n: $(cat "$n")"; done > "$OUT/nodes-cpulist.txt" 2>&1

# ---------- GPU: базовые ----------
run nvidia-smi.txt nvidia-smi
run nvidia-smi-L.txt nvidia-smi -L
run nvidia-smi-quiet-query.txt nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,compute_cap,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.max.sm,ecc.mode.current,ecc.errors.uncorrected.aggregate.total,ecc.errors.corrected.aggregate.total,retired_pages.pending,vbios_version,inforom.img,inforom.oem,compute_mode --format=csv
run nvidia-smi-pcie-query.txt nvidia-smi --query-gpu=index,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv
run nvidia-smi-throttle.txt nvidia-smi --query-gpu=index,clocks_throttle_reasons.active,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.gpu_idle --format=csv
run nvidia-smi-compute-apps.txt nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
run nvidia-smi-full-q.txt nvidia-smi -q
run nvidia-smi-q-ecc.txt nvidia-smi -q -d ECC
run nvidia-smi-q-clocks.txt nvidia-smi -q -d CLOCK,PERFORMANCE
run nvidia-smi-q-remap.txt nvidia-smi -q -d PAGE_RETIREMENT,ROW_REMAPPER

# ---------- Топология NVLink / PCIe ----------
runtopo nvidia-smi-topo-m.txt nvidia-smi topo -m
runtopo nvidia-smi-topo-m-ascii.txt nvidia-smi topo -m
runtopo nvlink-status.txt nvidia-smi nvlink --status
runtopo nvlink-capabilities.txt nvidia-smi nvlink -c
runtopo nvlink-remote-linkinfo.txt nvidia-smi nvlink -R
runtopo nvlink-info.txt nvidia-smi nvlink -info
runtopo nvlink-errorcounters.txt nvidia-smi nvlink -e
runtopo nvlink-crc-errorcounters.txt nvidia-smi nvlink -ec
runtopo nvlink-throughput-data.txt nvidia-smi nvlink -gt d
runtopo nvlink-throughput-raw.txt nvidia-smi nvlink -gt r
runtopo nvlink-linkwidth.txt nvidia-smi nvlink -gLWidth
runtopo nvlink-bwmode.txt nvidia-smi nvlink -gBwMode
runtopo nvidia-smi-q-gpulink.txt nvidia-smi -q

# ---------- PCIe ----------
if command -v lspci >/dev/null 2>&1; then
  run lspci-nvidia.txt lspci -nnvv
  lspci -nnvv 2>/dev/null | awk '/NVIDIA/{p=1} p{print} /^$/{if(p)p=0}' > "$OUT/lspci-nvidia-only.txt"
fi
for d in /sys/bus/pci/devices/0000:0[3456]:00.0; do
  [ -d "$d" ] || continue
  {
    echo "== $d"
    echo "current_link_speed: $(cat "$d/current_link_speed" 2>/dev/null)"
    echo "current_link_width: $(cat "$d/current_link_width" 2>/dev/null)"
    echo "max_link_speed:     $(cat "$d/max_link_speed" 2>/dev/null)"
    echo "max_link_width:     $(cat "$d/max_link_width" 2>/dev/null)"
    echo "numa_node:          $(cat "$d/numa_node" 2>/dev/null)"
    echo "local_cpulist:      $(cat "$d/local_cpulist" 2>/dev/null)"
    echo "driver:             $(basename "$(readlink "$d/driver" 2>/dev/null)" 2>/dev/null)"
  } >> "$OUT/pci-sysfs-gpus.txt"
done

# ---------- Инструменты сборки ----------
for t in gcc g++ make cmake ninja nvcc nvcc-12 nvcc-13 python3 pip3 git rg; do
  if command -v "$t" >/dev/null 2>&1; then
    { echo "== $t -> $(command -v $t)"; "$t" --version 2>&1 | head -5; } >> "$OUT/toolchain.txt"
  else
    echo "== $t: NOT FOUND" >> "$OUT/toolchain.txt"
  fi
done
{
  echo "== CUDA toolkit candidates"
  for c in /usr/local/cuda /usr/local/cuda-* /opt/cuda*; do
    [ -e "$c" ] && ls -ld "$c" && [ -x "$c/bin/nvcc" ] && "$c/bin/nvcc" --version
  done
} >> "$OUT/toolchain.txt" 2>&1

# ---------- Python / CUDA-зависимости ----------
VENV="${VENV:-$HOME/strata185/Strata/.venv}"
{ python3 -c 'import sys;print(sys.version);print(sys.executable)' 2>&1; } > "$OUT/python.txt"
{ python3 -m pip --version 2>&1; } >> "$OUT/python.txt"
if [ -x "$VENV/bin/python" ]; then
  {
    echo "== venv: $VENV"
    "$VENV/bin/python" -V 2>&1
    "$VENV/bin/python" -m pip list 2>&1 | head -60
    echo "== site-packages listing"
    ls -1 "$VENV"/lib/python*/site-packages 2>/dev/null
  } >> "$OUT/python.txt"
fi
for PYBIN in python3 "$VENV/bin/python"; do
  [ "$PYBIN" = "python3" ] && tag=system || tag=venv
  for m in torch vllm triton transformers numpy; do
    "$PYBIN" - "$m" "$tag" <<'PY' >> "$OUT/python-packages.txt" 2>&1
import importlib, sys
m, tag = sys.argv[1], sys.argv[2]
try:
    mod = importlib.import_module(m)
    print(f"[{tag}] {m}: version={getattr(mod,'__version__','?')} path={getattr(mod,'__file__','?')}")
    if m == "torch":
        print("   cuda_version:", mod.version.cuda)
        print("   cudnn:", mod.backends.cudnn.version())
        print("   available:", mod.cuda.is_available(), "device_count:", mod.cuda.device_count())
        print("   capability:", mod.cuda.get_device_capability() if mod.cuda.is_available() else None)
except Exception as e:
    print(f"[{tag}] {m}: IMPORT FAILED: {type(e).__name__}: {e}")
PY
  done
done

# ---------- Strata / vLLM: версии и коммиты ----------
{
  for repo in "$HOME/strata185/Strata"; do
    if [ ! -d "$repo/.git" ]; then echo "$repo: .git ОТСУТСТВУЕТ — commit SHA локально недоступен"; fi
    [ -d "$repo/.git" ] || continue
    echo "== $repo"
    git -C "$repo" rev-parse HEAD
    git -C "$repo" describe --tags --always 2>/dev/null
    git -C "$repo" log -1 --format='%H %ad %s' --date=iso
    git -C "$repo" status --porcelain | head -20
    git -C "$repo" remote -v
  done
} > "$OUT/repos.txt" 2>&1

# ---------- Переменные окружения (имена; значения только для CUDA/NCCL/NVIDIA) ----------
{
  echo "== env names matching CUDA|NCCL|NVIDIA|TORCH|TRITON|Strata|strata|LLAMA|HF_|OMP|MKL"
  env | grep -oiE '^[^=]*(CUDA|NCCL|NVIDIA|TORCH|TRITON|STRATA|LLAMA|HF_|OMP|MKL)[^=]*' | sort -u
  echo
  echo "== values of CUDA/NCCL/NVIDIA runtime vars (secrets excluded by pattern)"
  env | grep -E '^(CUDA|NCCL|NVIDIA|TORCH|CUDA_VISIBLE|PYTORCH)' | grep -viE 'key|token|secret|pass|pwd' | sort
} > "$OUT/env-vars.txt" 2>&1

# ---------- Окружение запущенного процесса Strata (имена + безопасные значения) ----------
{
  for pid in $(pgrep -f 'engine-cuda12/strata' 2>/dev/null); do
    echo "== PID $pid cmdline"
    tr '\0' ' ' < /proc/$pid/cmdline; echo
    echo "== PID $pid environ: names only"
    tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | sed -E 's/=.*//' | sort
    echo "== safe values (CUDA/NVIDIA/LD/STRATA/PATH, без секретов)"
    tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -E '^(CUDA|NVIDIA|LD_LIBRARY|STRATA|PATH|PYTORCH)' | grep -viE 'key|token|secret|pass|pwd'
  done
} > "$OUT/strata-process-env.txt" 2>&1

# ---------- Артефакты сборки Strata: версия, целевые архитектуры ----------
CU=$(ls "$HOME"/strata185/cuda-*/bin/cuobjdump /usr/local/cuda/bin/cuobjdump 2>/dev/null | head -1)
{
  for eng in "$HOME"/strata185/Strata/engine*/; do
    [ -d "$eng" ] || continue
    echo "== $eng"
    ls -la "$eng"
    [ -f "$eng/BUILD.json" ] && cat "$eng/BUILD.json"
    if [ -n "$CU" ]; then
      for b in "$eng"/strata "$eng"/strata-vision; do
        [ -f "$b" ] || continue
        echo "-- cuobjdump --list-elf archs: $b"
        "$CU" --list-elf "$b" 2>/dev/null | grep -oE 'sm_[0-9]+' | sort | uniq -c
        echo "-- cuobjdump --list-ptx archs: $b"
        "$CU" --list-ptx "$b" 2>/dev/null | grep -oE 'sm_[0-9]+' | sort | uniq -c
        echo "-- sha256: $(sha256sum "$b")"
      done
    else
      echo "cuobjdump не найден — архитектуры бинарных артефактов не проверены"
    fi
  done
  echo "== CMakeLists: project() и настройки архитектур"
  grep -n "project(" "$HOME"/strata185/Strata/CMakeLists.txt 2>/dev/null | head -3
  grep -rniE "CMAKE_CUDA_ARCHITECTURES|gencode|sm_[0-9]+|TORCH_CUDA_ARCH" "$HOME"/strata185/Strata/CMakeLists.txt 2>/dev/null | head -20
} > "$OUT/strata-build-artifacts.txt" 2>&1

# ---------- Конфигурация рабочего запуска (значения секретов скрыты) ----------
{
  for cfg in "$HOME"/strata185/Strata/strata-185*.json; do
    [ -f "$cfg" ] || continue
    echo "== $cfg  sha256=$(sha256sum "$cfg" | cut -d' ' -f1)"
    python3 - "$cfg" <<'PY'
import json, sys, re
p = sys.argv[1]
d = json.load(open(p))
SECRET = re.compile(r"key|token|secret|pass|pwd|auth", re.I)
for k, v in d.items():
    if SECRET.search(str(k)):
        print(f"  {k}: <скрыто: секрет>")
    else:
        print(f"  {k}: {json.dumps(v, ensure_ascii=False) if isinstance(v,(list,dict)) else v}")
PY
  done
} > "$OUT/strata-run-config.txt" 2>&1

# ---------- Службы ----------
{ systemctl list-units --type=service --state=running --no-pager --no-legend 2>&1 | head -40; } > "$OUT/services-running.txt"
{ systemctl status strata-iq3-s --no-pager 2>&1 | head -30; } > "$OUT/service-strata.txt" 2>&1

echo "DONE"
