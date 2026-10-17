#!/usr/bin/env bash
set -euo pipefail

APP=/models/strata/app
BASE=/models/strata/benchmarks/2026-10-17-v0145-1to4-rtx3090
OUT="${BASE}/128k-extension"
URL=http://127.0.0.1:8080
HELPER="${OUT}/bench-128k.py"
SERVER_PID=
GPU_MONITOR_PID=
SYSTEM_MONITOR_PID=
COUNTS=("$@")
if [[ "${#COUNTS[@]}" -eq 0 ]]; then
  COUNTS=(1 2 3 4)
fi

stop_processes() {
  for pid_name in GPU_MONITOR_PID SYSTEM_MONITOR_PID; do
    pid="${!pid_name}"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
    printf -v "${pid_name}" '%s' ''
  done
  if [[ -n "${SERVER_PID}" ]] && kill -0 -- "-${SERVER_PID}" 2>/dev/null; then
    kill -INT -- "-${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 -- "-${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- "-${SERVER_PID}" 2>/dev/null; then
      kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
      for _ in $(seq 1 15); do
        kill -0 -- "-${SERVER_PID}" 2>/dev/null || break
        sleep 1
      done
    fi
    kill -0 -- "-${SERVER_PID}" 2>/dev/null && kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=
}

trap stop_processes EXIT

cd "${APP}"
sudo -n systemctl stop strata-qwen38.service
if curl -fsS "${URL}/health" >/dev/null 2>&1; then
  echo "Port 8080 is already serving Strata; refusing to mix runs" >&2
  exit 1
fi

mkdir -p "${OUT}/validation"
date -u +%FT%TZ > "${OUT}/started-utc.txt"
nvidia-smi --query-gpu=index,pci.bus_id,uuid,name,vbios_version,memory.total,power.limit,temperature.gpu --format=csv > "${OUT}/validation/matrix-gpu-before.csv"
free -b > "${OUT}/validation/matrix-memory-before.txt"
sudo -n dmesg --level=err,warn > "${OUT}/validation/matrix-dmesg-before.txt" 2>/dev/null || true

for count in "${COUNTS[@]}"; do
  label="${count}gpu"
  run_dir="${OUT}/${label}"
  source_config="${BASE}/configs/${label}.json"
  config="${run_dir}/config.json"
  result="${run_dir}/results.json"
  if [[ -e "${result}" ]]; then
    echo "${label}: result already exists; refusing to overwrite" >&2
    exit 1
  fi
  mkdir -p "${run_dir}"
  jq --arg log "${run_dir}/engine.log" '.log = $log' "${source_config}" > "${config}"

  date -u +%FT%TZ > "${run_dir}/started-utc.txt"
  nvidia-smi --query-gpu=index,pci.bus_id,uuid,name,vbios_version,memory.total,power.limit,temperature.gpu --format=csv > "${run_dir}/gpu-before.csv"
  free -b > "${run_dir}/memory-before.txt"
  for pci in 01 47 81 c1; do
    sudo -n lspci -s "${pci}:00.0" -vv > "${OUT}/validation/${label}-${pci}-before.txt"
  done

  setsid env PYTHONUNBUFFERED=1 "${APP}/.venv/bin/python" "${APP}/serve/server.py" \
    --engine strata --config "${config}" --port 8080 > "${run_dir}/server-console.log" 2>&1 &
  SERVER_PID=$!
  echo "${SERVER_PID}" > "${run_dir}/server.pid"

  ready=0
  for _ in $(seq 1 900); do
    if curl -fsS "${URL}/health" > "${run_dir}/health.json" 2>/dev/null; then
      ready=1
      break
    fi
    kill -0 "${SERVER_PID}" 2>/dev/null || break
    sleep 1
  done
  if [[ "${ready}" -ne 1 ]]; then
    echo "${label}: server did not become ready" >&2
    tail -n 160 "${run_dir}/server-console.log" >&2 || true
    exit 1
  fi

  nvidia-smi --query-gpu=timestamp,index,pci.bus_id,uuid,utilization.gpu,memory.used,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current --format=csv -l 1 > "${run_dir}/gpu-telemetry.csv" &
  GPU_MONITOR_PID=$!
  vmstat -w 1 > "${run_dir}/system-telemetry.txt" &
  SYSTEM_MONITOR_PID=$!

  "${APP}/.venv/bin/python" "${HELPER}" \
    --app "${APP}" --url "${URL}" --config "${config}" --runs 3 --out "${result}"

  kill "${GPU_MONITOR_PID}" "${SYSTEM_MONITOR_PID}" 2>/dev/null || true
  wait "${GPU_MONITOR_PID}" 2>/dev/null || true
  wait "${SYSTEM_MONITOR_PID}" 2>/dev/null || true
  GPU_MONITOR_PID=
  SYSTEM_MONITOR_PID=
  nvidia-smi --query-gpu=index,pci.bus_id,uuid,memory.used,power.draw,power.limit,temperature.gpu --format=csv > "${run_dir}/gpu-after.csv"
  free -b > "${run_dir}/memory-after.txt"

  stop_processes
  cp "${run_dir}/engine.log" "${run_dir}/engine-log.txt"

  for pci in 01 47 81 c1; do
    sudo -n lspci -s "${pci}:00.0" -vv > "${OUT}/validation/${label}-${pci}-after.txt"
  done
  sudo -n dmesg --level=err,warn > "${OUT}/validation/${label}-dmesg.txt" 2>/dev/null || true
  date -u +%FT%TZ > "${run_dir}/finished-utc.txt"
done

sudo -n systemctl stop strata-qwen38.service
nvidia-smi --query-gpu=index,pci.bus_id,uuid,memory.used,pstate,power.draw,power.limit,temperature.gpu --format=csv > "${OUT}/final-idle.csv"
free -b > "${OUT}/validation/matrix-memory-after.txt"
sudo -n dmesg --level=err,warn > "${OUT}/validation/matrix-dmesg-after.txt" 2>/dev/null || true
date -u +%FT%TZ > "${OUT}/finished-utc.txt"
