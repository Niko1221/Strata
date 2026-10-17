#!/usr/bin/env bash
set -euo pipefail

APP=/models/strata/app
OUT=/models/strata/benchmarks/2026-10-17-v0145-1to4-rtx3090
URL=http://127.0.0.1:8080
SERVER_PID=
MONITOR_PID=
COUNTS=("$@")
if [[ "${#COUNTS[@]}" -eq 0 ]]; then
  COUNTS=(1 2 3 4)
fi

stop_processes() {
  if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  MONITOR_PID=
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

mkdir -p "${OUT}/internal-validation"

for count in "${COUNTS[@]}"; do
  label="${count}gpu"
  run_dir="${OUT}/${label}"
  config="${OUT}/configs/${label}.json"
  if [[ -e "${run_dir}/official/results.json" ]]; then
    echo "${label}: results already exist; refusing to overwrite" >&2
    exit 1
  fi
  mkdir -p "${run_dir}/official"

  date -u +%FT%TZ > "${run_dir}/started-utc.txt"
  nvidia-smi --query-gpu=index,pci.bus_id,uuid,name,vbios_version,memory.total,power.limit,temperature.gpu --format=csv > "${run_dir}/gpu-before.csv"
  sudo -n lspci -s 01:00.0 -vv > "${OUT}/internal-validation/${label}-01-before.txt"
  sudo -n lspci -s 47:00.0 -vv > "${OUT}/internal-validation/${label}-47-before.txt"
  sudo -n lspci -s 81:00.0 -vv > "${OUT}/internal-validation/${label}-81-before.txt"
  sudo -n lspci -s c1:00.0 -vv > "${OUT}/internal-validation/${label}-c1-before.txt"

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
  MONITOR_PID=$!

  "${APP}/.venv/bin/python" "${APP}/tools/strata_bench.py" \
    --url "${URL}" --runs 3 --long --out "${run_dir}/official"

  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
  MONITOR_PID=
  nvidia-smi --query-gpu=index,pci.bus_id,uuid,memory.used,power.draw,power.limit,temperature.gpu --format=csv > "${run_dir}/gpu-after.csv"

  stop_processes
  cp "${run_dir}/engine.log" "${run_dir}/official/engine-log.txt"
  cp "${config}" "${run_dir}/official/config.json"

  sudo -n lspci -s 01:00.0 -vv > "${OUT}/internal-validation/${label}-01-after.txt"
  sudo -n lspci -s 47:00.0 -vv > "${OUT}/internal-validation/${label}-47-after.txt"
  sudo -n lspci -s 81:00.0 -vv > "${OUT}/internal-validation/${label}-81-after.txt"
  sudo -n lspci -s c1:00.0 -vv > "${OUT}/internal-validation/${label}-c1-after.txt"
  sudo -n dmesg --level=err,warn > "${OUT}/internal-validation/${label}-dmesg.txt" 2>/dev/null || true
  date -u +%FT%TZ > "${run_dir}/finished-utc.txt"
done

sudo -n systemctl stop strata-qwen38.service
nvidia-smi --query-gpu=index,pci.bus_id,uuid,memory.used,pstate,power.draw,temperature.gpu --format=csv > "${OUT}/final-idle.csv"
