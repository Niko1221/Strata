#!/bin/bash
# Stop the serving task (frees the GPU), run the KV speed+accuracy A/B, restart the serving task.
set -u
# Paths: the repo is this script's folder; model data is $STRATA_DATA, else the Strata-data folder next to it.
ROOT=$(cd "$(dirname "$0")" && (pwd -W 2>/dev/null || pwd))   # pwd -W: a native path python.exe reads under git-bash
DATA=${STRATA_DATA:-$ROOT/../Strata-data}
PY=${PYTHON:-$ROOT/.venv/Scripts/python.exe}
# STRATA_SERVICE_TASK: a scheduled task serving on this GPU - stopped for the run, restarted after (unset: none).
svc_stop() { if [ -n "${STRATA_SERVICE_TASK:-}" ]; then schtasks /End /TN "$STRATA_SERVICE_TASK" >/dev/null 2>&1; fi; }
svc_start() { if [ -n "${STRATA_SERVICE_TASK:-}" ]; then schtasks /Run /TN "$STRATA_SERVICE_TASK" >/dev/null 2>&1; fi; }
svc_stop
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*serve/server.py*' -or \$_.Name -eq 'strata.exe' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" >/dev/null 2>&1
sleep 5
"$PY" "$ROOT/tools/kv_ab_report.py" 2>&1
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'strata.exe' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" >/dev/null 2>&1
sleep 3
svc_start
echo "=== service restarted ==="
