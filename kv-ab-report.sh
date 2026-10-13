#!/bin/bash
# Stop the serving task (frees the GPU), run the KV speed+accuracy A/B, restart the serving task.
set -u
ROOT=C:/Users/blsor/Strata
schtasks /End /TN StrataServe >/dev/null 2>&1
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*serve/server.py*' -or \$_.Name -eq 'strata.exe' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" >/dev/null 2>&1
sleep 5
"$ROOT/.venv/Scripts/python.exe" "$ROOT/tools/kv_ab_report.py" 2>&1
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'strata.exe' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" >/dev/null 2>&1
sleep 3
schtasks /Run /TN StrataServe >/dev/null 2>&1
echo "=== service restarted ==="
