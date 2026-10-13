#!/bin/bash
# A/B the FP16 sliding window + attention sinks (kv_window.hpp) through the real serving path:
# same engine (the locally built one), same prompt, same seed - once with the overlay off, once with it on.
# The serving scheduled task is stopped for the run and restarted at the end.
set -u
PORT=8083
ROOT=C:/Users/blsor/Strata   # native paths: git-bash does not translate them for python.exe here
PY="$ROOT/.venv/Scripts/python.exe"
CFG="$ROOT/strata-ab-kv-window.json"
KEY=$(cat "C:/Users/blsor/Strata-data/strata_api_key.txt")
PROMPT='{"model":"swift-1.5-iq3_xxs","messages":[{"role":"user","content":"Explain in one paragraph what a KV cache is in a transformer and why its size matters for long-context inference."}],"temperature":0,"max_tokens":160,"seed":4242}'
LOG=/tmp/ab-engine-server.log

stop_engine() {
  powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*serve/server.py*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" >/dev/null 2>&1
  taskkill /F /IM strata.exe >/dev/null 2>&1
  sleep 5
}

wait_ready() {
  for i in $(seq 1 200); do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
    [ "$code" = "200" ] && return 0
    sleep 3
  done
  return 1
}

run_arm() {
  local arm="$1"; shift
  echo "=== arm $arm (window ${1:-off}) ==="
  stop_engine
  rm -f "$LOG"
  # the engine exe comes from the config JSON (build/strata.exe)
  env "$@" "$PY" "$ROOT/serve/server.py" --engine strata --config "$CFG" --port $PORT > "$LOG" 2>&1 &
  if ! wait_ready; then
    echo "arm $arm: the server never became ready"
    tail -20 "$LOG"
    return 1
  fi
  echo "arm $arm: ready"
  curl -s --max-time 900 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
       -d "$PROMPT" "http://127.0.0.1:$PORT/v1/chat/completions" > "/tmp/ab-$arm.json"
  echo "arm $arm: response $(wc -c < "/tmp/ab-$arm.json") bytes"
  stop_engine
}

echo "stopping the serving task"
schtasks /End /TN StrataServe >/dev/null 2>&1
run_arm off
run_arm on STRATA_KV_WINDOW=128 STRATA_KV_SINK=8

echo "restarting the serving task"
schtasks /Run /TN StrataServe >/dev/null 2>&1
echo "=== done ==="
