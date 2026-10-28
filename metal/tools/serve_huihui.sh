#!/bin/bash
# metal/tools/serve_huihui.sh - stop / run-tests / restart the 8180 Huihui server safely.
#
#   serve_huihui.sh status              who is serving, memory
#   serve_huihui.sh stop                graceful TERM, then KILL; waits for the port to close
#   serve_huihui.sh start               start exactly the config's server line; wait for loaded + one live answer
#   serve_huihui.sh run CMD...          stop; run CMD (any exit); ALWAYS start again; health-check
#
# Never runs two engines: it will not start while anything answers on the port.
# Designed to survive the assistant losing its own model mid-run: the restart always follows the test.
set -u
ROOT=/Users/pc/src/Strata-fork
CFG=${HUIHUI_CFG:-$ROOT/strata-ud-q4_k_xl-huihui.json}
PORT=${HUIHUI_PORT:-8180}
MODEL=$(python3 -c "import json;print(json.load(open('$CFG'))['model_name'])" 2>/dev/null || echo qwen3.8-flash-next-ud-q4_k_xl-huihui)
RLOG=/tmp/bw/restart.log

say() { echo "[serve_huihui $(date +%H:%M:%S)] $*"; }
port_pid() { lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1; }
engine_pid() {
    local sp; sp=$(port_pid)
    [ -z "$sp" ] && return 0
    pgrep -P "$sp" 2>/dev/null | while read -r p; do ps -o comm= -p "$p" 2>/dev/null | grep -q strata && echo "$p"; done
}

status() {
    local p e
    p=$(port_pid)
    if [ -z "$p" ]; then say "port $PORT: nobody"; return 1; fi
    e=$(engine_pid)
    say "port $PORT: server.py pid $p, engine pid ${e:-?} ($(ps -o rss= -p "${e:-0}" 2>/dev/null | awk '{printf "%.1f GB", $1/1048576}') engine RSS)"
}

stop() {
    local p e
    p=$(port_pid)
    if [ -z "$p" ]; then say "already down"; return 0; fi
    say "stopping server pid $p (and children)"
    pkill -TERM -P "$p" 2>/dev/null
    kill -TERM "$p" 2>/dev/null
    for _ in $(seq 60); do
        p=$(port_pid)
        [ -z "$p" ] && break
        sleep 0.5
    done
    p=$(port_pid)
    if [ -n "$p" ]; then
        say "still listening after 30s: KILL"
        pkill -KILL -P "$p" 2>/dev/null
        kill -KILL "$p" 2>/dev/null
        sleep 2
    fi
    for e in $(pgrep -x strata 2>/dev/null); do kill -TERM "$e" 2>/dev/null; done
    sleep 3
    for e in $(pgrep -x strata 2>/dev/null); do kill -KILL "$e" 2>/dev/null; done
    if [ -n "$(port_pid)" ]; then say "ERROR: port $PORT still held"; return 1; fi
    say "down; free: $(vm_stat | awk '/free pages/{f=$3} /speculative/{s=$3} END{printf "%.1f GB", (f+s)*4096/1073741824}')"
}

wait_ready() {  # $1 = seconds
    local deadline=$(( $(date +%s) + ${1:-600} ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if curl -s --max-time 4 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"loaded"'; then
            say "models loaded"
            return 0
        fi
        sleep 2
    done
    say "TIMEOUT waiting for /v1/models"
    return 1
}

probe_answer() {  # one real completion; content or reasoning counts as life
    local deadline out
    deadline=$(( $(date +%s) + ${1:-120} ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        out=$(curl -s --max-time 90 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
            -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: pong\"}],\"max_tokens\":64,\"temperature\":0}")
        if echo "$out" | grep -qE '"content": ?"[^"]|"reasoning'; then
            say "live answer ok"
            return 0
        fi
        sleep 3
    done
    say "PROBE TIMEOUT: no completion in window"
    return 1
}

start() {
    if [ -n "$(port_pid)" ]; then say "ERROR: something already answers on $PORT; refusing to start a second engine"; return 1; fi
    say "starting: serve/server.py --engine strata --config $CFG --port $PORT"
    cd "$ROOT" || return 1
    nohup python3 serve/server.py --engine strata --config "$CFG" --port "$PORT" >> "$RLOG" 2>&1 &
    wait_ready 600 || { tail -20 "$RLOG"; return 1; }
    probe_answer 240 || { tail -20 "$RLOG"; return 1; }
    say "serving again"
}

case "${1:-status}" in
    status) status ;;
    stop)   stop ;;
    start)  start ;;
    run)
        shift
        mkdir -p /tmp/bw
        stop || { say "stop failed; aborting test, server left as is"; exit 2; }
        ( "$@"; echo "[harness] test exit: $?" ) 2>&1 | tee /tmp/bw/test-output.log
        tesc=${PIPESTATUS[0]}
        start || { say "RESTART FAILED - retry manually: $0 start"; exit 3; }
        say "done (test exit $tesc)"
        ;;
    *) echo "usage: serve_huihui.sh {status|stop|start|run CMD...}"; exit 64 ;;
esac
