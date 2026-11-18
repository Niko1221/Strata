#!/bin/bash
# measure.sh <arms-file>: the gap window when the Strata server is the agent's own model (the other local
# serve process must stay off). Stops the running Strata server, frees its RAM, runs the arms with gap.sh
# (that other server untouched), restarts the server.
# Foreground is best - progress streams to your terminal, and Ctrl-C or a closed terminal still restarts the server:
#   ./gap/measure.sh gap/arms-round13.txt 2>&1 | tee /tmp/round13.log
# Detached instead: nohup ./gap/measure.sh gap/arms-round13.txt > /tmp/round13.log 2>&1 &   then:  tail -f /tmp/round13.log
set -u
FORK=/Users/pc/src/Strata-fork
ARMS=${1:?usage: measure.sh arms-file}
OUT=$FORK/gap/results/$(date +%m%d-%H%M); mkdir -p "$OUT"
RESTARTED=0
trap 'restart_server' EXIT INT TERM
echo "measure start $(date +%T)  log: $OUT/arms.log"

server_pids() { pgrep -f "serve/server.py"; pgrep -f "build-metal/strata --serve"; }

stop_server() {
  local pids i
  pids=$(server_pids)
  [[ -z $pids ]] && { echo "strata server not running at start"; return; }
  echo "$pids" > "$OUT/server-pids.txt"
  kill $pids 2>/dev/null
  for i in {1..30}; do [[ -z $(server_pids) ]] && break; sleep 1; done
  pkill -9 -f "serve/server.py" 2>/dev/null; pkill -9 -f "build-metal/strata --serve" 2>/dev/null
  echo "strata server stopped $(date +%T)"
  # the engine wires its arena (mlock); wait until the wired pages are back
  for i in {1..60}; do
    local free=$(vm_stat | awk '/Pages free|Pages inactive|Pages speculative|Pages purgeable/ {gsub(/ *\.$| *pages:/,""); s+=$3} END {print s+0}')
    [[ $free -ge 2200000 ]] && { echo "RAM free: $((free*16384/1073741824)) GiB after $((i*2))s"; return; }
    sleep 2
  done
  echo "WARNING: RAM still short after 120s - arms may fail with 'no arena region'" >&2
}

restart_server() {
  [[ $RESTARTED == 1 ]] && return
  if [[ -z $(server_pids) ]]; then
    nohup "$FORK/run-iq3_xxs.sh" > "$OUT/server-restore.log" 2>&1 &
  fi
  local i
  for i in {1..150}; do
    sleep 2
    curl -s -m 3 http://127.0.0.1:8180/v1/models 2>/dev/null | grep -Eq '"value": ?"loaded"' && {
      RESTARTED=1; echo "strata server restored after ~$((i*2))s $(date +%T)"; return; }
  done
  echo "SERVER NOT RESTORED after 300s - check $OUT/server-restore.log" >&2
}

srv=$(cat "$FORK/gap/.server" 2>/dev/null); [[ -n ${srv:-} ]] && pgrep -x "$srv" >/dev/null && echo "WARNING: $srv is running; it steals RAM the arena needs - stop it first" >&2
stop_server
GAP_NO_SERVER=1 "$FORK/gap/gap.sh" "$ARMS" 2>&1 | tee "$OUT/arms.log"
restart_server
echo "done $(date +%T): $OUT"
echo "arms: $(ls -1dt "$FORK/gap/results"/*/ | head -1)"
