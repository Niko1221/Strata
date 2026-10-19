#!/bin/sh
# switch.sh — Model switcher for Strata (Apple Silicon Metal)
# Interactive menu loops until "q". One-shot mode: ./switch.sh [swift|qwen|status|stop]
set -e

PORT=8000
ROOT="/Volumes/vl/ai/strata"
LOG="$ROOT/strata-server.log"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=python3

# Reverse video for the state badge, only on a real terminal.
if [ -t 1 ]; then
  REV=$(printf '\033[7m')
  SRESET=$(printf '\033[27m')
else
  REV=""
  SRESET=""
fi

get_server_pid() {
  # -sTCP:LISTEN: only the server itself.  Plain lsof -ti :PORT also lists every
  # client connected to the port, which made the badge show extra PIDs and made
  # stop_server kill the user's curl, and wrapped the detail line past the redraw.
  lsof -ti ":$PORT" -sTCP:LISTEN 2>/dev/null | head -n 1 || true
}

stop_server() {
  PID=$(get_server_pid)
  if [ -n "$PID" ]; then
    echo "Stopping active Strata server (PID: $PID) on port $PORT..."
    kill $PID 2>/dev/null || true
    for i in $(seq 1 15); do
      if ! lsof -ti ":$PORT" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    PID_LEFT=$(get_server_pid)
    if [ -n "$PID_LEFT" ]; then
      echo "Force stopping server (PID: $PID_LEFT)..."
      kill -9 $PID_LEFT 2>/dev/null || true
      sleep 1
    fi
    echo "Port $PORT is free."
  fi
}

# One GET /metrics, parsed into "BADGE<TAB>DETAIL".  The server's own state machine
# (serve/server.py Service.metrics) reports live.state, tok/s, prompt read progress
# and the queue; anything the endpoint cannot answer is labelled here.
metrics_view() {
  METRICS_JSON="$1" "$PY" - <<'PY'
import json, os
raw = os.environ.get("METRICS_JSON") or ""
try:
    m = json.loads(raw)
except Exception:
    print("ERROR\tpid is up but /metrics is unreadable")
    raise SystemExit
live = m.get("live") or {}
eng = m.get("engine") or {}
hw = m.get("hardware") or {}
state = (live.get("state") or "unknown").upper()
bits = [state]
if state == "GENERATING":
    if live.get("tok_s"):
        bits.append("%s tk/s" % live["tok_s"])
    if live.get("generated") is not None:
        bits.append("%s tok" % live["generated"])
elif state == "READING":
    pr, pt = live.get("prompt_read"), live.get("prompt_total")
    if pr is not None and pt:
        bits.append("%d%%" % int(pr * 100 / pt))
    if live.get("prefill_tok_s_mean"):
        bits.append("%d tk/s read" % round(live["prefill_tok_s_mean"]))
q = int(live.get("queued") or 0)
if q:
    bits.append("queued %d" % q)
detail = str(eng.get("model") or "?")
ctx = int(eng.get("max_context") or 0)
if ctx:
    used = 0
    if state in ("READING", "GENERATING"):
        used = int(live.get("prompt_tokens") or live.get("prompt_total") or 0) + int(live.get("generated") or 0)
    pct = int(used * 100 / ctx)
    filled = (pct + 5) // 10
    bar = "#" * filled + "-" * (10 - filled)
    k = lambda n: ("%dk" % (n // 1000)) if n >= 1000 else str(n)
    detail += " | ctx %3d%% [%s] %s/%s" % (pct, bar, k(used), k(ctx))
if hw.get("gpu_util") is not None:
    detail += " | GPU %s%%" % hw["gpu_util"]
print("\t".join([" ".join(bits), detail]))
PY
}

# Sets BADGE and DETAIL for the current server state.
status_view() {
  PID=$(get_server_pid)
  if [ -z "$PID" ]; then
    BADGE="STOPPED"
    DETAIL="no Strata server on port $PORT"
    return 0
  fi
  JSON=$(curl -s -m 2 "http://127.0.0.1:$PORT/metrics" 2>/dev/null || true)
  if [ -z "$JSON" ]; then
    BADGE="STARTING"
    DETAIL="PID $PID on port $PORT — /metrics not answering yet"
    return 0
  fi
  VIEW=$(metrics_view "$JSON")
  BADGE=$(printf '%s' "$VIEW" | cut -f1)
  DETAIL=$(printf '%s' "$VIEW" | cut -f2-)
  DETAIL="$DETAIL | PID $PID"
  # Keep the badge+detail block at exactly two terminal lines: the live monitor's
  # redraw (up 2, clear) only stays correct if nothing wraps.
  W=$(tput cols 2>/dev/null || echo 80)
  BADGE=$(printf '%s' "$BADGE" | cut -c1-$((W - 2)))
  DETAIL=$(printf '%s' "$DETAIL" | cut -c1-$W)
}

show_status() {
  status_view
  printf '%s %s %s\n' "$REV" "$BADGE" "$SRESET"
  echo "$DETAIL"
}

# Live monitor: redraws the badge once a second until Ctrl+C.
monitor() {
  MON_STOP=0
  if [ -t 1 ]; then
    trap 'MON_STOP=1' INT
  fi
  echo "Live monitor (1 s) — Ctrl+C returns to the menu."
  echo ""
  while :; do
    status_view
    printf '%s %s %s\n' "$REV" "$BADGE" "$SRESET"
    printf '%s\n' "$DETAIL"
    sleep 1 || true
    if [ "$MON_STOP" -eq 1 ]; then
      break
    fi
    if [ -t 1 ]; then
      printf '\033[2A\033[J'
    fi
  done
  trap - INT
  echo ""
}

# Start a server in the background, then poll /metrics until it answers.
# The menu loop can continue while the server keeps running.
start_server_background() {
  SCRIPT="$1"
  stop_server
  nohup "$ROOT/$SCRIPT" >"$LOG" 2>&1 &
  echo "Starting $SCRIPT (log: $LOG)..."
  printf "Waiting for server to be ready"
  i=0
  while [ "$i" -lt 90 ]; do
    if curl -s -m 2 "http://127.0.0.1:$PORT/metrics" >/dev/null 2>&1; then
      echo " — ready."
      return 0
    fi
    printf "."
    sleep 2
    i=$((i + 1))
  done
  echo " — not ready yet; still loading. Check $LOG if it never comes up."
}

show_menu() {
  echo "=================================================="
  echo " Strata Model Switcher"
  echo "=================================================="
  echo "Currently active:"
  show_status
  echo ""
  echo "Select a model to start:"
  echo "  1) Swift 1.5 Abliterated IQ3_S (512k context, vision)"
  echo "  2) Qwen 3.8 Flash Next IQ3_S (512k context, vision)"
  echo "  3) Show status"
  echo "  4) Stop active server"
  echo "  5) Live monitor (state, tk/s, reading %, queue)"
  echo "  q) Quit"
  printf "Choice [1-5, q]: "
}

TARGET="$1"

# One-shot mode: keep the old foreground behavior for scripted use.
if [ -n "$TARGET" ]; then
  case "$TARGET" in
    swift|1)
      stop_server
      echo "Starting Swift 1.5 Abliterated IQ3_S..."
      exec "$ROOT/run-swift.sh"
      ;;
    qwen|2)
      stop_server
      echo "Starting Qwen 3.8 Flash Next IQ3_S..."
      exec "$ROOT/run-iq3_s.sh"
      ;;
    status)
      show_status
      ;;
    stop)
      stop_server
      ;;
    *)
      echo "Usage: ./switch.sh [swift | qwen | status | stop]"
      exit 1
      ;;
  esac
  exit 0
fi

# Interactive mode: menu always at the top of the window.
while :; do
  clear
  show_menu
  read -r CHOICE || exit 0
  case "$CHOICE" in
    1|swift)
      start_server_background run-swift.sh
      ;;
    2|qwen)
      start_server_background run-iq3_s.sh
      ;;
    3|status)
      echo ""
      show_status
      printf "\nPress Enter to return to the menu... "
      read -r _ || exit 0
      ;;
    4|stop)
      stop_server
      ;;
    5|monitor)
      echo ""
      monitor
      printf "Press Enter to return to the menu... "
      read -r _ || exit 0
      ;;
    q|Q)
      exit 0
      ;;
  esac
done
