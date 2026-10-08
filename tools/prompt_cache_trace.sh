#!/usr/bin/env bash
# Run the prompt-cache workloads against a live Strata server with trace logging on,
# then analyze the engine's prompt-cache lines.
#
#   tools/prompt_cache_trace.sh                       # rebuild engine, start server, run all scenarios, analyze
#   tools/prompt_cache_trace.sh --scenario pin_siblings
#   REBUILD=0 tools/prompt_cache_trace.sh             # skip the engine rebuild
#   BUILD_DIR=build-sycl tools/prompt_cache_trace.sh  # rebuild and run the SYCL engine
#   CONFIG=strata-coder-iq1_m.json tools/prompt_cache_trace.sh
#
# Env: CONFIG, PYTHON, REBUILD, BUILD_DIR, CACHE_MIB, CACHE_SLOTS, PROMPT_CACHE,
#      RUNNING_REUSE, CHECKPOINT_MOVE (default 1).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/strata-iq3_xxs.json}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"
REBUILD="${REBUILD:-1}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
case "$BUILD_DIR" in /*) ;; *) BUILD_DIR="$ROOT/$BUILD_DIR" ;; esac
CACHE_MIB="${CACHE_MIB:-4096}"
CACHE_SLOTS="${CACHE_SLOTS:-4}"
PROMPT_CACHE="${PROMPT_CACHE:-6}"
RUNNING_REUSE="${RUNNING_REUSE:-1}"
CHECKPOINT_MOVE="${CHECKPOINT_MOVE:-1}"
SCENARIO=""
while [ $# -gt 0 ]; do
  case "$1" in
    --scenario) SCENARIO="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }

mapfile -d '' -t CONFIG_VALUES < <("$PYTHON" - "$CONFIG" <<'PY'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8-sig"))
for value in (c.get("exe", ""), c.get("log", ""), c.get("port", 8080)):
    sys.stdout.buffer.write(str(value).encode() + b"\0")
PY
)
[ "${#CONFIG_VALUES[@]}" -eq 3 ] || { echo "could not read engine settings from config" >&2; exit 1; }
EXE="${CONFIG_VALUES[0]}"
LOG="${CONFIG_VALUES[1]}"
PORT="${CONFIG_VALUES[2]}"
[ -n "$EXE" ] || { echo "config has no \"exe\" path" >&2; exit 1; }
[ -n "$LOG" ] || { echo "config has no \"log\" path; the engine trace would be discarded" >&2; exit 1; }
URL="http://127.0.0.1:${PORT}"
if "$PYTHON" - "$PORT" <<'PY'
import socket, sys
try:
    connection = socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1)
except OSError:
    raise SystemExit(1)
connection.close()
PY
then
  echo "port $PORT is already in use; stop the existing server or choose another port" >&2
  exit 1
fi
WORK="$(mktemp -d)"
SERVER_LOG="$WORK/server.out"
ENGINE_LOG="$WORK/prompt-cache.log"
TRACE_CONFIG="$WORK/config.json"

TRACE_EXE="$EXE"
if [ "$REBUILD" = "1" ]; then
  echo "rebuilding the engine in $BUILD_DIR (the trace instrumentation is in generate.cpp) ..."
  cmake --build "$BUILD_DIR" --target strata
  TRACE_EXE="$BUILD_DIR/strata"
  [ -x "$TRACE_EXE" ] || { echo "rebuilt engine not found: $TRACE_EXE" >&2; exit 1; }
fi

"$PYTHON" - "$CONFIG" "$TRACE_CONFIG" "$CACHE_MIB" "$CACHE_SLOTS" "$PROMPT_CACHE" "$TRACE_EXE" <<'PY'
import json, sys

source, target, cache_mib, cache_slots, prompt_cache, exe = sys.argv[1:]
cfg = json.load(open(source, encoding="utf-8-sig"))
overrides = {
    "--conversation-cache-mib": cache_mib,
    "--conversation-cache-slots": cache_slots,
    "--prompt-cache": prompt_cache,
}
args = cfg.get("args", [])
kept = []
i = 0
while i < len(args):
    name = args[i].split("=", 1)[0]
    if name in overrides:
        i += 1 if "=" in args[i] else 2
    else:
        kept.append(args[i])
        i += 1
for name, value in overrides.items():
    kept.extend([name, value])
cfg["args"] = kept
cfg["host"] = "127.0.0.1"
cfg["exe"] = exe
with open(target, "w", encoding="utf-8") as stream:
    json.dump(cfg, stream, indent=2)
    stream.write("\n")
PY

log_offset() { [ -f "$LOG" ] && wc -c < "$LOG" || echo 0; }
START_OFFSET="$(log_offset)"

export STRATA_PROMPT_CACHE_TRACE=1
export STRATA_RUNNING_STATE_REUSE="$RUNNING_REUSE"
export STRATA_CHECKPOINT_MOVE="$CHECKPOINT_MOVE"
echo "cache configuration: ${CACHE_MIB} MiB, ${CACHE_SLOTS} parked slots, ${PROMPT_CACHE} prompt checkpoints"
echo "running-state allocation reuse: ${RUNNING_REUSE}"
echo "checkpoint-chain move on branch switch: ${CHECKPOINT_MOVE}"
echo "starting server: $PYTHON serve/server.py --engine strata --config $TRACE_CONFIG --host 127.0.0.1 --port $PORT"
( cd "$ROOT" && "$PYTHON" serve/server.py --engine strata --config "$TRACE_CONFIG" --host 127.0.0.1 --port "$PORT" ) \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  pkill -P "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "waiting for the model to load (first start can take a minute or two) ..."
"$PYTHON" - "$URL" "$SERVER_PID" <<'PY'
import json, os, sys, time, urllib.request
url, pid = sys.argv[1], int(sys.argv[2])
deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        print("server process exited before becoming ready", file=sys.stderr)
        sys.exit(1)
    try:
        with urllib.request.urlopen(url + "/health", timeout=5) as r:
            health = json.load(r)
            if r.status == 200 and health.get("service") == "strata" and health.get("loaded") is True:
                print("server ready"); sys.exit(0)
    except Exception:
        pass
    time.sleep(2)
print("server did not become ready in time", file=sys.stderr); sys.exit(1)
PY

echo "running workloads against $URL ..."
if [ -n "$SCENARIO" ]; then
  ( cd "$ROOT" && "$PYTHON" tools/prompt_cache_workloads.py --run --url "$URL" --scenario "$SCENARIO" )
else
  ( cd "$ROOT" && "$PYTHON" tools/prompt_cache_workloads.py --run --url "$URL" )
fi

cleanup; trap - EXIT INT TERM

# Keep only this run's engine lines (the log is appended across runs).
tail -c +"$((START_OFFSET + 1))" "$LOG" >"$ENGINE_LOG" 2>/dev/null || cp "$LOG" "$ENGINE_LOG"
echo
echo "prompt-cache trace captured: $ENGINE_LOG"
"$PYTHON" "$ROOT/tools/analyze_prompt_cache_log.py" "$ENGINE_LOG"
