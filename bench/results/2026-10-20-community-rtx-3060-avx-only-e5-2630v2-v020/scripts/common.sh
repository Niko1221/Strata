# Shared settings for the T0-T7 scripts (sourced, not run). See STRATA_TEST_RUNBOOK_20261016.md.
TOP=<WORKDIR>
T=$TOP/experiments/06_test                    # this folder
S=$TOP/experiments/03_run/Strata              # the installed Strata (setup's copy; its engine is engine/strata)
B=$TOP/experiments/04_bench                   # with_gpu_lease.py, numa_run.py
PYS=$TOP/experiments/03_run/.venv-setup/bin/python   # Strata's own Python (server, strata_bench, needle_bench)
PYE=$T/.venv-eval/bin/python                  # EvalPlus 0.3.1 (offline; dataset in $T/cache)
PYL=<WORKSPACE>/rag-pre-1/.venv/bin/python      # rag-pre's Python, for the GPU Jobs lease
PORT=8080
export XDG_CACHE_HOME=$T/cache                # EvalPlus reads HumanEval+ from here, never from the network
export PYTHONWARNINGS=ignore

# One folder per test day; a test run twice on the same day gets -2, -3, ...
DAY=${STRATA_TEST_DAY:-$(date +%F)}
R=$T/results/$DAY

newdir() {   # newdir NAME -> prints a fresh folder under $R
  local d=$R/$1 i=2
  while [ -e "$d" ]; do d=$R/$1-$i; i=$((i + 1)); done
  mkdir -p "$d" && echo "$d"
}

stamp() { date '+%F %T'; }
say() { echo "[$(stamp)] $*"; }

# make_config SRC.json OUT.json LOG [CTX]: a copy of a model config whose engine log goes to LOG (and context CTX)
make_config() {
  python3 - "$@" <<'EOF'
import json, sys
src, out, log = sys.argv[1:4]
c = json.load(open(src))
c["log"] = log
if len(sys.argv) > 4:
    a = c["args"]; a[a.index("--max-context") + 1] = sys.argv[4]
json.dump(c, open(out, "w"), indent=1)
EOF
}

# preflight: refuse to start when something else holds the port or an engine is running
preflight() {
  if pgrep -x strata > /dev/null || pgrep -f "serve/server.py --engine" > /dev/null; then
    echo "STOP: a Strata server or engine is already running (start_strata.sh?). Stop it with Ctrl+C first." >&2
    pgrep -a -x strata >&2; pgrep -af "serve/server.py --engine" >&2
    return 1
  fi
  if python3 -c "import socket,sys; s=socket.socket(); sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT)) else 1)"; then :; else
    echo "STOP: port $PORT is in use." >&2; return 1
  fi
  [ -x "$PYL" ] || { echo "STOP: $PYL not found (GPU Jobs lease)" >&2; return 1; }
  return 0
}

# under_lease LOG DESCRIPTION -- CMD...: run CMD while holding the GPU Jobs lease (other GPU services released)
under_lease() {
  local log=$1 desc=$2; shift 3
  "$PYL" "$B/with_gpu_lease.py" --log "$log" --description "$desc" -- "$@"
}

# server_up CONFIG LOGFILE: start the server on NUMA node 1 (as start_strata.sh does) and wait until it is loaded
server_up() {
  local cfg=$1 log=$2
  ( cd "$S" && exec python3 "$B/numa_run.py" --node 1 -- "$PYS" serve/server.py --engine strata \
      --config "$cfg" --port $PORT ) > "$log" 2>&1 &
  SERVER_PID=$!
  say "server starting (pid $SERVER_PID), waiting for the model to load ..."
  local t0=$(date +%s)
  while :; do
    if ! kill -0 $SERVER_PID 2> /dev/null; then say "server exited while loading: see $log"; return 1; fi
    if curl -s --max-time 5 http://127.0.0.1:$PORT/health | grep -q '"loaded": *true'; then break; fi
    if [ $(( $(date +%s) - t0 )) -gt 1800 ]; then say "server not ready after 30 min"; return 1; fi
    sleep 5
  done
  say "server ready after $(( $(date +%s) - t0 )) s"
}

server_down() {
  [ -n "${SERVER_PID:-}" ] || return 0
  kill -INT $SERVER_PID 2> /dev/null
  for _ in $(seq 60); do kill -0 $SERVER_PID 2> /dev/null || break; sleep 2; done
  kill -0 $SERVER_PID 2> /dev/null && { kill -TERM $SERVER_PID; sleep 5; }
  pkill -x strata 2> /dev/null
  SERVER_PID=
  say "server stopped"
}

# the model config every test uses: context 40,960 (so the 32K tests fit), otherwise setup's defaults.
# STRATA_QUANT picks the model (T2-T6):
#   q2 (default): ISTA-DASLab IQ2_XS (~2-bit), the official comparison (RTX 3060 report)  -> results/<day>/T2-speed ...
#   q4:           Unsloth UD-IQ4_XS (~4-bit), about half the speed                        -> results/<day>/T2-speed-q4 ...
QUANT=${STRATA_QUANT:-q2}
case $QUANT in
  q2) BASE_CFG=$B/cfg-ctx40k-auto.json;    QTAG=;    QNAME=iq2_xs ;;
  q4) BASE_CFG=$B/cfg-ctx40k-auto-q4.json; QTAG=-q4; QNAME=ud-iq4_xs ;;
  *) echo "STRATA_QUANT must be q2 or q4 (got '$QUANT')" >&2; exit 2 ;;
esac
export STRATA_QUANT=$QUANT    # the lease re-runs the script: keep the choice

# STRATA_ENGINE picks the Strata version (T2-T6): 0145 (default, setup's install in 03_run) or v020 (Strata 0.2.0,
# experiments/08_v020: its own engine build and serve/tools, the same models, pack and MTP; EXPERIMENT_LOG.md §19).
# v020 results get a -v020 suffix: results/<day>/T2-speed-v020 ...
# ENGINE_EXE / SRC_GIT / TESTBUILD: the engine binary, the source checkout (T0) and the ctest build (T1) of that version.
ENGINE_VER=${STRATA_ENGINE:-0145}
ENGINE_EXE=$S/engine/strata; SRC_GIT=$TOP/sources/Strata; TESTBUILD=$T/build-tests; VTAG=
case $ENGINE_VER in
  0145) ;;
  v020) S=$TOP/experiments/08_v020/Strata; BASE_CFG=${BASE_CFG%.json}-v020.json; QTAG=$QTAG-v020; VTAG=-v020
        ENGINE_EXE=$TOP/experiments/08_v020/build/strata; SRC_GIT=$TOP/sources/Strata-0.2.0; TESTBUILD=$T/build-tests-v020 ;;
  *) echo "STRATA_ENGINE must be 0145 or v020 (got '$ENGINE_VER')" >&2; exit 2 ;;
esac
export STRATA_ENGINE=$ENGINE_VER
