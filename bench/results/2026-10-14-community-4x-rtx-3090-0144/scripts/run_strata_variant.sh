#!/usr/bin/env bash
# Start Strata with one config variant in its own memory-capped systemd scope, wait until it serves, run the
# concurrency sweep against it, stop it.
#   run_strata_variant.sh <label> <config.json> <levels> [extra bench args...]
# Env: STRATA (required), SERVER_DIR, PORT, BIND, MEMMAX; KEEP=1 leaves the server running afterwards; ATTACH=1 starts nothing and
# benches the server already running on PORT (ab_quick's second prompt kind on the same server).
# Writes into results/: <label>.txt (run line + bench table + GPU line), sweep.jsonl and sweep.requests.jsonl
# (bench_concurrency.py: one row per level / every request's uid, timestamps and reply sha256), gpu-<label>.csv
# (nvidia-smi every 1 s during the bench), config-<label>.json, status-<label>.json, server-<label>.out and
# engine-<label>.run.log = the engine log of THIS engine start only (from its last 'engine started' line on; the
# config's log file is shared by all configs and cumulative - the old engine-<label>.log copies were whole files).
set -uo pipefail
LABEL=$1; CFG=$(readlink -f "$2"); LEVELS=$3; shift 3
HERE="$(cd "$(dirname "$0")" && pwd)"
RES="$HERE/results"
STRATA=${STRATA:?set STRATA to a Strata folder with its .venv}
SERVER_DIR=${SERVER_DIR:-$STRATA}   # the checkout whose serve/server.py runs (same venv)
PORT=${PORT:-8090}
# BIND: listen address, 127.0.0.1 unless set (never expose the server without --api-key)
LOG="$RES/server-$LABEL.out"
mkdir -p "$RES"
if [ "${ATTACH:-0}" != "1" ]; then
  systemctl --user stop strata-bench.scope 2>/dev/null
  systemctl --user reset-failed strata-bench.scope 2>/dev/null
  cd "$SERVER_DIR" || exit 1
  systemd-run --user --scope --unit=strata-bench -p MemoryMax="${MEMMAX:-200G}" -p MemorySwapMax=0 \
    "$STRATA/.venv/bin/python" "$SERVER_DIR/serve/server.py" --engine strata --config "$CFG" --port "$PORT" --host "${BIND:-127.0.0.1}" \
    > "$LOG" 2>&1 < /dev/null &
  sleep 5
fi
t0=$(date +%s)
until curl -s -m 3 "http://127.0.0.1:$PORT/health" | grep -q '"loaded": *true'; do
  if ! systemctl --user is-active -q strata-bench.scope; then echo "[$LABEL] server died during start"; tail -30 "$LOG"; exit 1; fi
  if [ $(( $(date +%s) - t0 )) -gt 1200 ]; then echo "[$LABEL] start timeout"; exit 1; fi
  sleep 5
done
[ "${ATTACH:-0}" = "1" ] || echo "[$LABEL] ready after $(( $(date +%s) - t0 )) s"
curl -s "http://127.0.0.1:$PORT/v1/status" > "$RES/status-$LABEL.json"
cp "$CFG" "$RES/config-$LABEL.json"
EXE=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['exe'])" "$CFG" 2>/dev/null)
ENGLOG=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['log'])" "$CFG" 2>/dev/null)
# run line: what was measured (the exe hash pins the binary, the server checkout's HEAD the HTTP layer)
echo "# run $(date -Iseconds) config $(basename "$CFG") exe $(basename "${EXE:-?}") sha256 $(sha256sum "$EXE" 2>/dev/null | cut -c1-16)" \
     "server $(git -C "$SERVER_DIR" rev-parse --short HEAD 2>/dev/null)$(git -C "$SERVER_DIR" diff --quiet HEAD 2>/dev/null || echo +dirty)" \
     > "$RES/$LABEL.txt"

# GPU clocks / power / temperature during the bench (the 3090s are power-limited, so the boost clock drifts with
# temperature); the timestamp column lines the samples up with the sweep row's ts and the request timestamps
GPUCSV="$RES/gpu-$LABEL.csv"
nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv,nounits -l 1 \
  > "$GPUCSV" 2>/dev/null &
SMI=$!
"$STRATA/.venv/bin/python" -I "$HERE/bench_concurrency.py" --url "http://127.0.0.1:$PORT" --label "$LABEL" \
  --levels "$LEVELS" --out "$RES/sweep.jsonl" "$@" 2>&1 | tee -a "$RES/$LABEL.txt"
kill "$SMI" 2>/dev/null; wait "$SMI" 2>/dev/null
# one line, per GPU: mean SM clock, mean power, mean (max) temperature, mean utilisation over the bench
python3 - "$GPUCSV" <<'EOF' | tee -a "$RES/$LABEL.txt"
import csv, sys
g = {}
try:
    rows = list(csv.reader(open(sys.argv[1])))[1:]
except OSError:
    rows = []
for r in rows:
    try:
        i, sm, w, t, u = int(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])
    except (ValueError, IndexError):
        continue
    g.setdefault(i, []).append((sm, w, t, u))
if not g:
    print("# gpu: no nvidia-smi samples")
else:
    m = lambda xs: sum(xs) / len(xs)
    print(f"# gpu means over {min(len(v) for v in g.values())} samples (1 s):", " | ".join(
        f"GPU{i} {m([s[0] for s in v]):.0f} MHz {m([s[1] for s in v]):.0f} W {m([s[2] for s in v]):.0f} C "
        f"(max {max(s[2] for s in v):.0f}) {m([s[3] for s in v]):.0f} %" for i, v in sorted(g.items())))
EOF
# the engine log of this engine start only (from the last 'engine started' marker on)
if [ -n "$ENGLOG" ] && [ -f "$ENGLOG" ]; then
  n=$(grep -n 'engine started' "$ENGLOG" | tail -1 | cut -d: -f1)
  tail -n +"${n:-1}" "$ENGLOG" > "$RES/engine-$LABEL.run.log"
fi
[ "${KEEP:-0}" = "1" ] || systemctl --user stop strata-bench.scope
