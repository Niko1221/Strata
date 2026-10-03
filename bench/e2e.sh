#!/bin/sh
# e2e.sh - one end-to-end decode measurement, for the sm_86 (RTX A3000) tuning work.
#
#   ./bench/e2e.sh CONFIG.json [MAX_NEW] [PORT]
#
# Starts serve/server.py with CONFIG (its own engine + args), sends one request and
# prints the request's decode tok/s from /metrics.  The default prompt is fresh prose
# on purpose: chat transcripts swing 1.5-1.7x with how much the model echoes back, so
# a fixed, non-repetitive prompt is the only way two runs compare.
#
# Needs the Python environment the server runs in (numpy, jinja2, ...): point PY at
# it, e.g. PY=~/.local/share/strata/.venv/bin/python.  In the nix packaging the engine
# also needs the libcuda path (packages/strata's lib-driver dir) - run it from the
# `strata-server` wrapper's environment, or set LD_LIBRARY_PATH yourself.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CFG=${1:?usage: e2e.sh CONFIG.json [MAX_NEW] [PORT]}
MAX_NEW=${2:-200}
PORT=${3:-8097}
PY=${PY:-python3}
SERVE=${SERVE:-$HERE/../serve/server.py}
LOG=${LOG:-/tmp/e2e-$$.log}

setsid "$PY" "$SERVE" --engine strata --config "$CFG" --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
cleanup() { kill -- "-$SRV" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

i=0
while [ "$i" -lt 90 ]; do
  grep -q "ready:" "$LOG" 2>/dev/null && break
  kill -0 "$SRV" 2>/dev/null || { echo "server exited; see $LOG"; tail -3 "$LOG"; exit 1; }
  sleep 2; i=$((i + 1))
done

curl -sS -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"strata\",\"messages\":[{\"role\":\"user\",\"content\":\"Write three sentences about the weather in Brussels today.\"}],\"max_tokens\":$MAX_NEW,\"reasoning_effort\":\"none\"}" \
  >/dev/null

curl -sS -m 10 "http://127.0.0.1:$PORT/metrics" | CFG="$CFG" "$PY" -c "
import json, os, sys
h = json.load(sys.stdin)['requests']
r = h[0] if h else {}
print(json.dumps({'decode_tok_s': r.get('decode_tok_s'), 'output_tokens': r.get('output_tokens'),
                  'prompt_tokens': r.get('prompt_tokens'), 'finish': r.get('finish'),
                  'engine': json.load(open(os.environ['CFG'])).get('exe')}))
"
cleanup
trap - EXIT INT TERM
