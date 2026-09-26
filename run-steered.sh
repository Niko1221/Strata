#!/usr/bin/env bash
# run-steered.sh - the residual control-vector projection over ANY run config.
#
#   ./run-steered.sh [CONFIG.json]
#
# CONFIG is a ready-made engine config (exe, args, tokenizer, ...); with no
# argument the script uses the single strata-*.json in the repository root
# (several found: it names them and stops; none: it says so).  The config is
# not touched - its copy gains the four control-vector flags and the server
# starts on it.
#
#   STEERING_SCALE    the signed projection strength (default 1.0)
#   STEERING_VECTOR   the control-vector GGUF (default: Qwen3.8-Flash-Next-refusal-projection.gguf
#                     in the repository root)
#   STEERING_FIRST / STEERING_LAST   the inclusive layer range (default 4..44)
#   STRATA_PORT       the server port (default 8080)
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STEERING_SCALE="${STEERING_SCALE:-1.0}"
STEERING_VECTOR="${STEERING_VECTOR:-$ROOT/Qwen3.8-Flash-Next-refusal-projection.gguf}"
STEERING_FIRST="${STEERING_FIRST:-4}"
STEERING_LAST="${STEERING_LAST:-44}"
PORT="${STRATA_PORT:-8080}"

CONFIG="${1:-}"
if [ -z "$CONFIG" ]; then
    candidates=("$ROOT"/strata-*.json)
    if [ ! -e "${candidates[0]:-}" ]; then
        echo "no strata-*.json run config in $ROOT - pass one: $0 strata-<quant>.json" >&2
        exit 2
    elif [ "${#candidates[@]}" -gt 1 ]; then
        echo "several run configs in $ROOT - pass one:" >&2
        printf '  %s\n' "${candidates[@]##*/}" >&2
        exit 2
    fi
    CONFIG="${candidates[0]}"
fi
CONFIG="$(cd -- "$(dirname -- "$CONFIG")" && pwd)/$(basename -- "$CONFIG")"
if [ ! -f "$CONFIG" ]; then
    echo "run config not found: $CONFIG" >&2
    exit 2
fi

OUT="$(mktemp "${TMPDIR:-/tmp}/strata-steered.XXXXXX.json")"
trap 'rm -f -- "$OUT"' EXIT

"$ROOT/.venv/bin/python" - "$CONFIG" "$OUT" "$STEERING_SCALE" "$STEERING_VECTOR" \
    "$STEERING_FIRST" "$STEERING_LAST" <<'PY'
import json, math, pathlib, sys

source, destination, scale_text, vector_text, first_text, last_text = sys.argv[1:]
scale = float(scale_text)
if not math.isfinite(scale):
    raise SystemExit('STEERING_SCALE must be finite')
first, last = int(first_text), int(last_text)
if first < 0 or last < first:
    raise SystemExit('STEERING_FIRST/STEERING_LAST: expected 0 <= FIRST <= LAST')
cfg = json.loads(pathlib.Path(source).read_text())
vector = pathlib.Path(vector_text).expanduser().resolve()
if not vector.is_file():
    raise SystemExit(f'steering GGUF not found: {vector}')
cfg['args'] += [
    '--control-vector-scaled', f'{vector}:{scale:g}',
    '--control-vector-layer-range', str(first), str(last),
    '--cvec-mode', 'project',
    '--cvec-dir', 'per-layer',
]
pathlib.Path(destination).write_text(json.dumps(cfg, indent=2))
PY

cd "$ROOT"
"$ROOT/.venv/bin/python" "$ROOT/serve/server.py" --engine strata --config "$OUT" --port "$PORT" --open
