#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STEERING_SCALE="${STEERING_SCALE:-1.0}"
STEERING_VECTOR="${STEERING_VECTOR:-$HOME/workspace/ai_models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-refusal-projection.gguf}"
PORT="${STRATA_PORT:-8080}"
CONFIG="$(mktemp "${TMPDIR:-/tmp}/strata-steered.XXXXXX.json")"
trap 'rm -f -- "$CONFIG"' EXIT

"$ROOT/.venv/bin/python" - "$ROOT/strata-iq3_xxs.json" "$CONFIG" "$STEERING_SCALE" "$STEERING_VECTOR" <<'PY'
import json, math, pathlib, sys

source, destination, scale_text, vector_text = sys.argv[1:]
scale = float(scale_text)
if not math.isfinite(scale):
    raise SystemExit('STEERING_SCALE must be finite')
cfg = json.loads(pathlib.Path(source).read_text())
vector = pathlib.Path(vector_text).expanduser().resolve()
if not vector.is_file():
    raise SystemExit(f'steering GGUF not found: {vector}')
cfg['args'] += [
    '--control-vector-scaled', f'{vector}:{scale:g}',
    '--control-vector-layer-range', '4', '44',
    '--cvec-mode', 'project',
    '--cvec-dir', 'per-layer',
]
pathlib.Path(destination).write_text(json.dumps(cfg, indent=2))
PY

cd "$ROOT"
"$ROOT/.venv/bin/python" "$ROOT/serve/server.py" --engine strata --config "$CONFIG" --port "$PORT" --open
