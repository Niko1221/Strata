#!/bin/bash
# Build the CANN layer and run the full-attention layer of blk.3 for its first token, plus rope.
#
#   bash cann/tests/run_attn.sh [model.gguf]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CANN_DIR="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$CANN_DIR/.." && pwd)"

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"

cmake -S "$CANN_DIR" -B "$ROOT/build-cann" >/dev/null
cmake --build "$ROOT/build-cann" -j"$(nproc)" >/dev/null
"$ROOT/build-cann/rope_parity"
"$ROOT/build-cann/attn_layer_parity" "$@"
