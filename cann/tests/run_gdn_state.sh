#!/bin/bash
# Build the CANN layer and run the GDN block across tokens with carried state.
#
#   bash cann/tests/run_gdn_state.sh [model.gguf] [tokens]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CANN_DIR="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$CANN_DIR/.." && pwd)"

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"

cmake -S "$CANN_DIR" -B "$ROOT/build-cann" >/dev/null
cmake --build "$ROOT/build-cann" -j"$(nproc)" >/dev/null
"$ROOT/build-cann/gdn_state_parity" "$@"
