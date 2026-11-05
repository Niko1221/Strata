#!/bin/bash
# Build the CANN layer and run the whole model for one token, NPU against a CPU reference.
#
#   bash cann/tests/run_model.sh [model.gguf] [layers] [token id] [cpu check 0|1]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CANN_DIR="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$CANN_DIR/.." && pwd)"

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"

cmake -S "$CANN_DIR" -B "$ROOT/build-cann" >/dev/null
cmake --build "$ROOT/build-cann" -j"$(nproc)" >/dev/null

ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then
  ARGS=(/data/strata-qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf 40 9707 1)
fi
"$ROOT/build-cann/model_forward" "${ARGS[@]}"
