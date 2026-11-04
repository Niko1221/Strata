#!/bin/bash
# Build the P1+P2 CANN layer and check the Q4_K/Q6_K decoders against ggml's own.
#
#   STRATA_CANN_GGML_DIR=/path/to/llama.cpp bash cann/tests/run_quant.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CANN_DIR="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$CANN_DIR/.." && pwd)"

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"

GGML_DIR="${STRATA_CANN_GGML_DIR:-}"
GGML_LIB="${STRATA_CANN_GGML_LIB_DIR:-$GGML_DIR/build/bin}"

args=()
if [ -n "$GGML_DIR" ]; then
  args+=("-DSTRATA_CANN_GGML_DIR=$GGML_DIR" "-DSTRATA_CANN_GGML_LIB_DIR=$GGML_LIB")
  export LD_LIBRARY_PATH="$GGML_LIB:$LD_LIBRARY_PATH"
else
  echo "warning: STRATA_CANN_GGML_DIR is unset; quant_parity will run without the ggml oracle" >&2
fi

cmake -S "$CANN_DIR" -B "$ROOT/build-cann" "${args[@]}" >/dev/null
cmake --build "$ROOT/build-cann" -j"$(nproc)" >/dev/null
"$ROOT/build-cann/quant_parity"
