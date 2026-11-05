#!/bin/bash
# Run a prompt through the whole stack and compare the next-token distribution with llama.cpp.
#
#   bash cann/tests/run_seq.sh [prompt]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CANN_DIR="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$CANN_DIR/.." && pwd)"

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"

cmake -S "$CANN_DIR" -B "$ROOT/build-cann" >/dev/null
cmake --build "$ROOT/build-cann" -j"$(nproc)" >/dev/null

MODEL=${MODEL:-/data/strata-qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
PROMPT=${1:-"The capital of France is"}

# tokenize through the running server, then walk the same ids through this stack
IDS=$(curl -s -X POST http://127.0.0.1:18789/tokenize \
    -H 'Content-Type: application/json' \
    -d "{\"content\": \"$PROMPT\"}" | python3 -c 'import json,sys; print(",".join(map(str, json.load(sys.stdin)["tokens"])))')
echo "prompt: $PROMPT"
echo "tokens: $IDS"
"$ROOT/build-cann/model_seq" "$MODEL" "$IDS"
