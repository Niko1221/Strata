#!/bin/bash
# The engine's greedy continuation has to be the same tokens llama.cpp picks, not merely close:
# one wrong token makes the rest of the sentence diverge.
set -uo pipefail
MODEL="${1:-/data/strata-qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="$HERE/../../build-cann"

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"

ours=$("$BIN/model_seq" "$MODEL" 760,6511,314,9338,369 4 0.0 2>/dev/null | grep '^GEN ' | awk '{print $2}' | tr '\n' ' ' | xargs)
if [ "$ours" != "11751 11 264 3177" ]; then
    echo "greedy tokens differ: got '$ours', llama.cpp gives '11751 11 264 3177'"
    exit 1
fi
echo "greedy tokens 11751 11 264 3177 (' Paris, a city')"
