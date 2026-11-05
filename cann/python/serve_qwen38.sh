#!/bin/bash
# Serve Qwen3.8-Flash-Next on the two 910B2C cards.
#
#   bash serve_qwen38.sh            # 8080, 32k context, both cards
#   CTX=131072 bash serve_qwen38.sh
#
# The n-gram table is 28.8 GB and only ever read a row at a time, so it stays on disk:
# -lm mmap keeps the file memory-mapped and -lzm on reads the big tensors on demand.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/data/models/IQ3_XXS}"
CTX="${CTX:-32768}"
PORT="${PORT:-8080}"
BIN="${BIN:-/data/build-cann/bin/llama-server}"

export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export LD_LIBRARY_PATH="/data/build-cann/bin:/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"

SHARD1=$(find "$MODEL_DIR" -name '*-00001-of-00002.gguf' | head -1)
# vision is opt-in: the projector is 907 MB and we only test text, so a load failure there
# would cost the whole run for nothing
MMPROJ=""
if [ "${VISION:-off}" = "on" ]; then
    MMPROJ=$(find "$MODEL_DIR" -name 'mmproj-*.gguf' | head -1)
fi
[ -n "$SHARD1" ] || { echo "no first shard under $MODEL_DIR"; exit 1; }

echo "model : $SHARD1"
echo "mmproj: ${MMPROJ:-none (VISION=on to load it)}"
echo "ctx   : $CTX   port: $PORT"

ARGS=(
  -m "$SHARD1"
  -lm mmap -lzm on
  -ngl 99
  --split-mode layer
  -c "$CTX"
  --host 0.0.0.0 --port "$PORT"
  --jinja
  -t "$(nproc)"
)
[ -n "$MMPROJ" ] && ARGS+=(--mmproj "$MMPROJ")

exec "$BIN" "${ARGS[@]}"
