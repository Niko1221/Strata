#!/bin/bash
# Baseline measurement for Qwen3.8 on the two NPUs.
#
#   bash measure_qwen38.sh [ctx] [n_predict]
#
# Prints: load time, prefill and decode rates, per-card HBM, and every fused op that llama.cpp
# had to move off the device (those lines appear as warnings at default verbosity).
set -uo pipefail
CTX="${1:-32768}"
NGEN="${2:-64}"
MODEL_DIR="${MODEL_DIR:-/data/models/IQ3_XXS}"

export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export LD_LIBRARY_PATH="/data/build-cann/bin:/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"

SHARD=$(find "$MODEL_DIR" -name '*-00001-of-00002.gguf' ! -name '*.incomplete' | head -1)
[ -n "$SHARD" ] || { echo "no complete first shard under $MODEL_DIR"; exit 1; }

# a prompt long enough that the prefill number means something
PROMPT=$(python3 - <<'PY'
print("The quick brown fox jumps over the lazy dog. " * 40)
PY
)

echo "model : $SHARD"
echo "ctx   : $CTX   new tokens: $NGEN"
echo

LOG=/data/logs/measure_$(date +%H%M%S).log
start=$(date +%s)

/data/build-cann/bin/llama-cli \
    -m "$SHARD" -lm mmap -lzm on -ngl 99 --split-mode layer \
    -c "$CTX" -n "$NGEN" --seed 1 -t "$(nproc)" \
    -p "$PROMPT" < /dev/null > "$LOG" 2>&1
rc=$?
end=$(date +%s)
echo "llama-cli rc=$rc  wall=$((end - start)) s  log=$LOG"
echo

echo "== load =="
grep -E 'load time|model size|offloaded|BUFFER|graph splits' "$LOG" | head -12

echo
echo "== throughput =="
grep -E 'Prompt:|Generation:|eval time' "$LOG" | tail -4

echo
echo "== fused ops that fell off the device =="
if grep -q 'assigned to device' "$LOG"; then
    grep 'assigned to device' "$LOG" | sed 's/^/  /' | sort | uniq -c | sort -rn | head -12
else
    echo "  (none reported - every fused op stayed where its layer lives)"
fi

echo
echo "== HBM in use =="
npu-smi info 2>/dev/null | grep -oE '[0-9]+ / 65536' | head -4

echo
echo "== sample output =="
sed -n '/^>/,$p' "$LOG" | head -12
