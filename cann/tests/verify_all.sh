#!/bin/bash
# Run every check in this directory and print one summary.
#
#   bash cann/tests/verify_all.sh [model.gguf]
#
# Each line is a real run against the model file on the NPU; nothing here is a fixture.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CANN_DIR="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$CANN_DIR/.." && pwd)"
BIN="$ROOT/build-cann"
MODEL="${1:-/data/strata-qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"

cmake -S "$CANN_DIR" -B "$ROOT/build-cann" >/dev/null || exit 1
cmake --build "$ROOT/build-cann" -j"$(nproc)" >/dev/null || exit 1

pass=0
fail=0
run() {
    local name="$1"
    shift
    local out
    if out=$("$@" 2>&1); then
        printf '  ok    %s\n' "$name"
        pass=$((pass + 1))
    else
        printf '  FAIL  %s\n' "$name"
        printf '%s\n' "$out" | tail -5 | sed 's/^/        /'
        fail=$((fail + 1))
    fi
}

echo "machine: $(npu-smi info -t board -i 0 2>/dev/null | grep -i 'chip name' | head -1 | tr -s ' ' | cut -d: -f2 | xargs) ${ASCEND_HOME_PATH##*/}"
echo "model  : $MODEL"
echo
echo "operators and blocks:"
run "aclnn ops against the CPU oracle" "$BIN/cann_parity"
run "quantized decoders against ggml" "$BIN/quant_parity" "$MODEL"
run "batched primitives" "$BIN/batched_parity"
run "gated delta net op" "$BIN/gdn_parity"
run "batched gated delta net" "$BIN/gdn_batched_parity" 4
run "one real layer from real weights" "$BIN/layer_parity" "$MODEL"
run "the MoE FFN of blk.0" "$BIN/moe_parity" "$MODEL"
run "the GDN block of blk.0" "$BIN/gdn_block_parity" "$MODEL"
run "GDN state across tokens" "$BIN/gdn_state_parity" "$MODEL" 4
run "rope" "$BIN/rope_parity"
run "the attention layer of blk.3" "$BIN/attn_layer_parity" "$MODEL"
run "attention with a KV cache" "$BIN/attn_cache_parity" "$MODEL" 4
run "the whole model, one token, NPU vs CPU" "$BIN/model_forward" "$MODEL" 40 64 1

echo
echo "against llama.cpp (needs a running server on 18789):"
if curl -sf http://127.0.0.1:18789/v1/models >/dev/null 2>&1; then
    python3 "$CANN_DIR/tests/compare_with_llama.py" "$MODEL" | sed 's/^/  /'
else
    echo "  skipped: no server on 18789"
fi

echo
echo "summary: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
