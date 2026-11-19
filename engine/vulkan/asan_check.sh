#!/usr/bin/env bash
# AddressSanitizer check for the native Vulkan tier.
#
# A stale binary in /tmp/opencode/router-asan showed a heap-buffer-overflow in
# read_spv (brace-initialised std::vector<char> instead of parentheses), which
# was fixed in the sources. This script rebuilds every Vulkan test with ASAN so
# the fix and the newer kernels stay verified.
set -euo pipefail
set -o pipefail

VULKAN="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$VULKAN/../.." && pwd)"
OUT="${ASAN_OUT:-/tmp/opencode/strata-vulkan-asan}"
MODEL="${MODEL:-/home/g/Code/llama.cpp/models/Qwen3.8-35B-A3B-Q4_K_M.gguf}"
mkdir -p "$OUT"

INC="-I$VULKAN/include"
LIB="-lvulkan -lpthread"
CFLAGS="-O1 -g -fsanitize=address -std=c++17"

build() {
    local name="$1"; shift
    echo "building $name ..."
    c++ $CFLAGS $INC "$@" -o "$OUT/$name" $LIB
}

SRCS="$VULKAN/src"
build asan-router "$SRCS/router.cpp" "$SRCS/router_test.cpp"
build asan-combine "$SRCS/moe_combine.cpp" "$SRCS/moe_combine_test.cpp"
build asan-bf16 "$SRCS/bf16_gemv.cpp" "$SRCS/bf16_gemv_test.cpp"
build asan-sampler "$SRCS/sampler.cpp" "$SRCS/sampler_test.cpp"
build asan-q8 "$SRCS/expert_stream.cpp" "$SRCS/q8_0_gemv_test.cpp"
build asan-iq "$SRCS/expert_stream.cpp" "$SRCS/iq_gemv_test.cpp"
build asan-stream "$SRCS/expert_stream.cpp" "$SRCS/router.cpp" \
      "$SRCS/moe_combine.cpp" "$SRCS/bf16_gemv.cpp" "$SRCS/sampler.cpp" \
      "$SRCS/expert_stream_test.cpp"

run() { echo "=== $1 ==="; "$OUT/$1" "${@:2}" 2>&1 | tail -4; }

run asan-router
run asan-combine
# The bf16 test loads a 13 MB shader fixture; run it from the repo root for
# the default relative shader path.
(cd "$ROOT" && run asan-bf16)
(cd "$ROOT" && run asan-sampler)
(cd "$ROOT" && run asan-q8 engine/vulkan/shaders/expert_q8_0_gemv.comp.spv)
(cd "$ROOT" && run asan-iq \
  engine/vulkan/shaders/expert_iq3_xxs_gemv.comp.spv \
  engine/vulkan/shaders/expert_iq2_xs_gemv.comp.spv \
  engine/vulkan/shaders/expert_iq1_m_gemv.comp.spv)

if [ -f "$MODEL" ]; then
    (cd "$ROOT" && run asan-stream \
      --model "$MODEL" \
      --shader engine/vulkan/shaders/expert_q4_k_gemv.comp.spv \
      --down-shader engine/vulkan/shaders/expert_q6_k_gemv.comp.spv \
      --q8-shader engine/vulkan/shaders/expert_q8_0_gemv.comp.spv \
      --silu-shader engine/vulkan/shaders/expert_silu_mul.comp.spv \
      --router-gemv-shader engine/vulkan/shaders/router_gemv.comp.spv \
      --router-topk-shader engine/vulkan/shaders/router_topk.comp.spv \
      --experts 32 --slots 64 --threads 8 --rows 64 --forward 8)
else
    echo "=== asan-stream: SKIP (model not found: $MODEL) ==="
fi
echo "ASAN check complete"
