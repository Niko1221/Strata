#!/usr/bin/env bash
# Build the native Vulkan engine: our full forward with streamed MoE experts.
# No llama.cpp / oneAPI / Level Zero needed — only Vulkan and C++17.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VULKAN="$(cd "$HERE/.." && pwd)"

URING_FLAGS=""
if pkg-config --exists liburing 2>/dev/null; then
    URING_FLAGS="-DSTRATA_HAVE_LIBURING -luring"
fi

c++ -O2 -std=c++17 -Wall -Wextra $URING_FLAGS \
    -I"$VULKAN/include" -I"$VULKAN/native" \
    "$VULKAN/native/model.cpp" \
    "$VULKAN/src/expert_stream.cpp" "$VULKAN/src/router.cpp" \
    "$VULKAN/src/moe_combine.cpp" "$VULKAN/src/bf16_gemv.cpp" "$VULKAN/src/sampler.cpp" \
    "$HERE/vulkan_engine.cpp" \
    -o "$HERE/strata-vulkan-engine" \
    -lvulkan -lpthread

echo "built $HERE/strata-vulkan-engine (native Vulkan forward, no llama.cpp)"
