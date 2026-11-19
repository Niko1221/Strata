#!/usr/bin/env bash
# Build the llama.cpp-Vulkan -> Strata protocol bridge.
#
# The adapter is compiled against the C API headers of a llama.cpp checkout and
# linked to the libs that were built from that SAME commit (an ABI mismatch is
# the classic way to break this). Defaults point at this machine's install;
# override with env vars on another PC.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LLAMA_CPP_SRC="${LLAMA_CPP_SRC:-/home/g/Code/llama-bench}"
LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-/home/g/Code/llama.cpp/build/bin}"

for f in "$LLAMA_CPP_SRC/include/llama.h" "$LLAMA_CPP_SRC/ggml/include/ggml-backend.h" \
         "$LLAMA_CPP_LIB/libllama.so" "$LLAMA_CPP_LIB/libggml.so" "$LLAMA_CPP_LIB/libggml-base.so"; do
    if [ ! -e "$f" ]; then echo "missing: $f (set LLAMA_CPP_SRC / LLAMA_CPP_LIB)" >&2; exit 1; fi
done

c++ -O2 -std=c++17 -Wall -Wextra \
    -I"$LLAMA_CPP_SRC/include" -I"$LLAMA_CPP_SRC/ggml/include" \
    "$HERE/llama_vulkan_engine.cpp" \
    -o "$HERE/strata-llama-vulkan-engine" \
    -L"$LLAMA_CPP_LIB" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$LLAMA_CPP_LIB"

echo "built $HERE/strata-llama-vulkan-engine"
echo "lib dir (put in config 'lib_dirs'): $LLAMA_CPP_LIB"
