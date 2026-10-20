#!/usr/bin/env bash
# Build the Strata 0.2.0-beta engine (engine 0.1.45.3) for this host: Xeon E5-2630 v2 (AVX + F16C, no FMA / AVX2), RTX 3060 (sm_86).
#
# - STRATA_ISA_FLOOR=avx: upstream's experimental older-CPU build (docs/INSTALL.md "Older CPUs").
# - CUDA 13.3 from pip wheels in ../../toolchain/.venv (no sudo here; toolchain/setup_toolchain.sh). The system
#   CUDA 12.9 and CUDA 13.0 both clash with glibc 2.43; CUDA 13.x refuses gcc 16, so the host compiler is g++-14.
# - Out-of-tree: sources/Strata is not modified; ggml (llama.cpp at the pinned commit) is fetched by CMake.
# Usage: ./build.sh [configure|build]   (default: both)
set -euo pipefail
cd "$(dirname "$0")"
TOP=$(cd ../.. && pwd -P)
CUDA="$TOP/toolchain/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$TOP/toolchain/.venv/bin:$CUDA/bin:$PATH"
export CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14
step=${1:-all}
if [ "$step" = all ] || [ "$step" = configure ]; then
  cmake -S "$TOP/sources/Strata-0.2.0" -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_CUDA=ON \
    -DSTRATA_ISA_FLOOR=avx \
    -DSTRATA_BUILD_TESTS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
    -DCUDAToolkit_ROOT="$CUDA" \
    -DCMAKE_CUDA_HOST_COMPILER=g++-14
fi
if [ "$step" = all ] || [ "$step" = build ]; then
  nice -n 10 cmake --build build --target strata -j "${JOBS:-20}"
fi
