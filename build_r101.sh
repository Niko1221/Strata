#!/bin/bash
# THE PLATINUM BUILD recipe: rp-graft (v0.2.14.1 + MMQ + rp prompt-path stack)
# on mixa3607/rocm-gfx906:10.1-complete (ROCm 10.1 core, bundled clang 24).
# Output: $SRC/build-r101/strata — must match the sha256 in the platinum-r101 release.
# Usage: SRC=/path/to/rp-graft-checkout bash build_r101.sh   (SRC defaults to $PWD)
set -euo pipefail
SRC=${SRC:-$PWD}
BUILD=build-r101
LOG="$SRC/$BUILD.log"
mkdir -p "$SRC/$BUILD"
exec >>"$LOG" 2>&1
echo "=== build start $(date)"
docker run --rm \
  -v "$SRC:$SRC" \
  -w "$SRC" \
  -e ROCM_PATH=/opt/rocm -e HIP_PATH=/opt/rocm \
  -e LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib/rocm_sysdeps/lib \
  mixa3607/rocm-gfx906:10.1-complete bash -c '
    set -euo pipefail
    apt-get update -qq && apt-get install -y -qq cmake ninja-build build-essential python3-dev libcurl4-openssl-dev >/dev/null
    /opt/rocm/llvm/bin/clang++ --version | head -1
    cmake -S . -B build-r101 \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_HIP_GFX906=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx906 \
      -DSTRATA_PORTABLE=OFF \
      -DSTRATA_MMQ_KQUANTS=ON \
      -DSTRATA_MMQ=ON \
      -DSTRATA_PREFILL_MMQ=OFF \
      -DSTRATA_BUILD_TESTS=OFF \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
      -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_PREFIX_PATH=/opt/rocm
    cmake --build build-r101 -j "$(nproc)" --target strata
  '
sha256sum "$SRC/$BUILD/strata"
echo "=== build done $(date)"
