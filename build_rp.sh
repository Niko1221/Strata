#!/bin/bash
# Build upstream Strata 0.2.14.1 (baf16363b) with stable clang 23 on 7.2.4 base -> build-c23
set -euo pipefail
SRC=/home/rpcribari/rp-graft
BUILD=build-rp
LOG="$SRC/$BUILD.log"
mkdir -p "$SRC/$BUILD"
exec >>"$LOG" 2>&1
echo "=== rp-graft c23 build start $(date)"
docker run --rm \
  -v /home/rpcribari:/home/rpcribari \
  -w /home/rpcribari/rp-graft \
  -e ROCM_PATH=/opt/rocm -e HIP_PATH=/opt/rocm \
  strata-runtime:latest bash -c '
    set -euo pipefail
    apt-get update -qq && apt-get install -y -qq --no-install-recommends wget ca-certificates ninja-build cmake >/dev/null
    mkdir -p /etc/apt/keyrings && wget -qO /etc/apt/keyrings/llvm.asc https://apt.llvm.org/llvm-snapshot.gpg.key
    echo "deb [signed-by=/etc/apt/keyrings/llvm.asc] http://apt.llvm.org/noble/ llvm-toolchain-noble-23 main" > /etc/apt/sources.list.d/llvm23.list
    apt-get update -qq && apt-get install -y -qq --no-install-recommends clang-23 clang-tools-23 lld-23 llvm-23 libclang-rt-23-dev libclang-common-23-dev >/dev/null
    cp /opt/rocm-7.2.4/lib/llvm/lib/clang/22/include/__clang_*.h /usr/lib/llvm-23/lib/clang/23/include/
    cp /home/rpcribari/c23-tools/llvm-offload-binary /home/rpcribari/c23-tools/libLLVM.so.24.0git /usr/lib/llvm-23/bin/
    ln -sf /usr/lib/llvm-23/bin/lld /usr/bin/ld.lld
    ln -sf /usr/lib/llvm-23/bin/lld /usr/bin/lld
    export PATH=/usr/lib/llvm-23/bin:/usr/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    export LD_LIBRARY_PATH=/usr/lib/llvm-23/bin:/home/rpcribari/c23-tools/sysdeps
    clang++-23 --version | head -1
    cmake -S . -B build-rp \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_HIP_GFX906=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx906 \
      -DSTRATA_PORTABLE=OFF \
      -DSTRATA_MMQ_KQUANTS=ON \
      -DSTRATA_MMQ=ON \
      -DSTRATA_PREFILL_MMQ=OFF \
      -DSTRATA_BUILD_TESTS=OFF \
      -DCMAKE_HIP_COMPILER=/usr/bin/clang++-23 \
      -DCMAKE_C_COMPILER=clang-23 \
      -DCMAKE_CXX_COMPILER=clang++-23 \
      -DCMAKE_PREFIX_PATH=/opt/rocm
    cmake --build build-rp -j 64 --target strata
  '
sha256sum "/home/rpcribari/rp-graft/build-rp/strata"
echo "=== rp-graft c23 build done $(date)"
