#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/workspace/strata-validation"
validation_root="$PWD"
export TMPDIR="$validation_root/tmp"
export PATH="$validation_root/toolchains/build-tools/bin:$PATH"
rocm_root=$(toolchains/rocm/bin/rocm-sdk path --root)
rocm_libraries="$validation_root/toolchains/rocm/lib/python3.10/site-packages/_rocm_sdk_libraries_gfx110X_dgpu"
export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr
export ROCM_PATH="$rocm_root" HIP_PATH="$rocm_root"
export PATH="$rocm_root/bin:$rocm_root/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$rocm_root/lib:$rocm_libraries/lib:${LD_LIBRARY_PATH:-}"
{
  date -u
  uname -a
  git -C strata rev-parse HEAD
  "$rocm_root/llvm/bin/clang++" --version
  cmake --version
  lscpu
} > logs/hip-metadata.txt
cmake -S strata -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_PREFILL_MMQ=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER="$rocm_root/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$rocm_root" \
  -DCMAKE_PREFIX_PATH="$rocm_root;$rocm_libraries" \
  -DCMAKE_HIP_FLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 --rocm-path=$rocm_root --rocm-device-lib-path=$rocm_root/lib/llvm/amdgcn/bitcode" \
  -DSTRATA_GGML_DIR="$validation_root/dependencies/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d" \
  > logs/hip-configure.txt 2>&1
cmake --build build-hip --parallel 2 > logs/hip-build.txt 2>&1
printf 'HIP full build passed\n' > logs/hip-result.txt
