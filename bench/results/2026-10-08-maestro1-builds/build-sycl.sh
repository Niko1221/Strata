#!/usr/bin/env bash
set -eo pipefail
cd "$HOME/workspace/strata-validation"
validation_root="$PWD"
oneapi_root="$validation_root/toolchains/intel-root/opt/intel/oneapi"
source "$oneapi_root/tbb/2023.1/env/vars.sh"
source "$oneapi_root/umf/1.1/env/vars.sh"
source "$oneapi_root/tcm/1.5/env/vars.sh"
source "$oneapi_root/compiler/2026.1/env/vars.sh"
source "$oneapi_root/mkl/2026.1/env/vars.sh"
source "$oneapi_root/dpl/2022.13/env/vars.sh"
set -u
export TMPDIR="$validation_root/tmp"
export PATH="$validation_root/toolchains/build-tools/bin:$PATH"
{
  date -u
  uname -a
  git -C strata rev-parse HEAD
  icpx --version
  cmake --version
  printf 'MKLROOT=%s\n' "$MKLROOT"
} > logs/sycl-metadata.txt
cmake -S strata/sycl -B build-sycl -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_C_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 \
  -DCMAKE_CXX_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 \
  -DSTRATA_SYCL_PARITY=ON \
  -DSTRATA_GGML_DIR="$validation_root/dependencies/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d" \
  > logs/sycl-configure.txt 2>&1
cmake --build build-sycl --parallel 3 > logs/sycl-build.txt 2>&1
printf 'SYCL full build passed\n' > logs/sycl-result.txt
