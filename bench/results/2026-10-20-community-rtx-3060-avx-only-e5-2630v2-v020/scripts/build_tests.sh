#!/usr/bin/env bash
# Build Strata's test suite (ctest) for this host, from the same source the setup copy's engine was built from
# (experiments/03_run/Strata), into a separate directory: the engine in use (03_run/Strata/engine/strata) is not touched.
# Same flags as setup's build (STRATA_ISA_FLOOR=avx, sm_86, CUDA 13.3 wheels, g++-14) plus STRATA_BUILD_TESTS=ON.
set -euo pipefail
cd "$(dirname "$0")"
TOP=<WORKDIR>
SRC=$TOP/experiments/03_run/Strata; OUT=build-tests; GGML=$SRC/third_party/llama.cpp
# STRATA_ENGINE=v020: Strata 0.2.0 (experiments/08_v020/Strata) into build-tests-v020, ggml from the engine build's
# checkout of the same pinned llama.cpp commit (3cf03257)
if [ "${STRATA_ENGINE:-0145}" = v020 ]; then
  SRC=$TOP/experiments/08_v020/Strata; OUT=build-tests-v020; GGML=$TOP/experiments/08_v020/build/_deps/strata_llamacpp-src
fi
CUDA="$TOP/toolchain/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$TOP/toolchain/.venv/bin:$CUDA/bin:$PATH"
export CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14
cmake -S "$SRC" -B "$OUT" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ISA_FLOOR=avx -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA" \
  -DCMAKE_CUDA_HOST_COMPILER=g++-14 -DSTRATA_GGML_DIR="$GGML"
# -k 0: keep going past targets whose sources the published tree leaves out
nice -n 10 cmake --build "$OUT" -j "${JOBS:-20}" -- -k 0
