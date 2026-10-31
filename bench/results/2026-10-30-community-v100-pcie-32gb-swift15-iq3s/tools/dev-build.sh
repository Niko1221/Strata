#!/usr/bin/env bash
# TASK-20261029-03 (RES-20261029-01 G1.1): build the Strata engine for the V100 from <strata-dev> with setup.py's own
# configure line (setup.py cmake_build + the CUDA engine defs: Release, sm_70, STRATA_EXPERIMENTAL_SM60, Q6K experts,
# MMQ k-quants, g++-14 host compiler, vendored llama.cpp), through ccache on /ai.
# Usage: dev-build.sh rel    same flags as production (the speed reference for code changes)
#        dev-build.sh prof   plus -lineinfo (ncu/nsys source lines) and -g -fno-omit-frame-pointer (perf call graphs)
# Builds only; starts no engine. Waits for <stage>/model-run.lock and runs at nice 10.
set -eu
exec 9><stage>/model-run.lock; flock 9
kind=$1 S=<strata-dev>; B=$S/build-v100-$kind
export PATH=/usr/local/cuda-12.8/bin:$PATH CCACHE_DIR=<dev>/cache/ccache NVCC_CCBIN=g++-14 CUDAHOSTCXX=g++-14 TMPDIR=<tmp>
mkdir -p $TMPDIR                                   # nvcc's intermediate files on /ai too, not the OS disk's /tmp
case $kind in
  rel)  extra=() ;;
  prof) extra=(-DCMAKE_CUDA_FLAGS=-lineinfo "-DCMAKE_CXX_FLAGS=-g -fno-omit-frame-pointer" "-DCMAKE_C_FLAGS=-g -fno-omit-frame-pointer") ;;
  *) echo "usage: $0 rel|prof"; exit 1 ;;
esac
echo "# $(date -u +%T) configure $B ($(git -C $S rev-parse --short HEAD))"
cmake -G Ninja -DCMAKE_MAKE_PROGRAM=$S/.venv/bin/ninja -S $S -B $B -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DSTRATA_GGML_DIR=$S/third_party/llama.cpp -DCMAKE_CUDA_HOST_COMPILER=g++-14 -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  "${extra[@]}" > $B.configure.log 2>&1 || { tail -20 $B.configure.log; exit 2; }
t0=$(date +%s)
nice -n 10 cmake --build $B --target strata -j 40 > $B.build.log 2>&1 || { tail -30 $B.build.log; exit 3; }
echo "# $(date -u +%T) built in $(( $(date +%s) - t0 )) s: $(ls -la $B/strata | awk '{print $5}') bytes, sha256 $(sha256sum $B/strata | cut -c1-16), kq lib: $(ls $B/libstrata_kq.so 2>/dev/null || echo none)"
