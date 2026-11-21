#!/bin/bash
# strata-hy4 (HIP) from source on Linux; no Linux HIP build script existed for the runner.
# ROCM_PATH (default /opt/rocm) and HY4_ARCH (default gfx1200) can be set; output: build-hy4/strata-hy4.
set -e
R=$(cd "$(dirname "$0")/../.." && pwd)
ROCM=${ROCM_PATH:-/opt/rocm}
ARCH=${HY4_ARCH:-gfx1200}
mkdir -p $R/build-hy4 && cd $R/build-hy4
E=$R/engine
RUN=$E/src/hy4flash/run
Z=$E/third_party/zstd-dec
INC="-I$E/include -I$RUN -I$Z -I$Z/common"
CF="-O3 -march=native -fopenmp -DNDEBUG"
objs=()
for c in $Z/common/*.c $Z/decompress/*.c; do o=zstd_$(basename $c .c).o; [ $o -nt $c ] || $ROCM/llvm/bin/clang -O3 -c $c $INC -DZSTD_DISABLE_ASM -o $o; objs+=($o); done
for c in $E/src/artifact/strata_reader.cpp $E/src/quant/sk_codecs.cpp $RUN/hy4_engram.cpp $RUN/hy4_cpu_experts.cpp; do
  o=$(basename $c .cpp).o; [ $o -nt $c ] || $ROCM/llvm/bin/clang++ -std=c++20 $CF $INC -c $c -o $o; objs+=($o); done
for c in $RUN/mla_attn.hip.cpp $RUN/hy4_main.hip.cpp; do
  o=$(basename $c .cpp).o
  $ROCM/bin/hipcc -std=c++20 $CF $INC --offload-arch=$ARCH -c $c -o $o; objs+=($o); done
$ROCM/bin/hipcc -fopenmp -o strata-hy4 "${objs[@]}" -lhipblas -lhipblaslt -lpthread
echo built
