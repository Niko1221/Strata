#!/bin/bash
# Builds tests/dsv41/dq_dense_test.cpp by hand (no CMake, no ggml): nvcc (CUDA) or the ROCm clang (HIP).
#   build_dq_test.sh cuda <outdir>      env: CUDA_ARCH=86 (sm_ number; 120 / 121 etc.), NVCC=nvcc, EXTRA="gguf gemm", JOBS=3
#   build_dq_test.sh hip  <outdir>      env: HIP_ROOT=<_rocm_sdk_devel>, HIP_ARCH=gfx1201, EXTRA="gguf gemm", HIPBLAS_LIB=<dir with libhipblas.so>
# SRC (the tree) defaults to the repo this script is in.  EXTRA: "gguf" adds ds4_gguf.cu (gg::rows / gg::mv), "gemm" adds the
# chunk-GEMM part (cuBLAS / hipBLAS, and on HIP the engine's WMMA kernel src/prefill/wmma_gemm.cu); both are off by default.
set -u
MODE=${1:?cuda|hip}; OUT=${2:?outdir}
SRC=${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}
EXTRA=${EXTRA:-}
JOBS=${JOBS:-3}
mkdir -p "$OUT"
DEFS="-DSTRATA_Q6K_EXPERTS=1"
FILES="src/dsv41/ds4_iq.cu src/dsv41/ds4_tq.cu"
case " $EXTRA " in *" gguf "*) DEFS="$DEFS -DDQ_TEST_GGUF=1"; FILES="$FILES src/dsv41/ds4_gguf.cu";; esac
case " $EXTRA " in *" gemm "*) DEFS="$DEFS -DDQ_TEST_GEMM=1"; case " $EXTRA " in *" gguf "*) ;; *) FILES="$FILES src/dsv41/ds4_gguf.cu";; esac;; esac
INC="-I$SRC/include -I$SRC/third_party/ggml -I$SRC/src/dsv41 -I$SRC/src/prefill"
: > "$OUT/build.status"

run_jobs() {   # run_jobs "<cmd>" ... with at most $JOBS at once
    local pids=() cmd
    for cmd in "$@"; do
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
        ( eval "$cmd" ) &
    done
    wait
}

if [ "$MODE" = cuda ]; then
    A=${CUDA_ARCH:?CUDA_ARCH, e.g. 86}
    NVCC=${NVCC:-nvcc}
    FL="-O2 -std=c++20 --generate-code=arch=compute_$A,code=sm_$A -Xcompiler -ffp-contract=off -Xcompiler -fPIC $DEFS $INC"
    if [ "$A" -lt 70 ]; then FL="$FL -include $SRC/tests/dsv41/dq_pascal_shim.h"; fi   # Pascal: no __dp4a (< 61), no wmma (< 70)
    cmds=()
    n=0
    for f in tests/dsv41/dq_dense_test.cpp $FILES; do
        o="$OUT/$(basename "$f").o"
        [ -f "$o" ] && [ "$o" -nt "$SRC/$f" ] && continue   # (up to date)
        x=""; case "$f" in *.cpp) x="-x cu";; esac
        cmds+=("cd $SRC && $NVCC $FL $x -c $f -o $o > $o.log 2>&1 || echo FAILED $f >> $OUT/build.status")
    done
    run_jobs "${cmds[@]}"
    LIBS=""; case " $EXTRA " in *" gemm "*) LIBS="-lcublas";; esac
    if grep -q FAILED "$OUT/build.status"; then cat "$OUT/build.status"; for l in "$OUT"/*.log; do echo "== $l"; grep -E "error" "$l" | head -12; done; exit 1; fi
    $NVCC -o "$OUT/dq_test" $OUT/*.o $LIBS > "$OUT/link.log" 2>&1 || { cat "$OUT/link.log"; exit 1; }
else
    HP=${HIP_ROOT:?HIP_ROOT}
    ARCH=${HIP_ARCH:?HIP_ARCH}
    CXX=${HIPCXX:-$HP/llvm/bin/clang++}
    export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr ROCM_PATH=$HP HIP_PATH=$HP
    FL="--rocm-path=$HP --rocm-device-lib-path=$HP/amdgcn/bitcode -O2 -std=c++20 --offload-arch=$ARCH -fPIE -ffp-contract=off"
    FL="$FL -include $SRC/include/strata/dsv41/hip_extra.h -include $SRC/include/strata/hip_compat/cuda_runtime.h"
    FL="$FL -DSTRATA_USE_HIP=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -isystem $HP/include $DEFS $INC -I$SRC/include/strata/hip_compat"
    case " $EXTRA " in *" gemm "*) FILES="$FILES src/prefill/wmma_gemm.cu";; esac
    cmds=()
    for f in tests/dsv41/dq_dense_test.cpp $FILES; do
        o="$OUT/$(basename "$f").o"
        [ -f "$o" ] && [ "$o" -nt "$SRC/$f" ] && continue   # (up to date)
        cmds+=("cd $SRC && $CXX $FL -x hip -c $f -o $o > $o.log 2>&1 || echo FAILED $f >> $OUT/build.status")
    done
    run_jobs "${cmds[@]}"
    if grep -q FAILED "$OUT/build.status"; then cat "$OUT/build.status"; for l in "$OUT"/*.log; do echo "== $l"; grep -E "error" "$l" | head -12; done; exit 1; fi
    LIBS="-L$HP/lib -lamdhip64 -Wl,-rpath,$HP/lib"
    case " $EXTRA " in *" gemm "*) LIBS="$LIBS -L${HIPBLAS_LIB:-$HP/lib} -lhipblas -Wl,-rpath,${HIPBLAS_LIB:-$HP/lib}";; esac
    $CXX --hip-link --rtlib=compiler-rt -unwindlib=libgcc --offload-arch=$ARCH -o "$OUT/dq_test" $OUT/*.o $LIBS > "$OUT/link.log" 2>&1 || { cat "$OUT/link.log"; exit 1; }
fi
echo "built $OUT/dq_test"
