#!/bin/sh
# Builds dsv4_plan, dsv4_run + both test binaries and runs the tests. Honors CXX (default g++-10, else g++).
# Output dir: ./build-out, or /tmp/dsv4_build if this tree is mounted noexec.
#
#   sh build.sh            CPU-emulated device (src/gpu.cpp): the only build the golden tests run against.
#   sh build.sh --cuda     ALSO builds dsv4_run_cuda on the real device (src/gpu_cuda.cu, needs nvcc).
#   sh build.sh --cuda-only  skips the emulated binaries and the tests.
#
# The two device backends define the same dsv4::gpu symbols, so they are never in the same binary: the
# tests always exercise the emulated one, which is bit-identical to the CPU reference.
set -e
cd "$(dirname "$0")"
CXX="${CXX:-$(command -v g++-10 || command -v g++)}"

WITH_CUDA=0; CUDA_ONLY=0
for a in "$@"; do
    case "$a" in
        --cuda) WITH_CUDA=1 ;;
        --cuda-only) WITH_CUDA=1; CUDA_ONLY=1 ;;
        *) echo "usage: $0 [--cuda|--cuda-only]"; exit 2 ;;
    esac
done

# Gestione sicura della directory di output per ambienti noexec
OUT=build-out; mkdir -p "$OUT"
cp /bin/true "$OUT/.x" 2>/dev/null && "$OUT/.x" 2>/dev/null || OUT=/tmp/dsv4_build
mkdir -p "$OUT"

# OpenMP e' OBBLIGATORIO per la velocita': senza, ogni "#pragma omp parallel for" (cpu_mv, gpu::matvec,
# experts_hit) e' inerte e il MoE gira su un core solo. Il probe evita di rompere la build su un
# compilatore senza libgomp.
OMP=""
# Il probe deve linkare un main vero: "-x c /dev/null" fallisce sempre per via di _start, con o senza OpenMP.
if printf '#include <omp.h>\nint main(){return omp_get_max_threads()>0?0:1;}\n' | "$CXX" -fopenmp -x c - -o "$OUT/.omp_probe" 2>/dev/null; then
    OMP="-fopenmp"
else
    echo "WARNING: $CXX has no working OpenMP: dsv4_run will run SINGLE-THREADED (roughly 20x slower)."
fi
rm -f "$OUT/.omp_probe"

FL="-std=c++17 -O2 -Wall -Wextra -Iinclude $OMP ${CXXFLAGS_EXTRA:-}"

# Le tabelle dei codebook IQ* sono generate (llama.cpp viene solo LETTO): rigenerale se mancano.
if [ ! -f src/iq_tables.cpp ] || [ ! -f include/dsv4/iq_tables.hpp ]; then
    echo "==> Generating IQ codebook tables (src/iq_tables.cpp)..."
    python3 tools/extract_iq_tables.py "${LLAMA_CPP_DIR:-$HOME/workspaces/llama.cpp}" src/iq_tables.cpp include/dsv4/iq_tables.hpp
fi

# Sorgenti condivisi della libreria core (incluso model.cpp)
# src/iq_tables.cpp e' generato: python3 tools/extract_iq_tables.py
# SRC_NO_GPU = the same without any device backend: the CUDA build adds src/gpu_cuda.cu to it.
CORE="src/gguf_header.cpp src/config.cpp src/mem_plan.cpp src/ops.cpp src/dequant.cpp src/iq_tables.cpp src/sampler.cpp src/serve_loop.cpp src/model.cpp"
SRC="$CORE src/gpu.cpp"
SRC_NO_GPU="$CORE"

if [ "$CUDA_ONLY" = 0 ]; then
    echo "==> Compiling test suite (CPU-emulated device)..."
    "$CXX" $FL $SRC tests/test_dsv4_core.cpp -o "$OUT/test_dsv4_core"
    "$CXX" $FL $SRC tests/test_ops.cpp -o "$OUT/test_ops"
    # The CUDA decode math, checked on the host against dequant.cpp: needs no GPU and no nvcc.
    "$CXX" $FL $SRC tests/test_dq_traits.cpp -o "$OUT/test_dq_traits"

    echo "==> Compiling dsv4_plan..."
    "$CXX" $FL $SRC src/dsv4_plan_main.cpp -o "$OUT/dsv4_plan"

    echo "==> Compiling dsv4_run (DeepSeek-V4-Flash runner, CPU-emulated device)..."
    "$CXX" $FL $SRC src/dsv4_run_main.cpp -o "$OUT/dsv4_run"

    echo "==> Running core tests..."
    "$OUT/test_dsv4_core"

    echo "==> Running operator tests..."
    "$OUT/test_ops" tests/golden.txt

    echo "==> Running CUDA decode-trait tests (host only, no GPU touched)..."
    "$OUT/test_dq_traits"
fi

if [ "$WITH_CUDA" = 1 ]; then
    NVCC="${NVCC:-$(command -v nvcc || true)}"
    [ -z "$NVCC" ] && for d in /usr/local/cuda/bin /usr/local/cuda-12.8/bin /usr/local/cuda-12/bin; do
        [ -x "$d/nvcc" ] && NVCC="$d/nvcc" && break
    done
    if [ -z "$NVCC" ]; then
        echo "ERROR: --cuda needs nvcc (set NVCC=/path/to/nvcc). Not built."
        exit 3
    fi
    # sm_89 is this box (RTX 4090); override for another card with DSv4_CUDA_ARCH=sm_80 etc.
    ARCH="${DSV4_CUDA_ARCH:-sm_89}"
    CUDA_INC="-Iinclude"
    CLIB="-L$(dirname "$(dirname "$NVCC")")/lib64 -lcudart"
    echo "==> Compiling dsv4_run_cuda (real CUDA device, $("$NVCC" --version | tail -1 | sed 's/^Build //'), arch $ARCH)..."
    "$NVCC" -std=c++17 -O2  ${NVCCFLAGS_EXTRA:-} -Xcompiler "$OMP" -Iinclude -gencode arch=compute_${ARCH#sm_},code=$ARCH \
        src/gpu_cuda.cu $SRC_NO_GPU src/dsv4_run_main.cpp -o "$OUT/dsv4_run_cuda" $CLIB
    echo "==> Compiling dsv4_plan_cuda..."
    "$NVCC" -std=c++17 -O2  ${NVCCFLAGS_EXTRA:-} -Xcompiler "$OMP" -Iinclude -gencode arch=compute_${ARCH#sm_},code=$ARCH \
        src/gpu_cuda.cu $SRC_NO_GPU src/dsv4_plan_main.cpp -o "$OUT/dsv4_plan_cuda" $CLIB
fi

echo ""
echo "========================================================"
echo " Build successful!"
[ "$CUDA_ONLY" = 0 ] && echo " Planner: $OUT/dsv4_plan" && echo " Runner:  $OUT/dsv4_run            (CPU-emulated device)"
[ "$WITH_CUDA" = 1 ] && echo " Runner:  $OUT/dsv4_run_cuda        (real CUDA device)"
echo "========================================================"
