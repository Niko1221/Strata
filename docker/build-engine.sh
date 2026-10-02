#!/usr/bin/env bash
# Configure and build the Strata engine for one AMD gfx target, inside strata-hip-builder.
#
# This is docs/AMD_HIP.md:41-47 plus what setup.py's build_engine_hip() adds (Ninja, the MMQ
# prefill, ccache), adapted for a system ROCm: a container from rocm/dev-* has hipcc, hipBLAS and
# hipBLASLt in /opt/rocm, so none of the --rocm-path / --rocm-device-lib-path flags that setup.py
# needs for AMD's TheRock wheels are required here.
#
# Env: SRC (default /src)  BUILD (default $SRC/build-hip)  JOBS  CCACHE_DIR
#      CMAKE_TESTS=0 to skip test targets  CMAKE_EXTRA="..." for extra -D flags
set -euo pipefail

ARCH="${1:-${STRATA_HIP_ARCH:-gfx1101}}"
SRC="${SRC:-/src}"
BUILD="${BUILD:-$SRC/build-hip}"
JOBS="${JOBS:-$(nproc)}"
ROCM="${ROCM_PATH:-/opt/rocm}"
CMAKE_TESTS="${CMAKE_TESTS:-1}"

log() { printf '[build-engine] %s\n' "$*"; }
die() { printf '[build-engine] ERROR: %s\n' "$*" >&2; exit 1; }

case "$ARCH" in
  gfx1100|gfx1101) ;;
  *) die "arch '$ARCH' is not in the HIP allowlist (gfx1100, gfx1101)." ;;
esac

# ---------------------------------------------------------------- preflight
# Cheap checks with specific advice, because the failure modes here are slow to diagnose later.
[ -d "$SRC/src" ] || die "no $SRC/src - mount the repository at \$SRC"
[ -x "$ROCM/bin/hipcc" ] || die "hipcc not found in $ROCM - this image must be a ROCm *dev* image"
[ -f "$ROCM/lib/cmake/hipblas/hipblas-config.cmake" ] \
  || [ -f "$ROCM/lib/cmake/hipblas/hipblasConfig.cmake" ] \
  || die "the hipBLAS CMake package is missing from $ROCM/lib/cmake - the '-complete' base image tag has it"

if [ -d "$ROCM/lib/rocblas/library" ]; then
  n_kernels=$(find "$ROCM/lib/rocblas/library" -name "*${ARCH}*" | wc -l)
  if [ "$n_kernels" -eq 0 ]; then
    log "WARNING: no rocBLAS Tensile kernels for $ARCH in this image;" \
        "GEMM will fall back or fail at runtime (not build time)"
  else
    log "rocBLAS: $n_kernels kernel object(s) for $ARCH"
  fi
fi

# cmake/hip_backend.cmake gates the arch before anything compiles. If it has not been widened yet
# (docs/DOCKER_GFX1101_PLAN.md Part C item 1), say so here rather than after a 5-minute configure.
if [ "$ARCH" != "gfx1100" ] && ! grep -q "$ARCH" "$SRC/cmake/hip_backend.cmake"; then
  die "cmake/hip_backend.cmake still accepts only gfx1100; it would abort the configure with
         'Strata HIP currently supports only gfx1100 wave32'.
       Apply the gfx1101 patch set first: docs/DOCKER_GFX1101_PLAN.md Part C items 1-4
       (that file, src/core/device.cu, include/strata/hip_compat/intrinsics.hpp)."
fi

export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr ROCM_PATH="$ROCM" HIP_PATH="$ROCM"
export PATH="$ROCM/bin:$ROCM/lib/llvm/bin:$PATH"

log "arch=$ARCH  src=$SRC  build=$BUILD  jobs=$JOBS  hipcc=$($ROCM/bin/hipcc --version 2>&1 | tail -1)"

# ---------------------------------------------------------------- configure
CONF=(cmake -G Ninja -S "$SRC" -B "$BUILD"
  -DCMAKE_BUILD_TYPE=Release
  -DSTRATA_ENABLE_HIP=ON
  -DSTRATA_ENABLE_CUDA=OFF
  -DSTRATA_PREFILL_MMQ=ON
  -DSTRATA_BUILD_TESTS="$([ "$CMAKE_TESTS" = 1 ] && echo ON || echo OFF)"
  -DCMAKE_HIP_ARCHITECTURES="$ARCH"
  # ccache is the difference between a 2-minute rebuild and a 20-minute one. CMake honours the C++
  # launcher for certain; the HIP one is honoured by CMake >= 3.23 and ignored with a warning below.
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
  -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
  ${CMAKE_EXTRA:+$CMAKE_EXTRA})

log "configuring: ${CONF[*]}"
"${CONF[@]}" || die "configure failed (see above). If the HIP compiler was not found, pass
       -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++ through CMAKE_EXTRA."

# ---------------------------------------------------------------- build
# Build ALL targets, not just strata: docs/AMD_HIP.md:141 says the registered checks need the
# parity/fixture targets present, and CTest would otherwise report a missing binary confusingly.
# A failed build is retried once - setup.py:1044 does the same for a flaky ptxas (issue #45), and
# the second attempt only compiles what is still missing.
cmake --build "$BUILD" -j "$JOBS" || {
  log "the build stopped - trying it once more (only the missing pieces recompile)"
  cmake --build "$BUILD" -j "$JOBS"
}

# ---------------------------------------------------------------- report
[ -x "$BUILD/strata" ] || die "the build finished but $BUILD/strata is not there"
for t in strata-device hip_intrinsics hip_expert_cache_staging; do
  [ -x "$BUILD/$t" ] && log "artifact: $t"
done
log "engine: $BUILD/strata  ($(stat -c %s "$BUILD/strata") bytes, sha256 $(sha256sum "$BUILD/strata" | cut -c1-16)...)"
log "next: run 'ctest --test-dir $BUILD --output-on-failure --timeout 60 -E \"^(ple_parity)\$\"'" \
    "on a machine that has the GPU (needs /dev/kfd + the render node)"
