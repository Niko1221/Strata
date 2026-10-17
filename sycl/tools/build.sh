#!/usr/bin/env bash
# Configure + build the SYCL port inside strata-sycl-dev.  sycl/build.sh [target...]
#
# No targets builds the engine and the image encoder (tools/vision with -DSTRATA_VISION_SYCL=ON, into
# build-vision-sycl/ - where sycl/setup_intel.py, sycl_vision_encoder(), looks for it); `strata` builds the
# engine alone and `strata-vision` the encoder alone.  The encoder needs a llama.cpp checkout at the pinned
# commit (third_party/llama.cpp, or LLAMA_DIR=<checkout>) and oneMKL (setvars.sh exports MKLROOT;
# MKL_DIR=<dir with lib/cmake/mkl> overrides); without either it says so and the engine still builds.
set -uo pipefail
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true; set -u
repo=${REPO:-$(git rev-parse --show-toplevel)}
b=${BUILD_DIR:-$repo/build-sycl}
vb=${VISION_BUILD_DIR:-$repo/build-vision-sycl}

# tools/vision (strata-vision) with ggml-sycl and oneMKL -> $vb/bin/strata-vision
build_vision() {
    local llama=${LLAMA_DIR:-$repo/third_party/llama.cpp}
    local mkl=${MKL_DIR:-}
    if [ -z "$mkl" ] && [ -n "${MKLROOT:-}" ] && [ -f "$MKLROOT/lib/cmake/mkl/MKLConfig.cmake" ]; then
        mkl="$MKLROOT/lib/cmake/mkl"
    fi
    [ -f "$llama/ggml/CMakeLists.txt" ] || {
        echo "strata-vision: no llama.cpp checkout at $llama" >&2
        echo "  setup.py downloads the pinned commit there (third_party/llama.cpp); LLAMA_DIR=<checkout> overrides" >&2
        return 1
    }
    [ -n "$mkl" ] || {
        echo "strata-vision: oneMKL not found (source /opt/intel/oneapi/setvars.sh, or MKL_DIR=<dir with lib/cmake/mkl>)" >&2
        return 1
    }
    [ -f "$vb/build.ninja" ] || cmake -S "$repo/tools/vision" -B "$vb" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DLLAMA_DIR="$llama" -DMKL_DIR="$mkl" \
        -DSTRATA_VISION_SYCL=ON 2>&1 | tail -8
    cmake --build "$vb" -j "${JOBS:-12}" --target strata-vision 2>&1 | tee "$vb/build.log" \
        | grep -E '^FAILED|error:|Linking' | tail -20
    echo "VISION BUILD EXIT ${PIPESTATUS[0]}"
}

if [ "${1:-}" = strata-vision ]; then
    build_vision
    ls -la "$vb/bin/strata-vision" 2>&1 | tail -1
    exit
fi

[ -f $b/build.ninja ] || cmake -S $repo/sycl -B $b -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DSTRATA_SYCL_AOT="${AOT:-}" -DSTRATA_SYCL_SPIN_MAX="${SPIN_MAX:-}" 2>&1 | tail -15
cmake --build $b -j ${JOBS:-12} ${1:+--target "$@"} -- -k 0 2>&1 | tee $b/build.log | grep -E '^FAILED|error:|^ninja: build stopped|Linking|^\[[0-9]+/[0-9]+\] Linking' | tail -40
echo "BUILD EXIT ${PIPESTATUS[0]}"
grep -c 'error:' $b/build.log | sed 's/^/errors: /'
grep -E '^FAILED' $b/build.log | sed 's#.*/##' | sort | uniq | head -60

# no targets: the encoder beside the engine, so a fresh build has both halves of the port
if [ $# -eq 0 ]; then
    echo
    echo "=== the image encoder (tools/vision, ggml-sycl) -> $vb ==="
    build_vision || echo "the engine is built; the encoder is NOT (see above) - images stay off"
    ls -la "$vb/bin/strata-vision" 2>&1 | tail -1
fi
