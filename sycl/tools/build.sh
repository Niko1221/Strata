#!/usr/bin/env bash
# Configure + build the SYCL port inside strata-sycl-dev.  sycl/build.sh [target...]
set -uo pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
repo=$(git rev-parse --show-toplevel)
b=${BUILD_DIR:-$repo/build-sycl}
[ -f $b/build.ninja ] || cmake -S $repo/sycl -B $b -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DSTRATA_SYCL_AOT="${AOT:-}" 2>&1 | tail -15
cmake --build $b -j ${JOBS:-12} ${1:+--target "$@"} -- -k 0 2>&1 | tee $b/build.log | grep -E '^FAILED|error:|^ninja: build stopped|Linking|^\[[0-9]+/[0-9]+\] Linking' | tail -40
echo "BUILD EXIT ${PIPESTATUS[0]}"
grep -c 'error:' $b/build.log | sed 's/^/errors: /'
grep -E '^FAILED' $b/build.log | sed 's#.*/##' | sort | uniq | head -60
