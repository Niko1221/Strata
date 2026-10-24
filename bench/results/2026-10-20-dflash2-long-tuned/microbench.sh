#!/usr/bin/env bash
set -euo pipefail
: "${STRATA_LONG_RESULTS:?Set a fresh output directory}"
mkdir -p "$STRATA_LONG_RESULTS"
git show 2103f3e4b7fc56cfa3231ecc001ebe0a931a7db9:src/kernels/cuda/dflash2.cu > "$STRATA_LONG_RESULTS/original.cu"
"${CUDACXX:-/usr/local/cuda/bin/nvcc}" -std=c++20 -O3 -DNDEBUG -arch=sm_89 \
  -Iinclude -I/usr/local/cuda/include "$STRATA_LONG_RESULTS/original.cu" \
  -x cu src/kernels/dflash2_parity.cpp -o "$STRATA_LONG_RESULTS/attention-original"
cmake --build build-dflash2 --target dflash2_parity mmvq_il_parity -j 4
for rep in 0 1 2; do
  STRATA_MMVQ_IL_Q8_HEAD=1 MMVQ_IL_ONLY=Q8_0 build-dflash2/mmvq_il_parity --bench \
    > "$STRATA_LONG_RESULTS/head-$rep.log" 2>&1
  order='before after'
  if [ "$rep" = 1 ]; then order='after before'; fi
  for tag in $order; do
    executable=build-dflash2/dflash2_parity
    if [ "$tag" = before ]; then executable="$STRATA_LONG_RESULTS/attention-original"; fi
    "$executable" --bench "$STRATA_LONG_RESULTS/attention-$tag-$rep.bits" \
      > "$STRATA_LONG_RESULTS/attention-$tag-$rep.log" 2>&1
  done
  cmp "$STRATA_LONG_RESULTS/attention-before-$rep.bits" "$STRATA_LONG_RESULTS/attention-after-$rep.bits"
done
sha256sum "$STRATA_LONG_RESULTS"/attention-*.bits > "$STRATA_LONG_RESULTS/attention-sha256.txt"
