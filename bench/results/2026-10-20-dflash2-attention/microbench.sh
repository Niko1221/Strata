#!/usr/bin/env bash
set -euo pipefail
: "${STRATA_DRAFT_RESULTS:?Set a fresh output directory}"
mkdir -p "$STRATA_DRAFT_RESULTS"
git show c7bebe215a6feb754e08dfbc02a3ac12d38fa55f:src/kernels/cuda/dflash2.cu > "$STRATA_DRAFT_RESULTS/dflash2-before.cu"
"${CUDACXX:-/usr/local/cuda/bin/nvcc}" -std=c++20 -O3 -DNDEBUG -arch=sm_89 \
  -Iinclude -I/usr/local/cuda/include "$STRATA_DRAFT_RESULTS/dflash2-before.cu" \
  -x cu src/kernels/dflash2_parity.cpp -o "$STRATA_DRAFT_RESULTS/attention-before"
cmake --build build-dflash2 --target dflash2_parity -j 4
for rep in 0 1 2; do
  order='before after'
  if [ "$rep" = 1 ]; then order='after before'; fi
  for tag in $order; do
    binary=build-dflash2/dflash2_parity
    if [ "$tag" = before ]; then binary="$STRATA_DRAFT_RESULTS/attention-before"; fi
    "$binary" --bench "$STRATA_DRAFT_RESULTS/attention-$tag-$rep.bits" \
      > "$STRATA_DRAFT_RESULTS/micro-$tag-$rep.log" 2>&1
  done
  cmp "$STRATA_DRAFT_RESULTS/attention-before-$rep.bits" "$STRATA_DRAFT_RESULTS/attention-after-$rep.bits"
done
sha256sum "$STRATA_DRAFT_RESULTS"/attention-*.bits > "$STRATA_DRAFT_RESULTS/attention-bits-sha256.txt"
