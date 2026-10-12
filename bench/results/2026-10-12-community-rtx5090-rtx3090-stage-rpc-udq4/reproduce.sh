#!/usr/bin/env bash
set -euo pipefail

: "${STRATA_REPO:?Set STRATA_REPO to the existing compatible Strata checkout}"
: "${STRATA_CONFIG:?Set STRATA_CONFIG to the coordinator configuration file}"
: "${BENCH_OUT:?Set BENCH_OUT to a new report output directory}"
STRATA_URL=${STRATA_URL:-http://127.0.0.1:8090}

cd -- "$STRATA_REPO"
exec python3 tools/strata_bench.py \
    --config "$STRATA_CONFIG" --url "$STRATA_URL" \
    --runs 3 --long --out "$BENCH_OUT"
