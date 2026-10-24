#!/usr/bin/env bash
# T2: the official speed benchmark (tools/strata_bench.py --long, 3 runs per test, medians and ranges), twice:
#   cold: the model files dropped from the page cache before the server starts
#   warm: started again right after, the files still in the page cache
# IQ2_XS (STRATA_QUANT=q4: UD-IQ4_XS), context 40,960, NUMA node 1 (the recommended setup, EXPERIMENT_LOG.md §8/§9).   ~40 min
set -u
. "$(dirname "$0")/common.sh"
if [ -z "${T2_INNER:-}" ]; then
  preflight || exit 1
  D=$(newdir T2-speed$QTAG)
  make_config "$BASE_CFG" "$D/config.json" "$D/engine.log"
  say "T2 ($QNAME) -> $D"
  T2_INNER=1 T2_DIR=$D under_lease "$D/gpu-lease.json" "strata T2 speed (experiments/06_test)" -- "$0"
  echo "lease exit $?"
  say "T2 done: $D"
  exit 0
fi
cd "$S" || exit 1
for arm in cold warm; do
  if [ $arm = cold ]; then python3 "$T/tools/drop_cache.py" "$T/manifest/sha256-reference.txt" | tee "$T2_DIR/drop-cache.txt"; fi
  say "strata_bench ($arm) ..."
  python3 "$B/numa_run.py" --node 1 -- "$PYS" tools/strata_bench.py --config "$T2_DIR/config.json" --long \
    --out "$T2_DIR/$arm" > "$T2_DIR/$arm.log" 2>&1
  echo "strata_bench $arm exit $?"
  cp "$T2_DIR/engine.log" "$T2_DIR/engine-$arm.log" 2>/dev/null
  sleep 10
done
