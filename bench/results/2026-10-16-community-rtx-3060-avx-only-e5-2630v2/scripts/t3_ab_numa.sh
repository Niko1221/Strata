#!/usr/bin/env bash
# T3: A/B of the NUMA placement with the official tools/ab_engine.py (a fresh server per arm, the arms' order
# alternating, 3 rounds; greedy, three short chats + one long prompt). Arms:
#   both:  the engine on both sockets (no binding; setup's default)
#   node1: the engine bound to NUMA node 1 (the GPU's node; what start_strata.sh does)
# IQ2_XS (STRATA_QUANT=q4: UD-IQ4_XS), context 40,960.   ~45 min
set -u
. "$(dirname "$0")/common.sh"
if [ -z "${T3_INNER:-}" ]; then
  preflight || exit 1
  D=$(newdir T3-ab-numa$QTAG)
  make_config "$BASE_CFG" "$D/config.json" "$D/engine.log"
  say "T3 ($QNAME) -> $D"
  T3_INNER=1 T3_DIR=$D under_lease "$D/gpu-lease.json" "strata T3 NUMA A/B (experiments/06_test)" -- "$0" "$@"
  echo "lease exit $?"
  say "T3 done: $D"
  exit 0
fi
cd "$S" || exit 1
export TMPDIR=$T3_DIR    # ab_engine writes each arm's config and engine log to the temp folder: keep them
"$PYS" tools/ab_engine.py "$T3_DIR/config.json" "both:" "node1:exe=$T/tools/strata-node1" --rounds 3 \
  --log "$T3_DIR/ab_engine.jsonl" "$@" 2>&1 | tee "$T3_DIR/ab_engine.txt"
echo "ab_engine exit ${PIPESTATUS[0]}"
