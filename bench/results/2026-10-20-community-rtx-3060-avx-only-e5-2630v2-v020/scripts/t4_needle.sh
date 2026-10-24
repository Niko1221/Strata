#!/usr/bin/env bash
# T4: long-context recall with the official tools/needle_bench.py (a code word hidden in the repository's text,
# greedy, thinking off): about 4K / 16K / 32K tokens x depth 10 / 50 / 90 %. IQ2_XS (STRATA_QUANT=q4: UD-IQ4_XS), context 40,960, node 1.  ~25 min
set -u
. "$(dirname "$0")/common.sh"
if [ -z "${T4_INNER:-}" ]; then
  preflight || exit 1
  D=$(newdir T4-needle$QTAG)
  make_config "$BASE_CFG" "$D/config.json" "$D/engine.log"
  say "T4 ($QNAME) -> $D"
  T4_INNER=1 T4_DIR=$D under_lease "$D/gpu-lease.json" "strata T4 needle (experiments/06_test)" -- "$0"
  echo "lease exit $?"
  say "T4 done: $D"
  exit 0
fi
trap server_down EXIT
server_up "$T4_DIR/config.json" "$T4_DIR/server.log" || exit 1
cd "$S" && "$PYS" tools/needle_bench.py --url http://127.0.0.1:$PORT --lengths 4k,16k,32k --depths 10,50,90 \
  --out "$T4_DIR/needles.json" 2>&1 | tee "$T4_DIR/needles.txt"
echo "needle_bench exit ${PIPESTATUS[0]}"
