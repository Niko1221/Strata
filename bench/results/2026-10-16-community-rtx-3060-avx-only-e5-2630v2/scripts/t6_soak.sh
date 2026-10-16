#!/usr/bin/env bash
# T6: soak - 65 minutes of back-to-back requests (docs/TEST_REQUESTS.md asks for an hour or more), with GPU and RAM
# samples every 10 s. IQ2_XS (STRATA_QUANT=q4: UD-IQ4_XS), context 40,960, node 1.   ~70 min
#   ./t6_soak.sh [--minutes M]
set -u
. "$(dirname "$0")/common.sh"
if [ -z "${T6_INNER:-}" ]; then
  preflight || exit 1
  D=$(newdir T6-soak$QTAG)
  make_config "$BASE_CFG" "$D/config.json" "$D/engine.log"
  say "T6 ($QNAME) -> $D"
  T6_INNER=1 T6_DIR=$D under_lease "$D/gpu-lease.json" "strata T6 soak (experiments/06_test)" -- "$0" "$@"
  echo "lease exit $?"
  say "T6 done: $D"
  exit 0
fi
trap server_down EXIT
server_up "$T6_DIR/config.json" "$T6_DIR/server.log" || exit 1
python3 "$T/tools/soak.py" --out "$T6_DIR" "$@" 2>&1 | tee "$T6_DIR/soak.txt"
echo "soak exit ${PIPESTATUS[0]}"
