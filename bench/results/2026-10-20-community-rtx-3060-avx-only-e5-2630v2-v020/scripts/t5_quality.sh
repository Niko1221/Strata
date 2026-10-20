#!/usr/bin/env bash
# T5: the RTX 3060 community report's correctness checks (HumanEval 0-39 with EvalPlus 0.3.1, 12 Japanese
# questions, 8 tool calls), so this machine's IQ2_XS can be put beside that report's IQ2_XS row.
# Generation through the server (IQ2_XS (STRATA_QUANT=q4: UD-IQ4_XS), context 40,960, node 1), then scoring offline without the GPU.  ~30 min
set -u
. "$(dirname "$0")/common.sh"
if [ -z "${T5_INNER:-}" ]; then
  preflight || exit 1
  D=$(newdir T5-quality$QTAG)
  make_config "$BASE_CFG" "$D/config.json" "$D/engine.log"
  say "T5 ($QNAME) -> $D"
  T5_INNER=1 T5_DIR=$D under_lease "$D/gpu-lease.json" "strata T5 quality (experiments/06_test)" -- "$0" "$@"
  echo "lease exit $?"
  say "scoring the code answers with EvalPlus 0.3.1 (offline, no GPU) ..."
  "$PYE" -I "$T/tools/strata_eval_subset.py" --out "$D" 2>&1 | tee "$D/evalplus.txt"
  say "T5 done: $D"
  exit 0
fi
trap server_down EXIT
server_up "$T5_DIR/config.json" "$T5_DIR/server.log" || exit 1
"$PYE" "$T/tools/quality_suite.py" --model "$QNAME" --out "$T5_DIR" "$@" 2>&1 | tee "$T5_DIR/quality.txt"
echo "quality_suite exit ${PIPESTATUS[0]}"
