#!/usr/bin/env bash
# T7 (optional): the official quant yardstick, tools/quant_quality.py: 5 x 1,000 tokens of fixed text read
# teacher-forced with every expert on the CPU (so it runs this machine's AVX1 kernels on every position).
#   run 1 and run 2 of IQ2_XS: are the rows the same? (repeatability; QUANTS.md: Q8_0 twice gave identical rows)
#   UD-IQ4_XS once: IQ2_XS scored against it (NOT the official Q8_0 reference, which does not fit on this disk)
# Long: each run reads 5,000 tokens on the CPU alone. Start it only with several hours to spare.
set -u
. "$(dirname "$0")/common.sh"
if [ -z "${T7_INNER:-}" ]; then
  preflight || exit 1
  D=$(newdir T7-quant-quality)
  make_config "$BASE_CFG" "$D/iq2_xs.json" "$D/engine-iq2_xs.log" 32768
  make_config "$S/strata-unsloth-ud-iq4_xs.json" "$D/ud-iq4_xs.json" "$D/engine-ud-iq4_xs.log"
  say "T7 -> $D"
  T7_INNER=1 T7_DIR=$D under_lease "$D/gpu-lease.json" "strata T7 quant_quality (experiments/06_test)" -- "$0"
  echo "lease exit $?"
  say "T7 done: $D"
  exit 0
fi
cd "$S" || exit 1
QQ() { "$PYS" tools/quant_quality.py "$@"; }
W="python3 $B/numa_run.py --node 1 --"
for run in iq2_xs:run1 iq2_xs:run2 ud-iq4_xs:run1; do
  m=${run%%:*}; n=${run##*:}
  say "quant_quality $m $n ..."
  QQ run --config "$T7_DIR/$m.json" --out "$T7_DIR/$m-$n.tsv" --wrap "$W" > "$T7_DIR/$m-$n.log" 2>&1
  echo "$m $n exit $?"
done
QQ compare "$T7_DIR/iq2_xs-run1.tsv" "$T7_DIR/iq2_xs-run2.tsv" --segments 2>&1 | tee "$T7_DIR/compare-repeat.txt"
cmp -s "$T7_DIR/iq2_xs-run1.tsv" "$T7_DIR/iq2_xs-run2.tsv" && echo "run1 and run2: byte-identical" | tee -a "$T7_DIR/compare-repeat.txt" \
  || echo "run1 and run2: differ" | tee -a "$T7_DIR/compare-repeat.txt"
QQ compare "$T7_DIR/ud-iq4_xs-run1.tsv" "$T7_DIR/iq2_xs-run1.tsv" --segments 2>&1 | tee "$T7_DIR/compare-vs-iq4.txt"
