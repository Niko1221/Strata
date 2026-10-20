#!/usr/bin/env bash
# T1: Strata's own tests that need no model.
#   1) the GPU-free Python tests (tools/test_*.py, 68 modules)        ~10 min, no GPU
#   2) the C++/CUDA test suite (ctest, 119 tests in build-tests/)     uses the GPU: run under the GPU Jobs lease
# T1_QUICK=1 ./t1_unit.sh: short form. ctest skips the benchmarks and the stress test (-E 'bench|stress') and
#   gives each test 120 s instead of 600 s.
set -u
. "$(dirname "$0")/common.sh"
if [ -z "${T1_INNER:-}" ]; then
  preflight || exit 1
  D=$(newdir T1-unit$VTAG)
  say "T1 -> $D"
  say "1/2 Python tests (no GPU) ..."
  STRATA_S=$S "$T/tools/run_py_tests.sh" "$D" | tee "$D/py-tests-result.txt"
  say "2/2 ctest (GPU, under the lease) ..."
  T1_INNER=1 T1_DIR=$D under_lease "$D/gpu-lease.json" "strata T1 ctest (experiments/06_test)" -- "$0"
  echo "lease exit $?"
  say "T1 done: $D"
  exit 0
fi
# inside the lease
CTEST=$TOP/experiments/03_run/.venv-setup/bin/ctest
# 3 test binaries (host_graph_launch_test, dma_batch_parity, gdn_rec_parity) are built without the RUNPATH the
# others carry and cannot find libcudart.so.13 (seen 2026-10-16): point them at the same CUDA runtime
export LD_LIBRARY_PATH=$TOP/toolchain/.venv/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
if [ -n "${T1_QUICK:-}" ]; then CT_OPTS=(-E 'bench|stress' --timeout 120); echo "ctest: short form (-E 'bench|stress', 120 s per test)"
else CT_OPTS=(--timeout 600); fi
cd "$TESTBUILD" || exit 1
# hb.py: status line (with ctest's latest "N/119 Test #..." line) every 60 s, failures/crashes shown at once,
# a warning when nothing happens for 15 min (ctest's own per-test limit is 600 s)
python3 "$T/tools/hb.py" --every 60 --stall 900 --label ctest --out "$T1_DIR/ctest.log" \
  --alert '\*\*\*(Failed|Timeout|Exception)|Illegal|Segmentation|SEGFAULT|Subprocess aborted' -- \
  "$CTEST" --output-on-failure "${CT_OPTS[@]}" --output-junit "$T1_DIR/ctest-junit.xml"
echo "ctest exit $?"
grep -E 'tests passed|tests failed' "$T1_DIR/ctest.log"
sed -n '/The following tests FAILED/,$p' "$T1_DIR/ctest.log"
