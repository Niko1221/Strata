#!/usr/bin/env bash
# tools/prefill_preempt_regression.sh - the full regression for the prefill-preemption feature, run N times
# (the task's "check everything three times").  Everything must pass, every time:
#   1. the C++ suite (ctest in build/)
#   2. the Python suites (unittest: the server, the preemption scheduler, detok, mcp, winjob, tools)
#   3. the engine parity harness on the real model (tools/prefill_preempt_test.py, GPU)
#
#   tools/prefill_preempt_regression.sh 3
set -u
cd "$(dirname "$0")/.."
N=${1:-3}
BUILD=build
overall=0

for i in $(seq 1 "$N"); do
    echo "================ RUN $i of $N ================"
    echo "--- ctest ($BUILD)"
    (cd "$BUILD" && ctest --output-on-failure 2>&1 | tail -4) || overall=1
    echo "--- python suites"
    .venv/bin/python -m unittest serve.test_server serve.test_preempt serve.test_detok serve.test_winjob 2>&1 | tail -3 || overall=1
    .venv/bin/python -m unittest discover -s tools -p "test_*.py" 2>&1 | tail -3 || overall=1
    echo "--- engine parity harness (GPU, real model)"
    .venv/bin/python tools/prefill_preempt_test.py --max-new 16 2>&1 | grep -E "scenario|ALL|FAIL" | tail -14 || overall=1
done

echo "================ RESULT: $([ $overall -eq 0 ] && echo ALL CLEAN || echo FAILURES) ================"
exit $overall
