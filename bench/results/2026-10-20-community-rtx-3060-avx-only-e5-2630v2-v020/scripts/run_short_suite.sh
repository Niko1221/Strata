#!/bin/bash
# The short test set on Strata 0.2.0 (EXPERIMENT_LOG.md §21): build the ctest suite, T1 (short form), T4, T5.
# T0 and T2 were run separately. Each step runs even if the one before failed; see each step's log.
T=<WORKDIR>/experiments/06_test
export STRATA_ENGINE=v020
cd "$T" || exit 1
echo "[$(date +%T)] build ctest suite -> build-tests-v020"
./build_tests.sh > build_tests-v020.log 2>&1; echo "build exit $? ($(grep -E '^\[[0-9]+/[0-9]+\]' build_tests-v020.log | tail -1 | cut -c1-40))"
echo "[$(date +%T)] T1 (short form)"; T1_QUICK=1 ./t1_unit.sh 2>&1 | grep -E 'python tests:|tests passed|tests failed|FAILED|done|exit' | tail -20
echo "[$(date +%T)] T4"; ./t4_needle.sh 2>&1 | tail -15
echo "[$(date +%T)] T5"; ./t5_quality.sh 2>&1 | tail -25
echo "[$(date +%T)] all done"
