#!/bin/bash
# Regression test for the draft layer's buffers being given back before process teardown (SYCL).
#
# `--spec 4` with a reply of a few hundred tokens used to end in `Segmentation fault (core dumped)` - exit 139 - after
# `decode`, `prefill` and `time to first token` had all printed.  A handler installed from a preloaded library
# (arc770's tools/probes/trace_segv.c, because ptrace is unavailable in this container) put the fault in
# `sycl::_V1::free(dense_, ...)` inside `libze_intel_gpu.so.1`, called from `strata::core::MtpDrafter::~MtpDrafter()`
# from `main`, and the frame below it said what the situation was: `level_zero backend failed with error: 37
# (UR_RESULT_ERROR_UNINITIALIZED)`.  The drafter is the last of main's objects to be destroyed, and by then the
# runtime is part-way down; freeing the draft layer's 111 MiB of dense weights and its expert arena into that faults
# in the driver.  A guard memcpy in the destructor reported the uninitialized runtime as an exception, which is how
# the same bug was seen twice with two different symptoms.
#
# The window width is not the trigger and neither is the teardown ordering: `--spec 4 --max-new 512` faulted with the
# verify window at 6, at 4 (`--suffix-draft 0`) and at 4 (`--mtp-max-t 4`); destroying the drafter's side queues after
# every free changed nothing; `EnableMemoryPool=0`, `DisableMemoryPoolRecycling=1` and
# `ForceGenericMemoryCacheAllocation=1` changed nothing; a run with no `--mtp` never faulted.  `--spec 3 --max-new 512`
# was clean, and `--spec 4 --max-new 192` was flaky on one build - it exited 0 twice and 139 once - because whether a
# free into a dying runtime survives is not a stable question.  This is the failure the SYCL port already names where
# it closes a server ("Their destructors ran against a runtime already shutting down and aborted (exit 139)") and
# works around with `_Exit`; the one-shot path had no such release for the drafter.
#
# So `MtpDrafter::release()` hands back everything the destructor used to - the captures first, then the buffers, each
# pointer nulled as it goes so it is safe to call twice - and generate.cpp calls it, for the canonical drafter and for
# the batch slots' drafters, next to the buffers it already frees by hand before returning.  Measured on an Arc A770
# 16 GB with IQ3_XXS (2026-11-03): the four arms below were 139, 139, 139/0 (flaky) and 139 on the unmodified build
# and all exit 0 with the release.
#
#   STRATA_BIN=build-sycl/strata MODEL_ARGS="--pack ... --native ... --ple-gguf ... --mtp ... \
#       --max-context 8192 --kv int8 --expert-cache auto --prefill auto --seed 1 ..." \
#       sycl/tools/test_mtp_teardown.sh
#
# MODEL_ARGS are the engine arguments of the box (everything but --tokens-file and the speculation arms).  Exit 0 when
# every arm completes cleanly, 1 when any of them does not, 2 for usage.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${STRATA_BIN:-}
[ -n "$BIN" ] && [ -x "$BIN" ] && [ -n "${MODEL_ARGS:-}" ] || { echo "usage: STRATA_BIN=<strata> MODEL_ARGS=\"...\" $0"; exit 2; }

# The first arm is repeated: the fault was flaky at the 192-token cap and reliable at 512, so a pass has to be seen
# twice at the depth that reproduced it.
ARMS=("--spec 4 --max-new 512" "--spec 4 --max-new 512" "--spec 4 --max-new 192" "--spec 3 --max-new 512")
FAILED=""
for arm in "${ARMS[@]}"; do
    OUT=$(mktemp)
    timeout "${TEARDOWN_TIMEOUT:-900}" "$BIN" $MODEL_ARGS $arm \
        --tokens-file "$HERE/data/prompt_deadlock_2744.tok" > "$OUT" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        FAILED="$FAILED  $arm -> exit $rc\n"
        grep -oE "terminate called[^\n]*|what\(\):  [^\n]*" "$OUT" | head -2 | sed 's/^/    /'
        tail -2 "$OUT" | sed 's/^/    /'
    fi
    rm -f "$OUT"
    sleep "${TEARDOWN_GAP:-4}"
done
if [ -n "$FAILED" ]; then
    printf 'FAIL: a run that finished generating did not exit cleanly:\n%b' "$FAILED"
    echo "the draft layer's buffers have to be given back while the device is still there"
    exit 1
fi
echo "PASS: ${#ARMS[@]} speculation arms generated their full reply and exited 0"
exit 0
