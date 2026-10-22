#!/bin/bash
# metal_batch_try.sh - stage-1 proof for batch slots on the Mac (Metal) engine.
#
# Runs the same prompts decoded alone (GEN, the solo window) and together in the batch windows (BGEN),
# greedy, and compares every slot's tokens with its solo tokens: they must be IDENTICAL (the batch
# window commits to the same summation order as the solo T=1 window).  It prints the batch window's
# build line and the aggregate decode rate.  Everything is logged, so a run can be read after the fact.
#
#   ./tools/metal_batch_try.sh            # 2 slots, 2 prompts, 128 tokens each
#   ./tools/metal_batch_try.sh 4 4 64     # 4 slots, 4 prompts, 64 tokens each
#
# The engine holds ~43 GB (a 33 GB arena plus weights) and sushi serve holds ~53 GB GPU-wired.  Both at
# once panicked this box (kernel watchdog, 2026-10-20): the script refuses to run while sushi is up.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

BATCH="${1:-2}"; N="${2:-$BATCH}"; MAXNEW="${3:-128}"
CFG=strata-iq3_xxs.json
EXE=build-metal/strata
OUT=/tmp/metal_batch_try
mkdir -p "$OUT"
LOG="$OUT/run_$(date +%Y%m%d_%H%M%S).log"

say() { echo "[metal_batch_try] $*"; }

# 1) the machine must be quiet: no sushi, no second strata engine, swap nearly free
if pgrep -fl "sushi serve" >/dev/null 2>&1; then
    say "sushi serve is running (pid $(pgrep -f 'sushi serve' | head -1))."
    say "It holds ~53-58 GB of GPU-wired memory; strata's engine plus it panicked macOS once."
    say "Stop sushi, then re-run this script.  To bring it back afterwards:"
    say "  $(pgrep -fl 'sushi serve' | head -1 | sed 's/^[0-9]* //')"
    exit 1
fi
if pgrep -fl "$EXE" >/dev/null 2>&1; then
    say "a strata engine is already running; stop it first."
    exit 1
fi
SWAP_USED=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1)
[ "${SWAP_USED:-0}" -gt 3000 ] && say "WARNING: ${SWAP_USED}M of swap is already used; the run may crawl."
FREE_GB=$(python3 - <<'PY'
import re, subprocess
out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
m = re.search(r"page size of (\d+) bytes", out)
ps = int(m.group(1)) if m else 16384
def n(key):
    for line in out.splitlines():
        if line.startswith(key):
            v = line.split()[-1].rstrip(".")
            return int(v) if v.isdigit() else 0
    return 0
free = n("Pages free") + n("Pages speculative")
print(free * ps // 2**30)
PY
)
say "free memory: ~${FREE_GB} GB (the engine needs ~43 GB); swap used: ${SWAP_USED}M"
if [ "${FREE_GB:-0}" -lt 50 ]; then
    say "refusing: under 50 GB free.  Close memory hogs (or drop sushi) and re-run."
    exit 1
fi

# 2) the engine binary must be the freshly built one
[ -x "$EXE" ] || { say "build first: cd build-metal && ninja"; exit 1; }

# 3) the test: solo, then the same prompts batched; every slot must match its solo tokens
say "running: batch $BATCH, $N prompts, $MAXNEW tokens each  (log: $LOG)"
say "the engine loads ~43 GB and warms the window: the first lines take a couple of minutes."
# STRATA_PARK_DEBUG=1: every window pool and every failed snapshot check prints its fields (the admission
# failures and park bugs on this box were always a metadata question; one line in the log beats a guess)
export STRATA_PARK_DEBUG=1
# STRATA_BATCH_VERIFY=1: at every admission the window byte-compares each slot region with its source; the
# first bad byte names the buffer and the layer (metal/src/engine/window.cpp check_region)
export STRATA_BATCH_VERIFY=1
BATCH_TEST_LOG="$OUT/engine.log" python3 tools/batch_test.py \
    --exe "$EXE" --config "$CFG" --batch "$BATCH" --n "$N" --max-new "$MAXNEW" \
    --dump "$OUT/tokens.json" 2>&1 | tee "$LOG"
rc=$?
say "batch_test exit $rc (0: every slot IDENTICAL to solo; 2: at least one DIFFERS)"

# 4) what the window did: the batch window's build line, the region line, any refusal
grep -E "batch slot region|batch window built|slot sessions|--batch is off|not two slot" "$LOG" "$OUT/engine.log" 2>/dev/null | sort -u | sed 's/^/    /'
# the STRATA_BATCH_VERIFY verdict per admitted slot: "all match" or the first bad byte of a named buffer
grep -E "all match|MISMATCH|first diff|ple_prev:" "$OUT/engine.log" 2>/dev/null | sed 's/^/    VERIFY: /'
grep -E "error|stub|exit 86|assert" "$OUT/engine.log" 2>/dev/null | head -5 | sed 's/^/    ENGINE: /'

say "artifacts: $LOG, $OUT/engine.log, $OUT/tokens.json (solo and batch token lists)"
say "done.  It is safe to restart sushi now."
exit $rc
