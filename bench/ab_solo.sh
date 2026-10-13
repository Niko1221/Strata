#!/bin/bash
# A/B the small-batch solo path in run_phase: the SAME binary, the same arguments, the only difference is
# STRATA_POOL_SOLO.  Interleaved on purpose - this machine drifts between a fast and a slow state, so a
# single pass over two arms measures the machine as much as the change (see pool.hpp's note on #29).
set -u
ROOT=/g/01_Projects/Strata
DATA=/d/25_Strata/Strata-main/Strata-data
EXE=$ROOT/build-cuda/strata.exe
MAXNEW=${MAXNEW:-128}
WORKERS=${WORKERS:-12}
OUT=$ROOT/bench/grid
mkdir -p "$OUT"

COMMON=(
  --pack "$DATA/packs/iq3_s"
  --native "$DATA/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
  --ple-gguf "$DATA/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"
  --expert-profile "$ROOT/data/expert-profile.bin"
  --expert-cache auto
  --prefill auto
  --spec 4
  --mtp "$DATA/mtp/rt"
  --max-context 262144
  --kv int8
  --kv-resident 32768
  --vram-reserve-mib 700
  --pcie-frac 0.00
  --spec-min-p 0.70
  --tokens-file "$ROOT/bench/prompt_zh.txt"
  --max-new "$MAXNEW"
  --pool-workers "$WORKERS"
  --peer-ocl 0
)

kill_engine() {
  taskkill //F //IM strata.exe >/dev/null 2>&1
  for _ in $(seq 1 90); do
    tasklist //FI "IMAGENAME eq strata.exe" 2>/dev/null | grep -qi strata.exe || break
    sleep 1
  done
  sleep 3
}

# the pair is run twice, in ABBA order, so a drift that favours the first arm is visible as an
# arm-order effect rather than as a result
for solo in 0 1 1 0; do
  kill_engine
  LOG="$OUT/solo$solo-$solo.log"
  echo "=== STRATA_POOL_SOLO=$solo -> $LOG"
  STRATA_POOL_SOLO=$solo "$EXE" "${COMMON[@]}" > "$LOG" 2>&1
  grep -E "^(speculation|verify window|pool multi|dispatch|decode)" "$LOG" | sed 's/^/    /'
done
kill_engine
echo "=== A/B done"
