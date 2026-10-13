#!/bin/bash
# One dual-card run with STRATA_OCL_STATS=1: the per-kernel breakdown of where MI50's time goes.
# This is the measurement that decides which of the three OpenCL passes to attack, so it runs alone
# (no sweep beside it) - the device-side timestamps are only comparable on a quiet machine.
set -u
ROOT=/g/01_Projects/Strata
DATA=/d/25_Strata/Strata-main/Strata-data
EXE=$ROOT/build-cuda/strata.exe
MAXNEW=${MAXNEW:-96}
LOG=$ROOT/bench/grid/nofold.log
mkdir -p "$ROOT/bench/grid"

taskkill //F //IM strata.exe >/dev/null 2>&1
for _ in $(seq 1 90); do
  tasklist //FI "IMAGENAME eq strata.exe" 2>/dev/null | grep -qi strata.exe || break
  sleep 1
done
sleep 3

echo "=== STRATA_OCL_STATS=1 -> $LOG"
STRATA_OCL_STATS=1 STRATA_OCL_NOFOLD=1 "$EXE" \
  --pack "$DATA/packs/iq3_s" \
  --native "$DATA/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" \
  --ple-gguf "$DATA/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" \
  --expert-profile "$ROOT/data/expert-profile.bin" \
  --expert-cache auto --prefill auto --spec 4 \
  --mtp "$DATA/mtp/rt" --max-context 262144 \
  --kv int8 --kv-resident 32768 --vram-reserve-mib 700 --pcie-frac 0.00 \
  --spec-min-p 0.70 --tokens-file "$ROOT/bench/prompt_zh.txt" --max-new "$MAXNEW" \
  --pool-workers 12 --peer-ocl 0 > "$LOG" 2>&1

echo "--- the device's own time per layer"
grep -E "^\[OCL\]" "$LOG" | tail -12
echo "--- engine totals"
grep -E "^(verify window|pool multi|dispatch|decode)" "$LOG"

taskkill //F //IM strata.exe >/dev/null 2>&1
echo "=== done"
