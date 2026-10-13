#!/bin/bash
# STRATA_OCL_ROWS sweep (4/8/16) on the packed-input build.  rows/group is how many weight rows one
# work-group reads before it retires; at WG=64 a group touches only 4 rows, so the barrier tree and the
# scheduler overhead are spread over very few bytes.  Bigger groups amortise - IF occupancy survives.
# Sequential, one run at a time: the numbers are only comparable on a quiet machine.
set -u
ROOT=/g/01_Projects/Strata
DATA=/d/25_Strata/Strata-main/Strata-data
EXE=$ROOT/build-cuda/strata.exe
MAXNEW=96
mkdir -p "$ROOT/bench/grid"

taskkill //F //IM strata.exe >/dev/null 2>&1
sleep 3

for R in "$@"; do
  LOG=$ROOT/bench/grid/rows$R.log
  echo "=== STRATA_OCL_ROWS=$R -> $LOG"
  STRATA_OCL_STATS=1 STRATA_OCL_ROWS=$R "$EXE" \
    --pack "$DATA/packs/iq3_s" \
    --native "$DATA/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" \
    --ple-gguf "$DATA/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" \
    --expert-profile "$ROOT/data/expert-profile.bin" \
    --expert-cache auto --prefill auto --spec 4 \
    --mtp "$DATA/mtp/rt" --max-context 262144 \
    --kv int8 --kv-resident 32768 --vram-reserve-mib 700 --pcie-frac 0.00 \
    --spec-min-p 0.70 --tokens-file "$ROOT/bench/prompt_zh.txt" --max-new "$MAXNEW" \
    --pool-workers 12 --peer-ocl 0 > "$LOG" 2>&1
  grep -aE "rows/group" "$LOG" | head -1
  grep -a "^decode" "$LOG"
  taskkill //F //IM strata.exe >/dev/null 2>&1
  for _ in $(seq 1 90); do
    tasklist //FI "IMAGENAME eq strata.exe" 2>/dev/null | grep -qi strata.exe || break
    sleep 1
  done
  sleep 3
done
echo "=== sweep done"
