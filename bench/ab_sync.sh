#!/bin/bash
# ABBA of the read-back forms on ONE binary: async (in-queue DMA, default) vs sync
# (STRATA_OCL_SYNCREAD=1: finish waits for the down kernel, THEN issues a blocking copy).
# The async form pays a measured ~0.15 ms/layer device-side gap before the copy starts.
set -u
ROOT=/g/01_Projects/Strata
DATA=/d/25_Strata/Strata-main/Strata-data
EXE=$ROOT/build-cuda/strata.exe
MAXNEW=${MAXNEW:-96}
G=$ROOT/bench/grid
mkdir -p "$G"

run() { # $1 tag  $2 extra env
  local tag=$1 env=$2
  taskkill //F //IM strata.exe >/dev/null 2>&1
  for _ in $(seq 1 90); do
    tasklist //FI "IMAGENAME eq strata.exe" 2>/dev/null | grep -qi strata.exe || break
    sleep 1
  done
  sleep 3
  echo "=== arm $tag ($env)"
  env STRATA_OCL_STATS=1 $env "$EXE" \
    --pack "$DATA/packs/iq3_s" \
    --native "$DATA/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" \
    --ple-gguf "$DATA/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" \
    --expert-profile "$ROOT/data/expert-profile.bin" \
    --expert-cache auto --prefill auto --spec 4 \
    --mtp "$DATA/mtp/rt" --max-context 262144 \
    --kv int8 --kv-resident 32768 --vram-reserve-mib 700 --pcie-frac 0.00 \
    --spec-min-p 0.70 --tokens-file "$ROOT/bench/prompt_zh.txt" --max-new "$MAXNEW" \
    --pool-workers 12 --peer-ocl 0 > "$G/sync-$tag.log" 2>&1
  grep -aE "finish 0\.|^decode" "$G/sync-$tag.log" | tail -2
}

run A ""
run B "STRATA_OCL_SYNCREAD=1"
run A2 ""
run B2 "STRATA_OCL_SYNCREAD=1"

echo "===== summary"
for t in A B A2 B2; do
  echo "--- $t"
  grep -aE "^\[OCL\]   finish" "$G/sync-$t.log" | tail -1
  grep -a "^decode" "$G/sync-$t.log"
done
taskkill //F //IM strata.exe >/dev/null 2>&1
echo "=== done"
