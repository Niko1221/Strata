#!/bin/bash
# Final A/B for the wave-shape change: SAME binary, STRATA_OCL_WG=32 (the reference's 32-lane tree) vs 64
# (native wave64 for the 80-call gate/up rows).  ABBA order against machine drift.  128 tokens per arm.
set -u
ROOT=/g/01_Projects/Strata
DATA=/d/25_Strata/Strata-main/Strata-data
EXE=$ROOT/build-cuda/strata.exe
MAXNEW=${MAXNEW:-128}
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
  --pool-workers 12
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

for wg in 32 64 64 32; do
  kill_engine
  LOG="$OUT/wg$wg-$wg.log"
  echo "=== STRATA_OCL_WG=$wg -> $LOG"
  STRATA_OCL_WG=$wg "$EXE" "${COMMON[@]}" > "$LOG" 2>&1
  grep -aE "^(verify window|pool multi|dispatch|decode)" "$LOG" | sed 's/^/    /'
  grep -a "device's own time" "$LOG" | tail -1 | sed 's/^/    /'
done
kill_engine
echo "=== WG A/B done"
