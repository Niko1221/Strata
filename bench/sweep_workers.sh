#!/bin/bash
# Why is the CPU expert pool still 46 ms/round when MI50 has already cut its work by 85%?
#
# 48 pool calls per round, ~0.96 ms each, almost independent of how many experts the call actually has
# (9.88 -> 1.50 distinct per layer moved it 59.5 -> 46.0).  That is a fixed per-call cost, not bandwidth.
# Hypothesis: the 21 pool workers, the host thread that walks the ring protocol, and the AMD driver's
# OpenCL submission threads are all on the same 22 physical cores, so FEWER workers should be FASTER.
#
# This sweeps the worker count on the dual-card arm.  The host core is left alone (pool-affinity all).
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

for w in "$@"; do
  kill_engine
  LOG="$OUT/w$w.log"
  echo "=== pool-workers $w  -> $LOG"
  "$EXE" "${COMMON[@]}" --pool-workers "$w" > "$LOG" 2>&1
  grep -E "^(speculation|verify window|pool multi|dispatch|decode)" "$LOG" | sed 's/^/    /'
done
kill_engine
echo "=== sweep done"
