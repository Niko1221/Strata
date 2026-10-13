#!/bin/bash
# Grid over the knobs that cost nothing to change.  Everything else (the engine, the pack, the tier)
# is held fixed, so any difference in t/s is the knob.
#   ./run_grid.sh <tag> [extra args...]
# Interleaved: a single pass over the grid drifts with the machine, so read the pair, not the number.
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
)

tag="$1"; shift
LOG="$OUT/$tag.log"
# a stale engine still holds 60 GB of the arena; a new one either waits for it or lands in the swap
taskkill //F //IM strata.exe >/dev/null 2>&1
for _ in $(seq 1 60); do
  tasklist //FI "IMAGENAME eq strata.exe" 2>/dev/null | grep -qi strata.exe || break
  sleep 1
done
sleep 2

echo "=== grid $tag  ${*:-(defaults)}"
"$EXE" "${COMMON[@]}" "$@" > "$LOG" 2>&1
grep -E "^(speculation|verify window|pool multi|dispatch|adaptive tier|decode|prefill)" "$LOG" | sed 's/^/    /'
