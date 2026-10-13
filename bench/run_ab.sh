#!/bin/bash
# A/B the second expert tier: the same engine, the same arguments, everything identical except the tier.
#   ./run_ab.sh base            -> bench/run-base.log
#   ./run_ab.sh ocl             -> bench/run-ocl.log   (--peer-ocl 0)
# MAXNEW=128 OCLIDX=0 override the token count and the OpenCL ordinal.
set -u
ROOT=/g/01_Projects/Strata
DATA=/d/25_Strata/Strata-main/Strata-data
EXE=$ROOT/build-cuda/strata.exe
MAXNEW=${MAXNEW:-128}

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

arm="$1"; shift
case "$arm" in
  ocl)  LOG=$ROOT/bench/run-ocl.log;  EXTRA=(--peer-ocl "${OCLIDX:-0}") ;;
  base) LOG=$ROOT/bench/run-base.log; EXTRA=() ;;
  *) echo "usage: $0 base|ocl" >&2; exit 2 ;;
esac

echo "=== $arm: max-new $MAXNEW ${EXTRA[*]:-} -> $LOG"
"$EXE" "${COMMON[@]}" "${EXTRA[@]}" "$@" > "$LOG" 2>&1
echo "exit=$?"
