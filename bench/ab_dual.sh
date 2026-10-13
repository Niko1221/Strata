#!/bin/bash
# 严格对照：同一个引擎、同一份prompt、同样的token 数，唯一差别是 --peer-ocl 是否挂 MI50。
#   ./ab_dual.sh single     -> 单卡 5060Ti（无 peer）
#   ./ab_dual.sh dual       -> 双卡 5060Ti + MI50
set -u
ROOT=/g/01_Projects/Strata
DATA=/d/25_Strata/Strata-main/Strata-data
EXE=$ROOT/build-cuda/strata.exe
MAXNEW=${MAXNEW:-160}

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
  --pool-workers 12
  --tokens-file "$ROOT/bench/prompt_zh.txt"
  --max-new "$MAXNEW"
)

arm="$1"; shift
case "$arm" in
  dual)   LOG=$ROOT/bench/ab-dual.log;   EXTRA=(--peer-ocl 0) ;;
  single) LOG=$ROOT/bench/ab-single.log; EXTRA=() ;;
  *) echo "usage: $0 single|dual" >&2; exit 2 ;;
esac

echo "=== $arm max-new=$MAXNEW ${EXTRA[*]:-} -> $LOG"
"$EXE" "${COMMON[@]}" "${EXTRA[@]}" "$@" > "$LOG" 2>&1
echo "exit=$?"
