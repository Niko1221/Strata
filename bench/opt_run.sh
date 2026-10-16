#!/bin/bash
# Optimization A/B harness: the user's IQ3_S text config is the fixed base; the caller's flags override.
#   ./opt_run.sh <tag> [flags...]   -> bench/opt/<tag>.log ; prints prefill + decode
# Base mirrors run_ab.sh's COMMON minus the two knobs under test (--pcie-frac, --vram-reserve-mib),
# which the caller must supply, so a run can OMIT --pcie-frac to enable the auto policy.
set -u
ROOT=/g/01_Projects/Strata
DATA=/d/25_Strata/Strata-main/Strata-data
EXE=$ROOT/build-cuda/strata.exe
MAXNEW=${MAXNEW:-128}
PROMPT=${PROMPT:-$ROOT/bench/prompt_zh.txt}
OUT=$ROOT/bench/opt
mkdir -p "$OUT"
tag="$1"; shift

taskkill //F //IM strata.exe >/dev/null 2>&1
for _ in $(seq 1 90); do
  tasklist //FI "IMAGENAME eq strata.exe" 2>/dev/null | grep -qi strata.exe || break
  sleep 1
done
sleep 2

BASE=(
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
  --spec-min-p 0.70
  --tokens-file "$PROMPT"
  --max-new "$MAXNEW"
)
LOG=$OUT/$tag.log
echo "=== $tag : extra=[$*] -> $LOG"
"$EXE" "${BASE[@]}" "$@" > "$LOG" 2>&1
echo "exit=$?"
grep -E "^(prefill|decode) " "$LOG" | sed 's/^/  /'
grep -E "expert cache [[:digit:]]+ slots|CPU experts" "$LOG" | sed 's/^/  /' | head -3
taskkill //F //IM strata.exe >/dev/null 2>&1
echo "=== done"
