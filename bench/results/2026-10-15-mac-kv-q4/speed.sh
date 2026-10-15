#!/bin/zsh
# Speed pairs (same day, back to back): decode and prompt-prefill, only --kv differs.
# Same engine args as strata-q2_0.json (the M0/M1 baseline template; /tmp/m1/g.sh's shape):
#   pack/native/ple-gguf/expert-profile, --expert-cache 64 --prefill auto --spec 2 --suffix-draft 0
#   --max-context 32768 --mmap-experts, env STRATA_METAL_STREAM=1 STRATA_METAL_STUBS=log
# usage: ./speed.sh            (writes ./logs/speed-*.log, prints the tok/s lines)
set -u
cd /Users/berton/Github/AI-Infra/Strata-metal-kv-q4 || exit 9
here=bench/results/2026-10-15-mac-kv-q4
mkdir -p $here/logs

args_base=(--pack /Users/berton/Github/AI-Infra/Strata-data/packs/q2_0
      --native /Users/berton/Github/AI-Infra/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
      --ple-gguf /Users/berton/Github/AI-Infra/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
      --expert-profile data/expert-profile.bin
      --expert-cache 64 --prefill auto --spec 2 --suffix-draft 0
      --max-context 32768 --mmap-experts)

# one run: $1 tag, $2 kv (or "-": no --kv flag at all), then the extra args
run() {
  tag=$1; kv=$2; shift 2
  a=("${args_base[@]}")
  [[ "$kv" != "-" ]] && a+=(--kv "$kv")
  a+=("$@")
  STRATA_METAL_STREAM=1 STRATA_METAL_STUBS=log ./build-metal/strata $a > $here/logs/speed-$tag.log 2>&1
  echo "exit=$? tag=$tag"
  grep -h "^decode \|^prefill " $here/logs/speed-$tag.log
}

# decode: short prompt (4 ids), 64 new tokens, three arms
run dec-q4    q4_0 --tokens 1000,2000,3000,4000 --max-new 64
run dec-int8  int8 --tokens 1000,2000,3000,4000 --max-new 64
run dec-none  -    --tokens 1000,2000,3000,4000 --max-new 64

# prompt: 2200-id prompt (ids 100..2299, data/long-prompt-2200.ids), 16 new tokens, fast path vs fast path
run pre-q4    q4_0 --tokens-file $here/data/long-prompt-2200.ids --max-new 16
run pre-int8  int8 --tokens-file $here/data/long-prompt-2200.ids --max-new 16
