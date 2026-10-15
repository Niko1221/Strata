#!/bin/zsh
# KV bytes and arena at 32768 and 65536 contexts (pairs {q4_0, int8}, --max-new 8, short prompt).
# The KV allocation itself is inside the engine's "N GB allocated before the arena" line (the q4-vs-int8
# delta there is the KV saving); the freed memory flows into the "expert arena N GB, M slots" line.
# usage: ./kvmem.sh          (writes ./logs/kvmem-*.log, prints the init lines)
set -u
cd /Users/berton/Github/AI-Infra/Strata-metal-kv-q4 || exit 9
here=bench/results/2026-10-15-mac-kv-q4
mkdir -p $here/logs

# one run: $1 tag, $2 kv, $3 max-context
run() {
  tag=$1; kv=$2; ctx=$3
  STRATA_METAL_STREAM=1 STRATA_METAL_STUBS=log ./build-metal/strata \
    --pack /Users/berton/Github/AI-Infra/Strata-data/packs/q2_0 \
    --native /Users/berton/Github/AI-Infra/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
    --ple-gguf /Users/berton/Github/AI-Infra/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
    --expert-profile data/expert-profile.bin \
    --expert-cache 64 --prefill auto --spec 2 --suffix-draft 0 \
    --max-context "$ctx" --mmap-experts --kv "$kv" \
    --tokens 1000,2000,3000,4000 --max-new 8 > $here/logs/kvmem-$tag.log 2>&1
  echo "exit=$? tag=$tag"
  grep -h "decode window on Apple\|allocated before the arena\|planned resident footprint\|expert arena\|RAM .* GB (" $here/logs/kvmem-$tag.log | head -5
}

run 32768-q4   q4_0 32768
run 32768-int8 int8 32768
run 65536-q4   q4_0 65536
run 65536-int8 int8 65536
