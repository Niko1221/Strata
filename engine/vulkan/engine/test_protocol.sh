#!/usr/bin/env bash
# Smoke test for the native Vulkan engine: no server, no HTTP, no llama.cpp.
# Requires the GGUF, the built binary and the checked-in SPIR-V shaders.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
ENGINE="$ROOT/engine/vulkan/engine/strata-vulkan-engine"
MODEL="${MODEL:-/home/g/Code/llama.cpp/models/Qwen3.8-35B-A3B-Q4_K_M.gguf}"

if [ ! -x "$ENGINE" ]; then
    echo "missing $ENGINE - run bash engine/vulkan/engine/build.sh first" >&2
    exit 1
fi
[ -e "$MODEL" ] || { echo "missing: $MODEL" >&2; exit 1; }

OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT

printf 'EXPERT 0 0\nSHEXP 0\nVSTAT\nVRAM\nQUIT\n' | "$ENGINE" \
    --model "$MODEL" --ctx 4096 --threads 8 \
    --q4-shader "$ROOT/engine/vulkan/shaders/expert_q4_k_gemv.comp.spv" \
    --down-shader "$ROOT/engine/vulkan/shaders/expert_q6_k_gemv.comp.spv" \
    --silu-shader "$ROOT/engine/vulkan/shaders/expert_silu_mul.comp.spv" \
    --router-gemv-shader "$ROOT/engine/vulkan/shaders/router_gemv.comp.spv" \
    --router-topk-shader "$ROOT/engine/vulkan/shaders/router_topk.comp.spv" \
    > "$OUT" 2>/dev/null

grep -q '^READY ' "$OUT" || { echo "FAIL: no READY"; exit 1; }
grep -q '^EXPERT 0 0 sum=' "$OUT" || { echo "FAIL: no EXPERT reply"; exit 1; }
grep -q '^SHEXP 0 sum=' "$OUT" || { echo "FAIL: no SHEXP reply"; exit 1; }
grep -q '^VSTAT hits=' "$OUT" || { echo "FAIL: no VSTAT"; exit 1; }
grep -q '^VRAM expert_slots=' "$OUT" || { echo "FAIL: no VRAM"; exit 1; }
# The native engine reports free shared memory (MemAvailable capped at the
# DEVICE_LOCAL heap), not the old placeholder 0, on a machine with /proc.
if [ -r /proc/meminfo ]; then
    FREE_MIB="$(sed -n 's/^VRAM .* vram_free_mib=\([0-9][0-9]*\).*/\1/p' "$OUT" | head -1)"
    [ -n "$FREE_MIB" ] && [ "$FREE_MIB" -gt 0 ] || { echo "FAIL: vram_free_mib is not a measurement"; exit 1; }
fi

echo "PASS"
grep -E '^(READY|EXPERT|SHEXP|VSTAT|VRAM) ' "$OUT"
