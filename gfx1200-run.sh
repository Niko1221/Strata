#!/usr/bin/env bash
# gfx1200 (RX 9060 XT) tuned launcher for the HIP backend.
#
# Paths resolve relative to the repository root; override any location through the environment:
#   STRATA_PACK, STRATA_NATIVE_GGUF, STRATA_PLE_GGUF, STRATA_PROFILE, STRATA_MTP_RT
#
# Measured (IQ1_M Coder, 2374-token prompt, 128 new): decode 31.0 tok/s, prefill 541 tok/s.
# Long context: append --max-context 262144 --kv int8 --kv-resident 65536 --prefill 16384
#   (64K prompt: 761 tok/s prefill, 26.3 decode; 128K: 637/22.7. See docs/AMD_HIP_GFX1200.md.)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export STRATA_HIPBLASLT_TUNING="${STRATA_HIPBLASLT_TUNING:-$ROOT/tools/hip/gfx1200-hipblaslt-100202.txt}"
export STRATA_PREFILL_MMQ="${STRATA_PREFILL_MMQ:-1}"

PACK="${STRATA_PACK:-$ROOT/data/packs/iq1_m}"
NATIVE="${STRATA_NATIVE_GGUF:?set STRATA_NATIVE_GGUF to the model's shard 1}"
PLE="${STRATA_PLE_GGUF:?set STRATA_PLE_GGUF to the model's shard 2 (the PLE table)}"
PROFILE="${STRATA_PROFILE:-$ROOT/data/expert-profile-coder.bin}"
MTP="${STRATA_MTP_RT:?set STRATA_MTP_RT to the MTP runtime directory}"

exec "$ROOT/build-hip/strata" \
  --pack "$PACK" \
  --native "$NATIVE" \
  --ple-gguf "$PLE" \
  --mmap-experts --resident-cpu-experts \
  --kv int8 --pcie-frac 0 \
  --expert-profile "$PROFILE" \
  --expert-cache auto --adapt-every 0 \
  --prefill 2560 --spec 4 --spec-min-p 0.5 \
  --mtp "$MTP" \
  --max-context 8192 --vram-reserve-mib 1792 \
  "$@"
