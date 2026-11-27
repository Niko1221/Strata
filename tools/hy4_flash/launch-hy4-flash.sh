#!/usr/bin/env bash
# Start the hy4-flash service with a GGUF, a native SQ model, or a model directory.
# Drafting follows the launcher defaults. Examples with MODEL set:
#   MTP=on DRAFT_TOKENS=3 ./tools/hy4_flash/launch-hy4-flash.sh
#   DFLY_MODEL=draft-q8_0.gguf DRAFT_TOKENS=4 ./tools/hy4_flash/launch-hy4-flash.sh
set -euo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$DIR"

# Bypass HTTP proxies for local API and monitor connections.
NO_PROXY_MERGED="${no_proxy:-}${no_proxy:+,}${NO_PROXY:-}"
NO_PROXY_MERGED="${NO_PROXY_MERGED:+$NO_PROXY_MERGED,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY_MERGED" NO_PROXY="$NO_PROXY_MERGED"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x .venv/bin/python ]]; then PYTHON_BIN=.venv/bin/python; else PYTHON_BIN=python3; fi
fi

for arg in "$@"; do
    case "$arg" in
        -h|--help) exec "$PYTHON_BIN" "$DIR/terminal/strata.py" launch --help ;;
    esac
done

ARGS=()
if [[ -n "${MODEL:-}" ]]; then
    ARGS+=(--model-path "$MODEL")
elif [[ " $* " != *" --model-path "* && " $* " != *" --status "* && " $* " != *" --stop "* ]]; then
    echo "Usage: MODEL=/path/to/model_dir_or_shard1.gguf $0 [options]" >&2
    exit 2
fi

[[ -n "${PACK:-}" ]]         && ARGS+=(--pack-path "$PACK")
[[ -n "${STRATA_BIN:-}" ]]   && ARGS+=(--engine-path "$STRATA_BIN")
[[ -n "${DRAFTING:-}" ]]     && ARGS+=(--drafting "$DRAFTING")
[[ -n "${MTP:-}" ]]          && ARGS+=(--mtp "$MTP")
[[ -n "${DFLY_MODEL:-}" ]]   && ARGS+=(--dfly-model "$DFLY_MODEL")
[[ -n "${DRAFT_TOKENS:-}" ]] && ARGS+=(--draft-tokens "$DRAFT_TOKENS")
[[ -n "${EXPERT_VRAM:-}" ]]  && ARGS+=(--expert-vram "$EXPERT_VRAM")
[[ -n "${KV:-}" ]]           && ARGS+=(--kv "${KV/q8_0/int8}")
[[ -n "${CPU_MISS_MAX:-}" ]] && ARGS+=(--set "STRATA_HY4_FLASH_CPU_MISS_MAX=$CPU_MISS_MAX")
[[ -n "${GPU_SAMPLE:-}" ]]   && ARGS+=(--set "STRATA_HY4_FLASH_GPU_SAMPLE=$GPU_SAMPLE")

exec "$PYTHON_BIN" "$DIR/terminal/strata.py" launch \
    --family hy4-flash \
    --gpu "${GPU:-0}" \
    --context "${CTX:-196608}" \
    --port "${PORT:-8080}" \
    --save-preset "${PRESET:-hy4-flash}" \
    --yes \
    "${ARGS[@]}" \
    "$@"
