#!/bin/sh
# Entry point for the Strata container. The engine is compiled during docker
# build and lives in the image, so the first start only downloads the model.
# The install config is kept on the /data volume so a recreated container skips
# the setup pass and goes straight to serving.
set -e
cd /opt/strata || exit 1

STRATA_DATA="${STRATA_DATA:-/data}"
FAMILY="${FAMILY:-qwen}"
MODEL="${MODEL:-IQ2_XS}"
CONTEXT="${CONTEXT:-32768}"
VISION="${VISION:-no}"          # no | yes | cpu (the image encoder on the CPU)
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-}"
KV="${KV:-}"                    # int8 | q4_0 | k8v4; empty: setup.py's own default (int8)
GPUS="${GPUS:-}"                # "0,2" or "all": one model across several cards (docs/MULTI_GPU.md)
GPU="${GPU:-}"                  # one card, numbered as nvidia-smi numbers them
LAYER_SPLIT="${LAYER_SPLIT:-}"  # with GPUS: where each later card's layers start (default: auto)
LOW_RAM="${LOW_RAM:-auto}"      # on: the experts come from the pack's experts.bin, not from RAM
GGUF_DIR="${GGUF_DIR:-}"        # a mounted folder with GGUF files you already have: no download
RESIDENT_BUDGET_GIB="${RESIDENT_BUDGET_GIB:-}"   # UD-Q4_K_XL: GiB of experts kept in RAM (default: setup's pick)
KV_STREAMING="${KV_STREAMING:-}" # auto | on | off; empty: setup.py's own default (auto)

# setup.py starts the newest strata-*.json it finds, so link in exactly the one
# this family and model were set up with. The config is the recorded output of
# that setup (the pack, the profile, the quant, the KV decision), not settings
# the entry point could rebuild from env vars. qwen has an empty family tag.
case "$FAMILY" in qwen) prefix="" ;; *) prefix="${FAMILY}-" ;; esac
tag="${prefix}$(printf '%s' "$MODEL" | tr 'A-Z' 'a-z')"
cfg="$STRATA_DATA/config/strata-$tag.json"
mkdir -p "$STRATA_DATA/config"

# REINSTALL is only needed to change settings for a model that is already set up
# (context, vision, KV, host, api_key). Switching between models already on the
# volume needs no setup pass: their config is already there.
#
# KV / GPUS / GPU / LAYER_SPLIT are passed only when set, so an unset one keeps
# setup.py's own default. LOW_RAM is always passed: setup.py measures the PC's RAM
# from /proc/meminfo, which in a container is the host's total, not the container's
# limit, so a memory-capped container has to ask for the low-RAM mode itself.
if [ "${REINSTALL:-0}" = "1" ] || [ ! -f "$cfg" ]; then
  if [ -n "$GGUF_DIR" ]; then
    echo "Setting up $tag from the GGUF files in $GGUF_DIR (the engine is already in the image)."
  else
    echo "Setting up $tag: downloading the model (~70 GB; the engine is already in the image)."
  fi
  set -- --family "$FAMILY" --model "$MODEL" --context "$CONTEXT" --vision "$VISION" \
    --data-dir "$STRATA_DATA" --host "$HOST" --api-key "$API_KEY" \
    --port "$PORT" --no-start --low-ram "$LOW_RAM"
  if [ -n "$KV" ]; then set -- "$@" --kv "$KV"; fi
  if [ -n "$GPUS" ]; then set -- "$@" --gpus "$GPUS"; fi
  if [ -n "$GPU" ]; then set -- "$@" --gpu "$GPU"; fi
  if [ -n "$LAYER_SPLIT" ]; then set -- "$@" --layer-split "$LAYER_SPLIT"; fi
  if [ -n "$GGUF_DIR" ]; then set -- "$@" --gguf-dir "$GGUF_DIR"; fi
  if [ -n "$RESIDENT_BUDGET_GIB" ]; then set -- "$@" --resident-budget-gib "$RESIDENT_BUDGET_GIB"; fi
  if [ -n "$KV_STREAMING" ]; then set -- "$@" --kv-streaming "$KV_STREAMING"; fi
  .venv/bin/python setup.py --setup --yes "$@"
  [ -e "/opt/strata/strata-$tag.json" ] && { cmp -s "/opt/strata/strata-$tag.json" "$cfg" || cp -f "/opt/strata/strata-$tag.json" "$cfg"; }
else
  # #1244: the copy on the volume is the one that counts, so a regular file left in /opt/strata by an earlier setup
  # (or by an image built with one) must not stand in for it: edits to /data/config would be ignored
  ln -sfn "$cfg" "/opt/strata/strata-$tag.json"
fi

# Later starts skip straight here: setup.py finds the installed config and
# launches serve/server.py (OpenAI- and Anthropic-compatible API on :8080).
# GPUS / GPU / LAYER_SPLIT are repeated on purpose. Given at the start they pin the
# cards for this model, and setup.py saves them in its config; without them a config
# that names one card is offered once to a pair, on its own, when the host has two
# cards that can share the model (setup.py's offer_together, docs/MULTI_GPU.md).
set -- --port "$PORT"
if [ -n "$GPUS" ]; then set -- "$@" --gpus "$GPUS"; fi
if [ -n "$GPU" ]; then set -- "$@" --gpu "$GPU"; fi
if [ -n "$LAYER_SPLIT" ]; then set -- "$@" --layer-split "$LAYER_SPLIT"; fi
exec .venv/bin/python setup.py "$@"
