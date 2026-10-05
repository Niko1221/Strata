#!/usr/bin/env bash
# Container entrypoint for strata-hip: guard the environment, honour the VRAM budget, make sure the
# model + pack exist (fetching them into the mounted Hugging Face cache if not), write the engine
# config, start the server.
#
# Flow: guard -> device -> VRAM budget -> model/pack (docker/bootstrap-model.sh) -> config -> serve
# Tuning is all by environment; see docker/README.md for the list.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # /opt/strata/docker
REPO="$(dirname "$DIR")"                                     # /opt/strata
PY="${PYTHON:-python3}"

log() { printf 'strata-hip: %s\n' "$*"; }
die() { printf 'strata-hip: ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. environment guards
# This host's own lesson (see /etc/profile.d/amd_gpu.sh on the reference machine): ROCm 7.x handles
# gfx1101 natively, and pretending to be gfx1100 makes the gfx1101 code object fail to load.  Neither
# of these may be inherited silently from a login shell or a leftover docker -e.
if [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ] && [ "${STRATA_ALLOW_HSA_OVERRIDE:-0}" != "1" ]; then
  die "HSA_OVERRIDE_GFX_VERSION='$HSA_OVERRIDE_GFX_VERSION' is set. Strata builds for the real gfx
       target; on gfx1101 that override makes the code object fail to load (llama.cpp segfaults).
       Remove it, or set STRATA_ALLOW_HSA_OVERRIDE=1 if you know why."
fi
unset HSA_OVERRIDE_GFX_VERSION
if [ -n "${AMD_SERIALIZE_KERNEL:-}" ]; then
  log "unsetting AMD_SERIALIZE_KERNEL=$AMD_SERIALIZE_KERNEL (a debug switch: host-syncs every launch)"
  unset AMD_SERIALIZE_KERNEL
fi
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-1}"

# ---------------------------------------------------------------- 2. the device
ARCH="$("$PY" "$DIR/hipinfo.py" --arch)" \
  || die "no AMD GPU visible: run with --device /dev/kfd --device /dev/dri/renderD<N> --group-add video --group-add render"
case "$ARCH" in
  gfx1100|gfx1101) ;;
  *) die "the GPU is $ARCH, and Strata's HIP backend builds for gfx1100/gfx1101 only.
          To support another card, widen cmake/hip_backend.cmake and src/core/device.cu first
          (docs/DOCKER_GFX1101_PLAN.md Part C) - then rebuild the image." ;;
esac
log "GPU (HIP index $HIP_VISIBLE_DEVICES):"
"$PY" "$DIR/hipinfo.py" | sed 's/^/  /'
# sysfs says a card is there; the engine's own probe says this binary's code object actually loads on
# it.  Worth the second: it costs a second now instead of failing after a 40-minute pack.
if [ -x /usr/local/bin/strata-device ]; then
  /usr/local/bin/strata-device > /tmp/strata-device.txt 2>&1 \
    || die "the engine cannot open this GPU - see below (wrong arch built in? a code object that will not load?)"
  head -5 /tmp/strata-device.txt | sed 's/^/  /'
fi

# ---------------------------------------------------------------- 3. the VRAM budget
export STRATA_VRAM_LATER_MIB="${STRATA_VRAM_LATER_MIB:-768}"
BUDGET_MIB="${STRATA_VRAM_BUDGET_MIB:-10240}"
SLACK_MIB="${STRATA_VRAM_SLACK_MIB:-256}"
[[ "$BUDGET_MIB" =~ ^[0-9]+$ ]] && [ "$BUDGET_MIB" -gt 1280 ] && [ "$BUDGET_MIB" -le 10240 ] \
  || die "VRAM budget must be 1281..10240 MiB; 10 GiB is the hard ceiling"
[[ "$SLACK_MIB" =~ ^[0-9]+$ ]] && [ "$SLACK_MIB" -ge 256 ] \
  || die "VRAM slack must be at least 256 MiB"
export STRATA_VRAM_BUDGET_MIB="$BUDGET_MIB" STRATA_VRAM_SLACK_MIB="$SLACK_MIB"
read -r VRAM_TOTAL VRAM_USED VRAM_FREE <<<"$("$PY" "$DIR/hipinfo.py" --vram)"
RESERVE_MIB="$("$PY" "$DIR/hipinfo.py" --reserve-mib --allocation-guard --budget-mib "$BUDGET_MIB" --slack-mib "$SLACK_MIB")"
LATER_MIB="$($PY -c 'import sys; sys.path.insert(0, "'"$DIR"'"); import hipinfo; print(hipinfo.LATER_MIB)')"
# hipMemGetInfo already excludes allocations, runtime reserve and slack from
# the 10 GiB ledger. Reserve only the later explicit allocations here.
GUI_SHARE=$((VRAM_TOTAL - BUDGET_MIB))
log "VRAM: ${BUDGET_MIB} MiB ceiling; desktop/GUI currently holds ${VRAM_USED} MiB"
log "  allocator preserves runtime reserve and ${SLACK_MIB} MiB slack"
log "  expert-cache sizing reserves ${RESERVE_MIB} MiB for later explicit allocations"

# ---------------------------------------------------------------- 4. the model, from the HF cache
MODEL="${STRATA_MODEL:-IQ3_XXS}"
export STRATA_MODEL="$MODEL"
export STRATA_HF_CACHE="${STRATA_HF_CACHE:-${HF_HUB_CACHE:-/hf-cache}}"
export STRATA_WORK="${STRATA_WORK:-/work}"
# Release facts for the defaults below come from the same resolver the container fetches with
# (paths may be empty here - --allow-missing; the keys are what this block uses).  If the model
# name is unknown to the resolver, bootstrap-model.sh is the failure authority, and the fallbacks
# below are today's pre-release-axis defaults.
eval "$({ "$PY" "$DIR/hfmodel.py" --model "$MODEL" --cache "$STRATA_HF_CACHE" --print shell \
         --allow-missing ${STRATA_HF_REPO:+--repo "$STRATA_HF_REPO"}; } 2>/dev/null)" || true
export STRATA_PACK_DIR="${STRATA_PACK_DIR:-$STRATA_WORK/packs/${STRATA_PACK_TAG:-}$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')}"
export STRATA_MTP="${STRATA_MTP:-$STRATA_WORK/mtp/rt}"
LOG="${STRATA_LOG:-$STRATA_WORK/logs/strata-hip.log}"
RUN_DIR="${STRATA_RUNTIME_DIR:-/run}"
# Window and output defaults, matched to the profile this engine is driven from
# (amanda/profiles/yolo-auto__yolo.json: contextWindow 262144, maxOutputTokens 32768, reasoning high).
# 131072 rather than the model's 262144 on purpose: KV + indexer keys cost 1.71 GB at 128k and 3.42 GB
# at 256k (measured, --kv int8), and that comes straight out of the expert cache - at 256k the cache
# would fall to ~800 experts and decode would crawl.  The model itself is fine with either.
MAX_CONTEXT="${STRATA_MAX_CONTEXT:-131072}"
[ "$MAX_CONTEXT" = 131072 ] || die "this deployment requires exactly 131072 tokens (128K context)"
MAX_TOKENS="${STRATA_MAX_TOKENS:-32768}"
# The model's template already defaults to xhigh, but a server-side default makes it a declared
# setting instead of a coincidence, and a client that names none gets the documented level.
REASONING="${STRATA_REASONING_EFFORT:-high}"
PREFILL="${STRATA_PREFILL:-2048}"
EXPERT_CACHE="${STRATA_EXPERT_CACHE:-800}"
[[ "$EXPERT_CACHE" = auto || "$EXPERT_CACHE" =~ ^[1-9][0-9]*$ ]] \
  || die "STRATA_EXPERT_CACHE must be auto or a positive slot budget"
KV="${STRATA_KV:-int8}"
SPEC="${STRATA_SPEC:-4}"
POOL_WORKERS="${STRATA_POOL_WORKERS:-0}"  # engine selects one worker per allowed physical core, excluding host
[[ "$POOL_WORKERS" =~ ^[0-9]+$ ]] || die "STRATA_POOL_WORKERS must be 0 (auto) or a positive integer"
MODEL_NAME="${STRATA_MODEL_NAME:-${STRATA_MODEL_NAME_DEFAULT:-qwen3.8-flash-next-$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')}}"
[ "${STRATA_VISION:-0}" = "1" ] && die "images/vision are NVIDIA-only on this backend (docs/AMD_HIP.md:101)"

# Check-and-fetch: downloads the quant into the mounted HF cache and builds the pack if either is
# missing (with a free-space gate first).  STRATA_AUTO_PREPARE=0 makes it a pure check that fails
# with the commands to run instead - what you want on a metered link.
log "checking the model and its pack ..."
"$PY" "$DIR/hfmodel.py" --model "$MODEL" --cache "$STRATA_HF_CACHE" --print available | sed 's/^/  /'
bash "$DIR/bootstrap-model.sh" "$([ "${STRATA_AUTO_PREPARE:-1}" = 1 ] && echo prepare || echo check)" \
  || die "the model/pack is not ready (see bootstrap above)"

# Where the engine's files ended up.  STRATA_NATIVE/STRATA_PLE_GGUF bypass the cache lookup for a
# GGUF that lives somewhere else entirely.
if [ -n "${STRATA_NATIVE:-}" ]; then
  NATIVE="$STRATA_NATIVE"; PLE="${STRATA_PLE_GGUF:-}"; REPO_ID="${STRATA_HF_REPO:-external}"
else
  eval "$("$PY" "$DIR/hfmodel.py" --model "$MODEL" --cache "$STRATA_HF_CACHE" --print shell \
          ${STRATA_HF_REPO:+--repo "$STRATA_HF_REPO"})"
  NATIVE="$STRATA_SHARD1"; PLE="${STRATA_PLE_FILE:-$STRATA_SHARD2}"; REPO_ID="$STRATA_REPO"
fi
[ -n "$NATIVE" ] && [ -e "$NATIVE" ] || die "shard 1 is still missing ('$NATIVE')"
[ -n "$PLE" ] && [ -e "$PLE" ] || die "the PLE table file is missing ('$PLE')"
case "$MODEL" in
  IQ1_M) EXPERT_PROFILE="${STRATA_EXPERT_PROFILE:-$REPO/data/expert-profile-coder.bin}" ;;
  *)     EXPERT_PROFILE="${STRATA_EXPERT_PROFILE:-$REPO/data/expert-profile.bin}" ;;
esac
if [ ! -f "$STRATA_MTP/experts.bin" ]; then
  SPEC=0
  log "no MTP draft layer at $STRATA_MTP: starting with --spec 0 (output will be slower than the" \
      "published numbers; see the bootstrap note above for why it is missing)"
fi
log "model $MODEL from $REPO_ID"
[ -n "${STRATA_LICENSE:-}" ] && log "  license: $STRATA_LICENSE"
log "  native=$NATIVE"
log "  ple=$PLE"
log "  pack=$STRATA_PACK_DIR  mtp=$STRATA_MTP  profile=$EXPERT_PROFILE"

# ---------------------------------------------------------------- 5. the engine config
mkdir -p "$RUN_DIR" "$(dirname "$LOG")" 2>/dev/null || true
CONFIG="${STRATA_CONFIG:-$RUN_DIR/strata-hip.json}"
export CONFIG NATIVE PLE PACK="$STRATA_PACK_DIR" MTP="$STRATA_MTP" EXPERT_PROFILE REPO LOG MODEL_NAME \
       RESERVE_MIB MAX_CONTEXT MAX_TOKENS REASONING PREFILL KV SPEC POOL_WORKERS EXPERT_CACHE
"$PY" - <<'PY'
import json, os
e = os.environ
args = ["--pack", e["PACK"], "--native", e["NATIVE"], "--ple-gguf", e["PLE"],
        "--mmap-experts",                       # ROCm: large pinned allocations fail even with free RAM
        "--expert-profile", e["EXPERT_PROFILE"], "--expert-cache", e["EXPERT_CACHE"],
        "--vram-reserve-mib", e["RESERVE_MIB"],
        "--prefill", e["PREFILL"]]
if int(e["SPEC"]) > 0:
    args += ["--mtp", e["MTP"], "--spec", e["SPEC"], "--spec-min-p", "0.5"]
args += ["--max-context", e["MAX_CONTEXT"], "--kv", e["KV"], "--pool-workers", e["POOL_WORKERS"],
         "--adapt-every", "0", "--pcie-frac", "0"]
cfg = {"exe": os.environ.get("STRATA_EXE", "/usr/local/bin/strata"), "args": args,
       "cwd": e["REPO"], "tokenizer": e["PACK"] + "/tokenizer", "model_name": e["MODEL_NAME"],
       "log": e["LOG"], "host": os.environ.get("STRATA_HOST", "0.0.0.0"),
       # a request that names neither max_tokens nor reasoning effort gets these; the request always wins
       "max_tokens": int(e["MAX_TOKENS"]),
       # clamp a too-long output cap to the room that is left instead of answering 400: a client with a
       # heuristic tokenizer will sometimes ask for more than the window has
       "fit_max_tokens": os.environ.get("STRATA_FIT_MAX_TOKENS", "1") == "1",
       "reasoning_effort": e["REASONING"]}
path = e["CONFIG"]
open(path, "w").write(json.dumps(cfg, indent=1) + "\n")
print("strata-hip: wrote %s\n            %s" % (path, " ".join(args)))
PY

# ---------------------------------------------------------------- 6. serve
cd "$REPO"
log "starting the server on ${STRATA_HOST:-0.0.0.0}:${STRATA_PORT:-8080} - first start maps tens of GB" \
    "into RAM, so the PC can be unresponsive for 1-3 minutes (normal; docs/DOCKER_GFX1101_PLAN.md P6)"
exec "$PY" -m serve.server --engine strata --config "$CONFIG" --port "${STRATA_PORT:-8080}"
