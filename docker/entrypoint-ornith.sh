#!/usr/bin/env bash
# Container entrypoint for the Ornith-1.5 / Qwen35MoE path (../run3.sh).
#
# It exists as a SEPARATE entrypoint on purpose: the Qwen3.8 flow (entrypoint-hip.sh) is built around a
# two-shard GGUF, an experts.bin pack and a Qwen4Exp MTP runtime, and pretending an Ornith checkpoint is
# one of those is exactly what the engine's architecture guard refuses.  This entrypoint:
#
#   guard -> device -> VRAM budget -> resolve the single GGUF + external MTP -> validate the artifact
#         -> build the engine config -> serve
#
# The artifact validation is `strata-qwen35-check`, the compiled-in Qwen35MoE geometry/tensor guard (the
# same code `src/core/qwen35.cpp` uses).  It reads the header only, so it is fast even on a 20 GB file.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # /opt/strata/docker
REPO="$(dirname "$DIR")"
PY="${PYTHON:-python3}"

log() { printf 'strata-ornith: %s\n' "$*"; }
die() { printf 'strata-ornith: ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. environment guards
if [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ] && [ "${STRATA_ALLOW_HSA_OVERRIDE:-0}" != "1" ]; then
  die "HSA_OVERRIDE_GFX_VERSION='$HSA_OVERRIDE_GFX_VERSION' is set; on gfx1101 it makes the code object
       fail to load. Remove it, or set STRATA_ALLOW_HSA_OVERRIDE=1 if you know why."
fi
unset HSA_OVERRIDE_GFX_VERSION
[ -n "${AMD_SERIALIZE_KERNEL:-}" ] && { log "unsetting AMD_SERIALIZE_KERNEL=$AMD_SERIALIZE_KERNEL"; unset AMD_SERIALIZE_KERNEL; }
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-1}"

# ---------------------------------------------------------------- 2. the device
ARCH="$("$PY" "$DIR/hipinfo.py" --arch)" \
  || die "no AMD GPU visible: run with --device /dev/kfd --device /dev/dri/renderD<N>"
case "$ARCH" in
  gfx1100|gfx1101) ;;
  *) die "the GPU is $ARCH; this launcher targets gfx1100/gfx1101 (docs/ORNITH_QWEN35MOE.md)" ;;
esac
log "GPU (HIP index $HIP_VISIBLE_DEVICES): $ARCH"
[ -x /usr/local/bin/strata-device ] && { /usr/local/bin/strata-device >/tmp/strata-device.txt 2>&1 \
  || die "the engine cannot open this GPU"; head -3 /tmp/strata-device.txt | sed 's/^/  /'; }

# ---------------------------------------------------------------- 3. VRAM budget (same contract as run.sh/run2.sh)
BUDGET_MIB="${STRATA_VRAM_BUDGET_MIB:-10240}"
[[ "$BUDGET_MIB" =~ ^[0-9]+$ ]] && [ "$BUDGET_MIB" -gt 1280 ] && [ "$BUDGET_MIB" -le 10240 ] \
  || die "VRAM budget must be 1281..10240 MiB; 10 GiB is the hard ceiling"
export STRATA_VRAM_BUDGET_MIB="$BUDGET_MIB"
export STRATA_VRAM_LATER_MIB="${STRATA_VRAM_LATER_MIB:-768}"
export STRATA_VRAM_SLACK_MIB="${STRATA_VRAM_SLACK_MIB:-256}"
export STRATA_VRAM_RUNTIME_RESERVE_MIB="${STRATA_VRAM_RUNTIME_RESERVE_MIB:-1024}"
log "VRAM: ${BUDGET_MIB} MiB ceiling; runtime reserve ${STRATA_VRAM_RUNTIME_RESERVE_MIB} MiB, slack ${STRATA_VRAM_SLACK_MIB} MiB"

# ---------------------------------------------------------------- 4. the model, from the HF cache
MODEL="${STRATA_MODEL:-ornith}"
export STRATA_HF_CACHE="${STRATA_HF_CACHE:-${HF_HUB_CACHE:-/hf-cache}}"
export STRATA_WORK="${STRATA_WORK:-/work}"
LOG="${STRATA_LOG:-$STRATA_WORK/logs/ornith/engine.log}"
RUN_DIR="${STRATA_RUNTIME_DIR:-/run}"
MAX_CONTEXT="${STRATA_MAX_CONTEXT:-131072}"
MAX_TOKENS="${STRATA_MAX_TOKENS:-32768}"
REASONING="${STRATA_REASONING_EFFORT:-high}"
PREFILL="${STRATA_PREFILL:-2048}"
EXPERT_CACHE="${STRATA_EXPERT_CACHE:-auto}"
POOL_WORKERS="${STRATA_POOL_WORKERS:-0}"
KV="${STRATA_KV:-int8}"
SPEC="${STRATA_SPEC:-3}"
MODEL_NAME="${STRATA_MODEL_NAME:-ornith-1.5-35b-a3b-ad-q4-iq4}"
MTP_MODE="${STRATA_MTP_MODE:-auto}"      # auto | 0 | a path inside the container

log "resolving the Ornith artifact in the HF cache ..."
"$PY" "$DIR/hfmodel.py" --model "$MODEL" --cache "$STRATA_HF_CACHE" --print available | sed 's/^/  /'
eval "$("$PY" "$DIR/hfmodel.py" --model "$MODEL" --cache "$STRATA_HF_CACHE" --print shell \
        ${STRATA_MTP_REPO:+--repo "$STRATA_MTP_REPO"})"

# STRATA_NATIVE / STRATA_MTP_PATH bypass the cache lookup (an artifact mounted from somewhere else).
NATIVE="${STRATA_NATIVE:-$STRATA_SHARD1}"
[ -n "$NATIVE" ] && [ -e "$NATIVE" ] || die "the Ornith GGUF is not in $STRATA_HF_CACHE.
     Download it on the host (or let the launcher do it):
       hf download $STRATA_REPO --include '$STRATA_HF_INCLUDE'"
MTP_GGUF=""
if [ "$MTP_MODE" != "0" ] && [ "$MTP_MODE" != "off" ]; then
  case "$MTP_MODE" in
    auto) MTP_GGUF="${STRATA_MTP_PATH:-$STRATA_MTP_GGUF}" ;;
    *)    MTP_GGUF="$MTP_MODE" ;;
  esac
fi
if [ -n "$MTP_GGUF" ] && [ ! -e "$MTP_GGUF" ]; then
  log "no MTP draft at '$MTP_GGUF'; falling back to --spec 0 (the external draft is separate: $STRATA_MTP_REPO / $STRATA_MTP_FILE)"
  MTP_GGUF=""
fi
log "model $MODEL"
log "  gguf=$NATIVE"
log "  mtp=${MTP_GGUF:-<none>}"

# ---------------------------------------------------------------- 5. validate the artifact (the guard)
log "validating the qwen35moe geometry and tensor set ..."
if command -v strata-qwen35-check >/dev/null 2>&1; then
  strata-qwen35-check "$NATIVE" || die "the artifact failed the Qwen35MoE guard above"
else
  log "strata-qwen35-check is not installed in this image; skipping the header guard"
fi

# ---------------------------------------------------------------- 6. the engine config
mkdir -p "$RUN_DIR" "$(dirname "$LOG")" 2>/dev/null || true
CONFIG="${STRATA_CONFIG:-$RUN_DIR/strata-ornith.json}"
# The engine gains a qwen35moe serve path behind --native; until that backend is enabled in the build,
# the engine refuses the artifact at load with a message naming the architecture.  The config below is
# the intended invocation, kept here so the launcher is not the thing that has to change.
export CONFIG NATIVE MTP_GGUF LOG MODEL_NAME MAX_CONTEXT MAX_TOKENS REASONING PREFILL KV SPEC POOL_WORKERS EXPERT_CACHE
"$PY" - <<'PY'
import json, os
e = os.environ
args = ["--native", e["NATIVE"]]
if e["MTP_GGUF"]:
    args += ["--mtp", e["MTP_GGUF"], "--spec", e["SPEC"], "--spec-min-p", "0.5"]
args += ["--expert-cache", e["EXPERT_CACHE"], "--prefill", e["PREFILL"],
         "--max-context", e["MAX_CONTEXT"], "--kv", e["KV"],
         "--pool-workers", e["POOL_WORKERS"], "--adapt-every", "0", "--pcie-frac", "0"]
cfg = {"exe": os.environ.get("STRATA_EXE", "/usr/local/bin/strata"), "args": args,
       "cwd": "/opt/strata", "tokenizer": e["NATIVE"], "model_name": e["MODEL_NAME"],
       "log": e["LOG"], "host": os.environ.get("STRATA_HOST", "0.0.0.0"),
       "max_tokens": int(e["MAX_TOKENS"]), "reasoning_effort": e["REASONING"]}
open(e["CONFIG"], "w").write(json.dumps(cfg, indent=1) + "\n")
print("strata-ornith: wrote %s\n               %s" % (e["CONFIG"], " ".join(args)))
PY

# ---------------------------------------------------------------- 7. check-only stops here
if [ "${STRATA_CHECK_ONLY:-0}" = "1" ]; then
  log "check-only: the artifact and its geometry are valid; not starting the server"
  exit 0
fi

# ---------------------------------------------------------------- 8. serve
# The engine is the source of truth for what it can serve.  run3.sh checks this before the download too;
# this is the in-container backstop (e.g. an image swapped in later).
CAPS="$(/usr/local/bin/strata --capabilities 2>/dev/null | tr '\n' ' ')"
case " $CAPS " in
  *" qwen35moe "*) ;;
  *) log "this engine build has no qwen35moe execution backend (it serves: ${CAPS:-unknown})."
     log "The Ornith artifact and geometry validated; running the model is future work (docs/ORNITH_QWEN35MOE.md)."
     exit 78 ;;
esac
cd "$REPO"
log "starting the server on ${STRATA_HOST:-0.0.0.0}:${STRATA_PORT:-8080}"
exec "$PY" -m serve.server --engine strata --config "$CONFIG" --port "${STRATA_PORT:-8080}"
