#!/usr/bin/env bash
# Run the Ornith-1.5-35B-A3B server container on this machine's AMD GPU (Qwen35MoE path).
#
#   ./run3.sh                     Ornith AD-Q4_K-IQ4_XS + external Qwen3.6 MTP, http://127.0.0.1:9931
#   ./run3.sh --no-mtp            target-only (spec off) for a direct speed comparison
#   ./run3.sh --spec 4            a longer draft window
#   ./run3.sh --detach            background;  ./run3.sh --check  asks it afterwards whether it is up
#   ./run3.sh --offline           never download: fail with the command to run instead
#   ./run3.sh --dry-run           print the docker command and the reasoning, change nothing
#
# Ornith-1.5 is a DIFFERENT architecture from Qwen3.8-Flash-Next (Qwen35MoE: 40 layers, 30 gated-delta-net
# recurrent + 10 full-attention, 256 experts top-8, 2048-wide).  It therefore has its own launcher, its own
# container name and its own work directory; run.sh and run2.sh are untouched.  See
# docs/ORNITH_QWEN35MOE.md for the architecture, the artifacts and the validation behind them.
#
# Main model:  AtomicChat/Ornith-1.5-35B-A3B-GGUF / Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf  (~20.1 GB)
# MTP draft:   EryriLabs/Ornith-1.5-35B-A3B-BigBang-MTP-GGUF / mtpdraft-Q8_0.gguf           (~2.0 GB)
#
# Both live in the shared Hugging Face cache; prepared artifacts go under the writable /work mount, kept
# separate from the Qwen3.8 pack/MTP trees.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ARCH=""                                        # empty: whatever this machine actually has
MODEL="${STRATA_MODEL:-ornith}"                # the hfmodel.py key for the single-file family
IMAGE="${STRATA_IMAGE:-}"
NAME=""
export STRATA_VRAM_LATER_MIB="${STRATA_VRAM_LATER_MIB:-768}"
BUDGET="${STRATA_VRAM_BUDGET_MIB:-10240}"      # this card has 12 272 MiB; the contract is 10 GiB
DEFAULT_MODEL_DIR="${DEFAULT_MODEL_DIR:-$HOME/Development/models}"
HF_CACHE="${STRATA_HF_CACHE_HOST:-}"
WORK="${STRATA_WORK_HOST:-}"
PORT="${STRATA_PORT_HOST:-9931}"
BIND="${STRATA_BIND:-127.0.0.1}"
MEMORY="${STRATA_CONTAINER_MEMORY:-96g}"
MAX_CONTEXT="${STRATA_MAX_CONTEXT:-131072}"    # 128K default; 262144 is available and tested separately
SPEC="${STRATA_SPEC:-3}"                       # measured default; override with --spec / --no-mtp
DETACH=0 CHECK=0 FRESH=0 OFFLINE=0 DRY=0 CHECK_ONLY=0 EXPLICIT_CACHE=""
MODEL_FILE="" MTP_PATH=""
EXTRA_ENV=()
EXTRA_ARGS=()

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
xrun()   { if [ "$DRY" = 1 ]; then printf '  +'; printf ' %q' "$@"; printf '\n'; else "$@"; fi; }

usage() {
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
  cat <<'USAGE'

Options:
      --model KEY         hfmodel.py family key (default: ornith)
      --model-file PATH   the main GGUF directly (skips the cache lookup)
      --hf-cache PATH     model cache in the Hugging Face layout (default: ~/Development/models)
      --work PATH         writable Strata state: packs, MTP, logs (default: ~/Development/strata-work)
      --budget MiB        VRAM Strata may use for itself (default 10240 on a 12 GiB card)
  -p, --port PORT         host port (default 9931)
      --bind ADDR         host interface to publish on (default 127.0.0.1)
  -d, --detach            run in the background instead of in the foreground
      --check             query /v1/models of the running container and exit
      --check-only        validate the GPU, the artifacts and the geometry, then exit (no server)
      --offline           never download: fail with instructions instead
      --fresh             remove an existing container of the same name first
      --max-context N     KV context (default 131072; 262144 when validated on this card)
      --max-tokens N      output cap a request gets when it names none (default 32768)
      --reasoning-effort L  off, minimal, low, medium, high (default high)
      --prefill N         prefill chunk (default 2048)
      --expert-cache N    expert slot budget (default auto)
      --pool-workers N    CPU expert workers (default 0 = auto)
      --spec N            MTP draft length (default 3)
      --no-mtp            disable the external MTP draft (--spec 0)
      --mtp PATH          a specific MTP GGUF (default: the cached mtpdraft-Q8_0.gguf)
  -e, --env KEY=VALUE     pass extra environment through (repeatable)
      --image NAME:TAG    override the image (default strata-hip:<arch>-latest)
      --allow-hsa-override  let HSA_OVERRIDE_GFX_VERSION through (breaks gfx1101; you have been told)
  -n, --dry-run           print the command instead of running it
  -h, --help              this text
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model)           MODEL="${2:?--model needs a value}"; shift 2 ;;
    --model-file)      MODEL_FILE="${2:?}"; shift 2 ;;
    --hf-cache)        HF_CACHE="${2:?}"; shift 2 ;;
    --work)            WORK="${2:?}"; shift 2 ;;
    --budget)          BUDGET="${2:?}"; shift 2 ;;
    -p|--port)         PORT="${2:?}"; shift 2 ;;
    --bind)            BIND="${2:?}"; shift 2 ;;
    -d|--detach)       DETACH=1; shift ;;
    --check)           CHECK=1; shift ;;
    --check-only)      CHECK_ONLY=1; DETACH=0; shift ;;
    --offline)         OFFLINE=1; shift ;;
    --fresh)           FRESH=1; shift ;;
    --max-context)     MAX_CONTEXT="${2:?--max-context needs a value}"; shift 2 ;;
    --max-tokens)      EXTRA_ENV+=(-e "STRATA_MAX_TOKENS=${2:?}"); shift 2 ;;
    --reasoning-effort) EXTRA_ENV+=(-e "STRATA_REASONING_EFFORT=${2:?}"); shift 2 ;;
    --prefill)         EXTRA_ENV+=(-e "STRATA_PREFILL=${2:?}"); shift 2 ;;
    --expert-cache)    EXPLICIT_CACHE="${2:?--expert-cache needs a value}"; shift 2 ;;
    --pool-workers)    EXTRA_ENV+=(-e "STRATA_POOL_WORKERS=${2:?}"); shift 2 ;;
    --spec)            SPEC="${2:?--spec needs a value}"; shift 2 ;;
    --no-mtp)          SPEC=0; shift ;;
    --mtp)             MTP_PATH="${2:?--mtp needs a value}"; shift 2 ;;
      --allow-hsa-override) EXTRA_ENV+=(-e "STRATA_ALLOW_HSA_OVERRIDE=1" -e "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"); shift ;;
    -e|--env)          EXTRA_ENV+=(-e "${2:?--env needs KEY=VALUE}"); shift 2 ;;
    --image)           IMAGE="${2:?}"; shift 2 ;;
    -n|--dry-run)      DRY=1; shift ;;
    -h|--help)         usage; exit 0 ;;
    *)                 usage >&2; die "unknown option '$1'" ;;
  esac
done

[[ "$BUDGET" =~ ^[0-9]+$ ]] && [ "$BUDGET" -gt 1280 ] && [ "$BUDGET" -le 10240 ] \
  || die "budget must be 1281..10240 MiB; 10 GiB is the hard ceiling"
[[ "$MAX_CONTEXT" =~ ^[0-9]+$ ]] && [ "$MAX_CONTEXT" -gt 0 ] && [ "$MAX_CONTEXT" -le 262144 ] \
  || die "--max-context must be 1..262144 (Ornith's native maximum)"
[[ "$SPEC" =~ ^[0-9]+$ ]] || die "--spec must be a non-negative integer"

PY="${PYTHON:-python3}"
HIPINFO="$ROOT/docker/hipinfo.py"
command -v docker >/dev/null || die "docker not found.  Build first: ./build.sh"
[ -f "$HIPINFO" ] || die "$HIPINFO is missing (is this the Strata repository root?)"

# ---------------------------------------------------------------- the card
ARCH_GUESSED="$("$PY" "$HIPINFO" --arch 2>/dev/null)" || die "no AMD GPU visible through the KFD topology." \
  note "Strata's HIP backend needs the amdgpu driver and /dev/kfd."
ARCH="${ARCH:-$ARCH_GUESSED}"
case "$ARCH" in
  gfx1100|gfx1101) ;;
  *) die "this machine's GPU is $ARCH; the Ornith launcher targets gfx1100/gfx1101 (docs/ORNITH_QWEN35MOE.md)" ;;
esac
RENDER="$("$PY" "$HIPINFO" --render-node)"
read -r VRAM_TOTAL VRAM_USED _VRAM_FREE <<<"$("$PY" "$HIPINFO" --vram)"
RESERVE="$("$PY" "$HIPINFO" --reserve-mib --allocation-guard --budget-mib "$BUDGET" --slack-mib "${STRATA_VRAM_SLACK_MIB:-256}")"
log "GPU: $ARCH ($RENDER), ${VRAM_TOTAL} MiB VRAM; ${BUDGET} MiB Strata ceiling"
note "allocation guard preserves runtime overhead and slack; cache reserves ${RESERVE} MiB for later buffers"

if [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ] && [ "${STRATA_ALLOW_HSA_OVERRIDE:-0}" != "1" ]; then
  die "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION is set in your environment. On gfx1101 it makes
       the code object fail to load. Unset it, or pass --allow-hsa-override if you really mean it."
fi

# ---------------------------------------------------------------- the image
if [ -z "$IMAGE" ]; then
  for cand in "strata-hip:${ARCH}-latest" "strata-hip:${ARCH}"; do
    if docker image inspect "$cand" >/dev/null 2>&1; then IMAGE="$cand"; break; fi
  done
fi
[ -n "$IMAGE" ] || die "no strata-hip image for $ARCH. Build it:  ./build.sh   (takes 10-25 minutes the first time)"
NAME="${NAME:-strata-ornith-${ARCH}}"
log "image: $IMAGE   container: $NAME"
[ "$FRESH" = 1 ] && [ "$DRY" != 1 ] && docker rm -f "$NAME" >/dev/null 2>&1 || true

# ---------------------------------------------------------------- the cache and the space it needs
if [ -z "$HF_CACHE" ]; then
  HF_CACHE="$DEFAULT_MODEL_DIR"
  [ -e "$HOME/.cache/huggingface/hub" ] \
    && note "hint: export HF_HUB_CACHE=$HF_CACHE  makes 'hf' and this script share one cache"
fi
WORK="${WORK:-$(dirname "$DEFAULT_MODEL_DIR")/strata-work}"
WORK="$(readlink -m "$WORK" 2>/dev/null || printf '%s' "$WORK")"
HF_CACHE="$(readlink -m "$HF_CACHE" 2>/dev/null || printf '%s' "$HF_CACHE")"
OR_NATIVE="${STRATA_NATIVE:-}"
OR_MTP="${MTP_PATH:-${STRATA_MTP_PATH:-}}"
if [ -n "$OR_NATIVE" ]; then
  eval "$("$PY" "$ROOT/docker/hfmodel.py" --model "$MODEL" --cache "$HF_CACHE" --print shell --allow-missing 2>/dev/null || true)"
  note "using --model-file $OR_NATIVE (cache lookup skipped for the main GGUF)"
fi
log "HF cache:  $HF_CACHE  (mounted read-write: a first run may download into it)"
log "work dir:  $WORK  packs=$WORK/packs/ornith-ad-q4-iq4-xs  mtp=$WORK/mtp/ornith-qwen36  logs=$WORK/logs/ornith"

# What is already there?  (asks the same resolver the container uses, so no drift between them)
eval "$("$PY" "$ROOT/docker/hfmodel.py" --model "$MODEL" --cache "$HF_CACHE" --print shell --allow-missing)"
existing_ancestor() { local p="$1"; while [ ! -e "$p" ] && [ "$p" != "/" ]; do p="$(dirname "$p")"; done; printf '%s' "$p"; }
free_gib() { df -BG --output=avail "$(existing_ancestor "$1")" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0; }
dev_of()   { stat -c %d "$(existing_ancestor "$1")" 2>/dev/null || echo none; }
same_fs()  { [ "$(dev_of "$1")" = "$(dev_of "$2")" ]; }
if [ "${STRATA_CACHED:-0}" != 1 ]; then
  # ~20.1 GB main GGUF + ~2 GB MTP + a 20 GB floor; prepared artifacts are small (the native path reads
  # the GGUF itself), so unlike Qwen3.8 there is no 50 GB experts.bin term.
  need="$(awk -v d="${STRATA_DOWNLOAD_GB:-20.1}" -v m="${STRATA_MTP_DOWNLOAD_GB:-2.0}" \
            'BEGIN{printf "%d", d + m + 8 + 20}')"
  have="$(free_gib "$HF_CACHE")"
  log "Ornith is not in the cache yet: ~${need} GB free required (main + MTP + a 20 GB floor)"
  if [ "$OFFLINE" = 1 ]; then
    die "--offline, and $MODEL is not cached. Fetch it on the host:
         HF_HUB_CACHE=$HF_CACHE hf download $STRATA_REPO --include '$STRATA_HF_INCLUDE'
         HF_HUB_CACHE=$HF_CACHE hf download $STRATA_MTP_REPO --include '$STRATA_MTP_FILE'"
  fi
  if [ "${have:-0}" -lt "$need" ]; then
    die "$MODEL needs ${need} GB free - the main GGUF, the MTP draft and 20 GB that must stay free - but
       the filesystem holding $HF_CACHE has ${have} GB free.  Pick a bigger place for the cache:
         ./run3.sh --hf-cache /mnt/storage/models --work /mnt/storage/strata-work"
  fi
else
  note "$MODEL is cached"
fi

# ---------------------------------------------------------------- port
if [ "$DRY" != 1 ] && [ "$CHECK" != 1 ] && [ "$CHECK_ONLY" != 1 ]; then
  if "$PY" - "$BIND" "$PORT" <<'PY'
import socket, sys
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind((sys.argv[1], int(sys.argv[2]))); sys.exit(1)
except OSError:
    sys.exit(0)
finally:
    s.close()
PY
  then
    die "port $PORT on $BIND is already in use - Strata is probably already running:
       docker logs -f $NAME    (or ./run3.sh --check)"
  fi
fi

# ---------------------------------------------------------------- --check: just ask the running server
if [ "$CHECK" = 1 ]; then
  log "asking http://$BIND:$PORT/v1/models ..."
  "$PY" - "$BIND" "$PORT" <<'PY'
import json, sys, urllib.request
try:
    with urllib.request.urlopen(f"http://{sys.argv[1]}:{sys.argv[2]}/v1/models", timeout=10) as r:
        models = json.load(r).get("data", [])
except Exception as e:
    print(f"    not answering yet: {e}", file=sys.stderr)
    raise SystemExit(1)
if not models:
    print("    answering, but no model is listed", file=sys.stderr)
    raise SystemExit(1)
for m in models:
    status = (m.get("status") or {}).get("value", "?")
    ctx = (m.get("meta") or {}).get("n_ctx", "?")
    print(f"    {m.get('id')}  {status}  ctx={ctx}")
    if status != "loaded":
        raise SystemExit(1)
PY
  exit $?
fi

# Refuse inference with an older engine image that cannot admit allocations.
if [ "$DRY" = 0 ] && [ "$CHECK_ONLY" = 0 ]; then
  budget_guard="$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "io.strata.vram-budget"}}' 2>/dev/null || true)"
  [ "$budget_guard" = 1 ] || die "image $IMAGE predates the HIP allocation guard; rebuild with ./build.sh before inference"
fi

# ---------------------------------------------------------------- run it
CHECK_ONLY_SUFFIX=""
[ "$CHECK_ONLY" = 1 ] && CHECK_ONLY_SUFFIX="-check"
ARGS=(docker run --rm --name "$NAME$CHECK_ONLY_SUFFIX"
      --entrypoint /opt/strata/docker/entrypoint-ornith.sh
      --device /dev/kfd --device "$RENDER"
      --group-add video --group-add render
      --ulimit memlock=-1:-1
      --shm-size=16g --memory "$MEMORY"
      -e "STRATA_MODEL=$MODEL"
      -e "STRATA_VRAM_BUDGET_MIB=$BUDGET"
      -e "STRATA_MAX_CONTEXT=$MAX_CONTEXT"
      -e "STRATA_POOL_WORKERS=${STRATA_POOL_WORKERS:-0}"
      -e "STRATA_PREFILL=${STRATA_PREFILL:-2048}"
      -e "STRATA_EXPERT_CACHE=${STRATA_EXPERT_CACHE:-${EXPLICIT_CACHE:-auto}}"
      -e "STRATA_SPEC=$SPEC"
      -e "STRATA_VRAM_LATER_MIB=$STRATA_VRAM_LATER_MIB"
      -e "STRATA_VRAM_RUNTIME_RESERVE_MIB=${STRATA_VRAM_RUNTIME_RESERVE_MIB:-1024}"
      -e "STRATA_VRAM_SLACK_MIB=${STRATA_VRAM_SLACK_MIB:-256}"
      -e "STRATA_PORT=$PORT"
      -e "STRATA_WORK=/work"
      -e "HIP_VISIBLE_DEVICES=0" -e "HSA_ENABLE_SDMA=1"
      -v "$HF_CACHE:/hf-cache" -v "$WORK:/work")
[ "$CHECK_ONLY" = 1 ] || ARGS+=(-p "$BIND:$PORT:$PORT")
[ -n "$OR_NATIVE" ] && ARGS+=(-v "$OR_NATIVE:/models/ornith.gguf:ro" -e "STRATA_NATIVE=/models/ornith.gguf")
[ -n "$OR_MTP" ]    && ARGS+=(-v "$OR_MTP:/models/mtp.gguf:ro" -e "STRATA_MTP_PATH=/models/mtp.gguf")
[ -n "${HF_TOKEN:-}" ]    && ARGS+=(-e "HF_TOKEN=$HF_TOKEN")
[ -n "${HF_REVISION:-}" ] && ARGS+=(-e "HF_REVISION=$HF_REVISION")
for tuning_key in STRATA_MAX_TOKENS STRATA_REASONING_EFFORT STRATA_PREFILL_RING STRATA_STAGER_RING \
                  STRATA_STAGER_THREADS STRATA_IO_THREADS STRATA_HIPBLASLT_TUNING STRATA_KV \
                  STRATA_MTP_REPO STRATA_MTP_FILE; do
  [ -z "${!tuning_key:-}" ] || ARGS+=(-e "$tuning_key=${!tuning_key}")
done
[ "$CHECK_ONLY" = 1 ] && ARGS+=(-e "STRATA_AUTO_PREPARE=1" -e "STRATA_CHECK_ONLY=1")
[ ${#EXTRA_ENV[@]} -gt 0 ] && ARGS+=("${EXTRA_ENV[@]}")
if [ "$DETACH" = 1 ]; then ARGS+=(-d "$IMAGE"); else ARGS+=(-it "$IMAGE"); fi

log "starting Ornith on $ARCH (server on http://$BIND:$PORT, logs: $WORK/logs/ornith/)"
note "spec=$SPEC ($([ "$SPEC" = 0 ] && echo 'no MTP' || echo "external Qwen3.6 MTP")); context=$MAX_CONTEXT"
note "first start downloads ~22 GB and validates the artifact: 1-3 minutes of a slow PC is normal"
if [ "$DETACH" = 1 ]; then
  xrun "${ARGS[@]}"
  [ "$DRY" = 1 ] || { note "container $NAME;  logs: docker logs -f $NAME"; note "ready check: ./run3.sh --check"; }
else
  xrun "${ARGS[@]}"
fi
