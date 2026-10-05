#!/usr/bin/env bash
# Run the Strata server container on this machine's AMD GPU.
#
#   ./run.sh                      IQ3_XXS, 10 GiB VRAM budget, http://127.0.0.1:9931
#   ./run.sh --model IQ3_S        the best-quality quant this card's 10 GiB slice still takes
#   ./run.sh --model IQ2_XS       a smaller quant (fetched into your HF cache on first start)
#   ./run.sh --detach             background;  ./run.sh --check  asks it afterwards whether it is up
#   ./run.sh --offline            never download: fail with the command to run instead
#   ./run.sh --dry-run            print the docker command and the reasoning, change nothing
#
# The expert-cache budget is tuned per quant: IQ3_S is the tightest fit (its experts.bin is ~50 GB
# against IQ3_XXS's ~43 GB, so every slot holds ~17% more bytes), and the budget below scales it
# down (680 largest-blob slots instead of 800) so each quant spends the same VRAM: the 10 GiB
# ceiling is NOT raised, and the engine still caps the cache to the free room under the guard.
#
# Models are read from, and downloaded into, a Hugging Face cache in ~/Development/models (mounted
# at /hf-cache) - the big filesystem, not the root disk that holds ~/.cache/huggingface.  One download
# serves the host's `hf` client, this container and any other container.  The pack and the MTP draft
# layer are Strata artifacts and go to the writable /work mount (~/Development/strata-work) instead.
#
# Refuses to start on the wrong card, with HSA_OVERRIDE_GFX_VERSION set, or when the filesystem that
# holds the cache cannot hold the quant - all before docker sees a single byte of data.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ARCH=""                                        # empty: whatever this machine actually has
MODEL="${STRATA_MODEL:-IQ3_S}"                 # the shipped default quant
IMAGE="${STRATA_IMAGE:-}"
NAME=""
export STRATA_VRAM_LATER_MIB="${STRATA_VRAM_LATER_MIB:-768}"
BUDGET="${STRATA_VRAM_BUDGET_MIB:-10240}"      # this card has 12 272 MiB; the contract is 10 GiB
# The model cache lives with the models, on the big filesystem: ~$HOME/Development (a symlink to
# /mnt/storage/Development here) - NOT ~/.cache/huggingface, whose filesystem has ~90 GB free and
# cannot hold a 76 GB quant plus its pack.  Override with --hf-cache / $STRATA_HF_CACHE_HOST.
DEFAULT_MODEL_DIR="${DEFAULT_MODEL_DIR:-$HOME/Development/models}"
HF_CACHE="${STRATA_HF_CACHE_HOST:-}"
WORK="${STRATA_WORK_HOST:-}"
PORT="${STRATA_PORT_HOST:-9931}"
BIND="${STRATA_BIND:-127.0.0.1}"
MEMORY="${STRATA_CONTAINER_MEMORY:-96g}"
MAX_CONTEXT="${STRATA_MAX_CONTEXT:-131072}"          # 128K; preserves an explicit environment override
DETACH=0 CHECK=0 FRESH=0 OFFLINE=0 DRY=0 CHECK_ONLY=0 EXPLICIT_CACHE=""
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
  -m, --model QUANT        IQ3_XXS (default), IQ3_S, IQ2_XS, Q2_0, IQ1_M (the Coder)
      --hf-cache PATH      model cache in the Hugging Face layout (default: ~/Development/models,
                           which is on the big filesystem here; NOT ~/.cache/huggingface, whose disk
                           is too small for a quant + its pack)
      --work PATH          writable Strata state: packs, MTP, logs
                           (default: ~/Development/strata-work)
      --budget MiB         VRAM Strata may use for ITSELF (default 10240 on a 12 GiB card; what is
                           left over stays for the desktop and the GUI)
  -p, --port PORT          host port (default 9931)
      --bind ADDR          host interface to publish on (default 127.0.0.1; use 0.0.0.0 plus an
                           API key if you must)
  -d, --detach             run in the background instead of in the foreground
      --check              query /v1/models of the running container and exit
      --check-only         run the container's model/pack check and exit (no download, no server)
      --offline            STRATA_DOWNLOAD_MODEL=0: never download, fail with instructions instead
      --fresh              remove an existing container of the same name first
      --max-context N      KV context (must remain 131072 / 128K)
      --max-tokens N       output cap a request gets when it names none (default 32768)
      --reasoning-effort L reasoning level a request gets when it names none: off, minimal, low,
                           medium, high (default high; the request's own value always wins)
      --prefill N          prefill chunk (default: 2048; engine reduces it if borrowing cannot fit)
      --expert-cache N     expert slot budget (default: 800 on IQ3_XXS, 680 on the other quants;
                           native slots are sized per pair, and IQ3_S's blobs are ~17% larger, so
                           680 on IQ3_S holds what 800 holds on IQ3_XXS - both stay inside the
                           10 GiB ceiling)
      --pool-workers N     CPU expert workers (default: 0, physical cores minus the host's)
  -e, --env KEY=VALUE      pass extra environment through (repeatable)
      --image NAME:TAG     override the image (default strata-hip:<arch>-latest)
      --allow-hsa-override let HSA_OVERRIDE_GFX_VERSION through (breaks gfx1101; you have been told)
  -n, --dry-run            print the command instead of running it
  -h, --help               this text
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    -m|--model)        MODEL="${2:?--model needs a value}"; shift 2 ;;
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
    --expert-cache)    EXPLICIT_CACHE="${2:?--expert-cache needs a value}"; EXTRA_ENV+=(-e "STRATA_EXPERT_CACHE=$EXPLICIT_CACHE"); shift 2 ;;
    --pool-workers)    EXTRA_ENV+=(-e "STRATA_POOL_WORKERS=${2:?}"); shift 2 ;;
    -e|--env)          EXTRA_ENV+=(-e "${2:?--env needs KEY=VALUE}"); shift 2 ;;
    --image)           IMAGE="${2:?}"; shift 2 ;;
    --allow-hsa-override) EXTRA_ENV+=(-e "STRATA_ALLOW_HSA_OVERRIDE=1" -e "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"); shift ;;
    -n|--dry-run)      DRY=1; shift ;;
    -h|--help)         usage; exit 0 ;;
    *)                 usage >&2; die "unknown option '$1'" ;;
  esac
done

[[ "$BUDGET" =~ ^[0-9]+$ ]] && [ "$BUDGET" -gt 1280 ] && [ "$BUDGET" -le 10240 ] \
  || die "budget must be 1281..10240 MiB; 10 GiB is the hard ceiling"
[ "$MAX_CONTEXT" = 131072 ] || die "this deployment keeps exactly 131072 tokens (128K context)"
# IQ3_S stores its experts ~17% wider than IQ3_XXS (arena 50.3 GB vs 42.9 GB), so the same VRAM
# holds fewer of them: 680 slots on IQ3_S spend what 800 IQ3_XXS slots spent.  This is the tuned
# budget per quant - not a bigger VRAM slice, which stays capped at $BUDGET MiB.
case "$MODEL" in
  IQ3_XXS) EXPERT_CACHE_DEFAULT=800 ;;   # the shipped default quant
  *)       EXPERT_CACHE_DEFAULT=680 ;;   # the other quants, IQ3_S included
esac
# A slot budget above the tuned value buys nothing under the same 10 GiB ceiling: the engine would
# only cut it back to the free room.  Say so rather than let an explicit number look like more VRAM.
wanted="${STRATA_EXPERT_CACHE:-${EXPLICIT_CACHE:-$EXPERT_CACHE_DEFAULT}}"
if [ "$wanted" != auto ] && [[ "$wanted" =~ ^[0-9]+$ ]] && [ "$wanted" -gt "$EXPERT_CACHE_DEFAULT" ]; then
  warn "expert cache budget $wanted exceeds the tuned $EXPERT_CACHE_DEFAULT for $MODEL; VRAM stays
      capped at $BUDGET MiB, so the engine will cut it back to what fits - raise quality by no other means"
fi

PY="${PYTHON:-python3}"
HIPINFO="$ROOT/docker/hipinfo.py"
case "$MODEL" in
  IQ3_XXS|IQ3_S|IQ2_XS|Q2_0|IQ1_M) ;;
  *) [ -n "${STRATA_HF_REPO:-}" ] || die "unknown quant '$MODEL'.  Strata ships IQ3_XXS (default),
       IQ3_S, IQ2_XS, Q2_0 and IQ1_M (the Coder); pass STRATA_HF_REPO=<org/name> for another release." ;;
esac
command -v docker >/dev/null || die "docker not found.  Build first: ./build.sh"

# ---------------------------------------------------------------- the card
[ -f "$HIPINFO" ] || die "$HIPINFO is missing (is this the Strata repository root?)"
ARCH_GUESSED="$("$PY" "$HIPINFO" --arch 2>/dev/null)" || die "no AMD GPU visible through the KFD topology." \
  note "Strata's HIP backend needs the amdgpu driver and /dev/kfd."
ARCH="${ARCH:-$ARCH_GUESSED}"
case "$ARCH" in
  gfx1100|gfx1101) ;;
  *) die "this machine's GPU is $ARCH; Strata's HIP backend supports gfx1100 (RX 7900 XT/XTX) and
       gfx1101 (RX 7700 XT / 7800 XT). See docs/DOCKER_GFX1101_PLAN.md Part C to widen it." ;;
esac
RENDER="$("$PY" "$HIPINFO" --render-node)"
read -r VRAM_TOTAL VRAM_USED _VRAM_FREE <<<"$("$PY" "$HIPINFO" --vram)"
# HIP allocation admission already enforces the ceiling and reserves runtime
# overhead plus slack. Cache sizing holds back only later explicit buffers.
RESERVE="$("$PY" "$HIPINFO" --reserve-mib --allocation-guard --budget-mib "$BUDGET" --slack-mib "${STRATA_VRAM_SLACK_MIB:-256}")"
log "GPU: $ARCH ($RENDER), ${VRAM_TOTAL} MiB VRAM; ${BUDGET} MiB Strata ceiling"
note "allocation guard preserves runtime overhead and slack; cache reserves ${RESERVE} MiB for later buffers"

if [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ] && [ "${STRATA_ALLOW_HSA_OVERRIDE:-0}" != "1" ]; then
  die "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION is set in your environment (a login shell
       or a leftover -e). On gfx1101 it makes the code object fail to load. Unset it, or pass
       --allow-hsa-override if you really mean it."
fi

# ---------------------------------------------------------------- the image
# build.sh tags every build with the commit hash AND the moving -latest tag; use -latest so a
# rebuild is picked up without touching this script.
if [ -z "$IMAGE" ]; then
  for cand in "strata-hip:${ARCH}-latest" "strata-hip:${ARCH}"; do
    if docker image inspect "$cand" >/dev/null 2>&1; then IMAGE="$cand"; break; fi
  done
fi
if [ -z "$IMAGE" ]; then
  die "no strata-hip image for $ARCH. Build it:  ./build.sh   (takes 10-25 minutes the first time)"
fi
IMAGE_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null | cut -c8-19)"
log "image: $IMAGE (sha256:${IMAGE_ID:-unknown})"
NAME="${NAME:-strata-${ARCH}}"
[ "$FRESH" = 1 ] && [ "$DRY" != 1 ] && docker rm -f "$NAME" >/dev/null 2>&1 || true

# ---------------------------------------------------------------- the cache and the space it needs
if [ -z "$HF_CACHE" ]; then
  HF_CACHE="$DEFAULT_MODEL_DIR"
  [ -n "${HF_HUB_CACHE:-}" ] && [ "$HF_HUB_CACHE" != "$HF_CACHE" ] \
    && note "ignoring HF_HUB_CACHE=$HF_HUB_CACHE in favour of $HF_CACHE (space); --hf-cache to choose"
  # Your other tooling (hf, huggingface-cli, ...) uses ~/.cache/huggingface by default, and a cache
  # is content-addressed, so moving it here is a plain 'mv'.  Point them all at one place:
  [ -e "$HOME/.cache/huggingface/hub" ] \
    && note "hint: export HF_HUB_CACHE=$HF_CACHE  makes 'hf' and this script share one cache"
fi
WORK="${WORK:-$(dirname "$DEFAULT_MODEL_DIR")/strata-work}"
WORK="$(readlink -m "$WORK" 2>/dev/null || printf '%s' "$WORK")"
HF_CACHE="$(readlink -m "$HF_CACHE" 2>/dev/null || printf '%s' "$HF_CACHE")"
log "HF cache:  $HF_CACHE  (mounted read-write: a first run may download into it)"
log "work dir:  $WORK  (packs, MTP draft layer, logs)"

# What is already there?  (asks the same resolver the container uses, so no drift between them)
eval "$("$PY" "$ROOT/docker/hfmodel.py" --model "$MODEL" --cache "$HF_CACHE" --print shell --allow-missing)"
existing_ancestor() { local p="$1"; while [ ! -e "$p" ] && [ "$p" != "/" ]; do p="$(dirname "$p")"; done; printf '%s' "$p"; }
free_gib() { df -BG --output=avail "$(existing_ancestor "$1")" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0; }
dev_of()   { stat -c %d "$(existing_ancestor "$1")" 2>/dev/null || echo none; }
gb_ceil()  { awk -v n="$1" 'BEGIN{printf "%d", n + 1}'; }
same_fs()  { [ "$(dev_of "$1")" = "$(dev_of "$2")" ]; }
if [ "${STRATA_CACHED:-0}" != 1 ]; then
  # The same +8 GB headroom the container's own gate uses (docker/bootstrap-model.sh), plus a 20 GB
  # floor that must still be free afterwards: a disk at 100 % breaks the engine's mmap and the
  # desktop at once.  When the cache and the work dir share a filesystem, the pack's experts.bin
  # competes with the download on that same filesystem and is counted too.
  # (awk, not $(( )): the sizes are fractional, 75.8 is not a bash integer.)
  same=0; same_fs "$HF_CACHE" "$WORK" && same=1
  need="$(awk -v d="${STRATA_DOWNLOAD_GB:-80}" -v a="${STRATA_ARENA_GB:-43}" -v s="$same" \
            'BEGIN{printf "%d", d + 8 + 20 + (s ? a + 6 : 0)}')"
  [ "$same" = 1 ] && note "cache and work dir are on one filesystem: counting the pack's ${STRATA_ARENA_GB} GB too"
  have="$(free_gib "$HF_CACHE")"
  log "$MODEL is not in the cache yet: ~${STRATA_DOWNLOAD_GB} GB to download, ${need} GB free required (incl. a 20 GB floor)"
  if [ "$OFFLINE" = 1 ]; then
    die "--offline, and $MODEL is not cached. Fetch it on the host:
         HF_HUB_CACHE=$HF_CACHE hf download ${STRATA_REPO:-<repo>} --include '${STRATA_HF_INCLUDE:-$MODEL/*}'"
  fi
  if [ "${have:-0}" -lt "$need" ]; then
    die "$MODEL needs ${need} GB free - the quant, its pack, and 20 GB that must stay free
       afterwards - but the filesystem holding $HF_CACHE has ${have} GB free.
       Do not start the download: it would fill the disk and die part-way.  Pick a bigger place for
       the cache (find one with  df -h ), e.g.:
         ./run.sh --hf-cache /mnt/storage/models --work /mnt/storage/strata-work
       (a cache downloaded elsewhere can simply be moved: it is content-addressed)"
  fi
else
  note "$MODEL is cached (both shards)"
fi
need_pack="$(awk -v a="${STRATA_ARENA_GB:-51}" 'BEGIN{printf "%d", a + 6}')"; have_work="$(free_gib "$WORK")"
if [ "${have_work:-0}" != 0 ] && [ "${have_work:-0}" -lt "$need_pack" ] && [ ! -f "$WORK/packs/$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')/experts.bin" ]; then
  warn "$WORK has ${have_work} GB free; the pack writes ~${STRATA_ARENA_GB} GB of experts.bin."
  warn "Pass --work /somewhere/bigger if the packing step fails for space."
fi

# ---------------------------------------------------------------- port
# Only when a container is actually about to bind it: --check asks an existing server, and
# --check-only inspects the model/pack from inside a throwaway container - both are exactly what you
# run *while* the server holds the port, so requiring it free made them unusable.
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
       docker logs -f $NAME    (or ./run.sh --check)"
  fi
fi

# ---------------------------------------------------------------- --check: just ask the running server
if [ "$CHECK" = 1 ]; then
  log "asking http://$BIND:$PORT/v1/models ..."
  # The heredoc belongs to python, not to log: attached to log it never ran and --check reported
  # success for a server that was down.  It reports the model's own status, not just a port open.
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
# Check-only and dry-run do not start an inference engine.
if [ "$DRY" = 0 ] && [ "$CHECK_ONLY" = 0 ]; then
  budget_guard="$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "io.strata.vram-budget"}}' 2>/dev/null || true)"
  [ "$budget_guard" = 1 ] || die "image $IMAGE predates the HIP allocation guard; rebuild with ./build.sh before inference"
fi

# ---------------------------------------------------------------- run it
# --check-only is a throwaway inspector: it must not publish the port or take the running server's
# container name, both of which would collide with the very server you are asking about.
CHECK_ONLY_SUFFIX=""
[ "$CHECK_ONLY" = 1 ] && CHECK_ONLY_SUFFIX="-check"
ARGS=(docker run --rm --name "$NAME$CHECK_ONLY_SUFFIX"
      --device /dev/kfd --device "$RENDER"          # only the discrete card: the iGPU stays invisible
      --group-add video --group-add render
      --ulimit memlock=-1:-1                        # Strata locks RAM for the GPU
      --shm-size=16g --memory "$MEMORY"
      -e "STRATA_MODEL=$MODEL"
      -e "STRATA_VRAM_BUDGET_MIB=$BUDGET"
      -e "STRATA_MAX_CONTEXT=$MAX_CONTEXT"
      -e "STRATA_POOL_WORKERS=${STRATA_POOL_WORKERS:-0}"
      -e "STRATA_PREFILL=${STRATA_PREFILL:-2048}"
      -e "STRATA_EXPERT_CACHE=${STRATA_EXPERT_CACHE:-$EXPERT_CACHE_DEFAULT}"
      -e "STRATA_VRAM_LATER_MIB=$STRATA_VRAM_LATER_MIB"
      -e "STRATA_VRAM_RUNTIME_RESERVE_MIB=${STRATA_VRAM_RUNTIME_RESERVE_MIB:-1024}"
      -e "STRATA_VRAM_SLACK_MIB=${STRATA_VRAM_SLACK_MIB:-256}"
      -e "STRATA_PORT=$PORT"
      -e "HIP_VISIBLE_DEVICES=0" -e "HSA_ENABLE_SDMA=1"
      -e "STRATA_DOWNLOAD_MODEL=$([ "$OFFLINE" = 1 ] && echo 0 || echo 1)"
      -v "$HF_CACHE:/hf-cache" -v "$WORK:/work")
[ "$CHECK_ONLY" = 1 ] || ARGS+=(-p "$BIND:$PORT:$PORT")
[ -n "${HF_TOKEN:-}" ]        && ARGS+=(-e "HF_TOKEN=$HF_TOKEN")
[ -n "${HF_REVISION:-}" ]     && ARGS+=(-e "HF_REVISION=$HF_REVISION")
[ -n "${STRATA_MAX_TOKENS:-}" ]  && ARGS+=(-e "STRATA_MAX_TOKENS=$STRATA_MAX_TOKENS")
[ -n "${STRATA_REASONING_EFFORT:-}" ] && ARGS+=(-e "STRATA_REASONING_EFFORT=$STRATA_REASONING_EFFORT")
for tuning_key in STRATA_PREFILL_RING STRATA_STAGER_RING STRATA_STAGER_THREADS \
                  STRATA_IO_THREADS STRATA_HIPBLASLT_TUNING STRATA_PREFILL_TIMING \
                  STRATA_PREFILL_MEMORY_REPORT STRATA_PREFILL_LEND_PCT; do
  [ -z "${!tuning_key:-}" ] || ARGS+=(-e "$tuning_key=${!tuning_key}")
done
[ "$CHECK_ONLY" = 1 ] && ARGS+=(--entrypoint bash -e "STRATA_AUTO_PREPARE=1")
[ ${#EXTRA_ENV[@]} -gt 0 ] && ARGS+=("${EXTRA_ENV[@]}")
if [ "$CHECK_ONLY" = 1 ]; then
  ARGS+=("$IMAGE" /opt/strata/docker/bootstrap-model.sh check)
elif [ "$DETACH" = 1 ]; then
  ARGS+=(-d "$IMAGE")
else
  ARGS+=(-it "$IMAGE")
fi

log "starting $MODEL on $ARCH (server on http://$BIND:$PORT, logs: $WORK/logs/)"
[ "$DRY" = 1 ] || mkdir -p "$WORK/logs" "$WORK/packs"
note "first start downloads/packs what is missing, then maps tens of GB: 1-3 minutes of a slow PC is normal"
if [ "$DETACH" = 1 ]; then
  xrun "${ARGS[@]}"
  [ "$DRY" = 1 ] || { note "container $NAME;  logs: docker logs -f $NAME"; note "ready check: ./run.sh --check"; }
else
  xrun "${ARGS[@]}"
fi
