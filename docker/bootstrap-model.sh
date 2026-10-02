#!/usr/bin/env bash
# Check the model a strata-hip container needs, and get what is missing.
#
#   bootstrap-model.sh prepare   # default: download / build anything missing, then return
#   bootstrap-model.sh check     # never write anything: report and exit 1 if work is pending
#
# Everything is keyed on the host's Hugging Face cache (mounted at $STRATA_HF_CACHE, default
# /hf-cache): the quant is read from there, and `hf download` writes back into it, so a model
# fetched by one container is instantly visible to the host and to every other container.
#
# Steps, each skipped when its artifact is already there:
#   1. the quant's two GGUF shards          hf download <repo> --include '<quant>/*'   (~76 GB)
#   2. <pack>/tokenizer/                    tools/strata_tokenizer.py
#   3. <pack>/native_experts.txt, experts.bin  tools/iq_pack.py --experts-bin          (~43 GB)
#   4. the MTP draft layer                  tools/mtp_fetch.py + mtp_pack.py + mtp_rt.py  (~5 GB)
# Steps 2-3 are what makes the pack the engine starts from (docs/AMD_HIP.md:61-67); step 4 is
# optional - without it the entrypoint starts with --spec 0 instead of failing.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$DIR")"                                # /opt/strata
PY="${PYTHON:-python3}"
MODE="${1:-prepare}"

log() { printf 'bootstrap: %s\n' "$*"; }
warn() { printf 'bootstrap: note: %s\n' "$*" >&2; }
die() { printf 'bootstrap: ERROR: %s\n' "$*" >&2; exit 1; }

MODEL="${STRATA_MODEL:-IQ3_XXS}"
HF_CACHE="${STRATA_HF_CACHE:-${HF_HUB_CACHE:-/hf-cache}}"
WORK="${STRATA_WORK:-/work}"
PACK="${STRATA_PACK_DIR:-$WORK/packs/$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')}"
MTP="${STRATA_MTP:-$WORK/mtp/rt}"
MTP_DIR="$(dirname "$MTP")"
RESOLVE=("$PY" "$DIR/hfmodel.py" --model "$MODEL" --cache "$HF_CACHE" --print shell --allow-missing)
[ -n "${STRATA_HF_REPO:-}" ] && RESOLVE+=(--repo "$STRATA_HF_REPO")
[ -n "${STRATA_HF_REV:-}" ] && RESOLVE+=(--rev "$STRATA_HF_REV")

resolve() { eval "$("${RESOLVE[@]}")"; }
gate_space() {   # gate_space <path> <gib needed> <what>
  local path="$1" need="$2" what="$3" avail
  avail="$(df -BG --output=avail "$path" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)"
  [ "${avail:-0}" = 0 ] && { warn "cannot read free space for $path - continuing"; return 0; }
  if [ "${avail%.*}" -lt "${need%.*}" ]; then
    die "$what needs ~${need} GB but $path has only ${avail} GB free.  /hf-cache and /work are bind
       mounts of host directories chosen by ./run.sh (defaults: ~/Development/models and
       ~/Development/strata-work), so the room has to exist there:
         ./run.sh --hf-cache <bigger>/models --work <bigger>/strata-work
       $MODEL in total wants ~$STRATA_DOWNLOAD_GB GB downloaded plus ~$STRATA_ARENA_GB GB of pack."
  fi
  log "space: $path has ${avail} GB free, $what needs ~${need} GB"
}

# ---------------------------------------------------------------- 1. the model
resolve
log "model $MODEL (repo $STRATA_REPO), HF cache $STRATA_HF_CACHE: $([ "$STRATA_CACHED" = 1 ] \
    && echo 'cached' || echo 'NOT cached yet')"
if [ "$STRATA_CACHED" != 1 ]; then
  if [ "$MODE" = check ]; then
    echo "$MODEL is not in the HF cache; run the container without STRATA_AUTO_PREPARE=0 to fetch it," >&2
    echo "or fetch it on the host: hf download $STRATA_REPO --include '$STRATA_HF_INCLUDE'" >&2
    exit 1
  fi
  [ "${STRATA_DOWNLOAD_MODEL:-1}" = 1 ] || die "$MODEL is not in $HF_CACHE and STRATA_DOWNLOAD_MODEL=0.
       Host side:  HF_HUB_CACHE=$HF_CACHE hf download $STRATA_REPO --include '$STRATA_HF_INCLUDE'"
  command -v hf >/dev/null || die "the 'hf' client is missing from this image (pip install huggingface-hub)"
  # The two shards are one release: a partial download is resumable, but never usable.
  gate_space "$HF_CACHE" "$(awk -v n="$STRATA_DOWNLOAD_GB" 'BEGIN{printf "%d", n + 28}')" "the $MODEL download (incl. a 20 GB free-space floor)"
  [ -n "${HF_TOKEN:-}" ] || warn "no HF_TOKEN: releases that need license acceptance answer 401 without it"
  log "downloading '$STRATA_HF_INCLUDE' from $STRATA_REPO into $HF_CACHE (~$STRATA_DOWNLOAD_GB GB, resumable)"
  export HF_HUB_CACHE="$HF_CACHE"
  hf download "$STRATA_REPO" --include "$STRATA_HF_INCLUDE" ${HF_REVISION:+--revision "$HF_REVISION"} \
    || die "hf download failed (network, or a gated repo needing HF_TOKEN). It is resumable: just restart."
  resolve
  [ "$STRATA_CACHED" = 1 ] || die "after downloading, $MODEL still has no usable shard pair in $HF_CACHE"
fi
log "shard 1 (experts):   $STRATA_SHARD1"
log "shard 2 (PLE table): $STRATA_SHARD2"
if [ "$MODE" = check ]; then log "model: ok"; fi

# ---------------------------------------------------------------- 2-3. the pack the engine starts from
need_pack=0
[ -f "$PACK/native_experts.txt" ] || need_pack=1
[ -f "$PACK/tokenizer/vocab.json" ] || need_pack=1
[ -f "$PACK/experts.bin" ] || need_pack=1          # the HIP mmap path needs it (docs/AMD_HIP.md:64)
if [ "$need_pack" = 1 ]; then
  if [ "$MODE" = check ]; then echo "pack $PACK is incomplete; the container builds it on first start" >&2; exit 1; fi
  mkdir -p "$PACK"
  gate_space "$PACK" "$(awk -v n="$STRATA_ARENA_GB" 'BEGIN{printf "%d", n + 26}')" "the pack (experts.bin)"
  if [ ! -f "$PACK/tokenizer/vocab.json" ]; then
    log "writing the tokenizer into $PACK/tokenizer ..."
    "$PY" "$REPO_ROOT/tools/strata_tokenizer.py" --gguf "$STRATA_SHARD1" --out "$PACK" \
      || die "strata_tokenizer.py failed"
  fi
  if [ ! -f "$PACK/native_experts.txt" ] || [ ! -f "$PACK/experts.bin" ]; then
    log "packing the model (reads all of shard 1; writes ~$STRATA_ARENA_GB GB) ..."
    "$PY" "$REPO_ROOT/tools/iq_pack.py" --gguf "$STRATA_SHARD1" --out "$PACK" --experts-bin \
      || die "iq_pack.py failed (see its output above)"
    if [ ! -f "$PACK/native_experts.txt" ]; then      # the flag alone may not emit the native index
      log "re-packing without --experts-bin to write the native expert index ..."
      "$PY" "$REPO_ROOT/tools/iq_pack.py" --gguf "$STRATA_SHARD1" --out "$PACK" || die "iq_pack.py failed"
    fi
  fi
fi
[ -f "$PACK/tokenizer/vocab.json" ] || die "pack $PACK has no tokenizer/vocab.json"
[ -f "$PACK/native_experts.txt" ] || die "pack $PACK has no native_experts.txt"
log "pack: $PACK"
if [ "$MODE" = check ]; then log "pack: ok"; fi

# ---------------------------------------------------------------- 4. the MTP draft layer (optional)
if [ ! -f "$MTP/experts.bin" ]; then
  if [ "$MODE" = check ]; then warn "no MTP draft layer at $MTP (--spec 0 without it)"; else
    mkdir -p "$MTP_DIR"
    gate_space "$MTP_DIR" 15 "the MTP draft layer"
    # gguf-py comes from the pinned llama.cpp checkout that built the engine (tools/_paths.py:16).
    export STRATA_GGUF_PY="${STRATA_GGUF_PY:-$REPO_ROOT/gguf-py}"
    log "fetching the ~5 GB of MTP tensors from the original Qwen checkpoint and packing them ..."
    if ! "$PY" "$REPO_ROOT/tools/mtp_fetch.py" fetch --out "$MTP_DIR" \
       || ! "$PY" "$REPO_ROOT/tools/mtp_pack.py" --src "$MTP_DIR" --experts q2_0 --out "$MTP_DIR/mtp-q2_0.gguf" \
       || ! "$PY" "$REPO_ROOT/tools/mtp_rt.py" --gguf "$MTP_DIR/mtp-q2_0.gguf" --out "$MTP"; then
      warn "the MTP draft layer could not be prepared (strata-hip starts with --spec 0, so decoding is"
      warn "slower than the published numbers). STRATA_GGUF_PY=$STRATA_GGUF_PY must point at llama.cpp's gguf-py."
    else
      log "MTP draft layer: $MTP"
    fi
  fi
else
  log "MTP draft layer: $MTP"
fi
log "ready"
