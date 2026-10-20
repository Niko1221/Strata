#!/bin/sh
# Entry point of the Strata container: `strata launch --no-tui` with the settings the TUI asks for, read from environment
# variables, then stay up and stop cleanly on SIGTERM (`docker stop`).
#
#   docker run ... strata              start the server (default)
#   docker run ... strata strata ...   run any strata command instead (e.g. `strata launch --plan --model q2_0`)
#   docker run ... strata sh           a shell
#
# Settings (all optional; names match the screens of `strata launch`):
#   MODEL         model size, e.g. IQ3_XXS, Q2_0 (empty: Strata picks one for this card and says so)
#   FAMILY        model family when a size is offered by several (qwen, swift, coder, ...)
#   MODEL_PATH    GGUF files you already have, a folder or one .gguf, inside the container (mount it read-only)
#   CONTEXT       32k, 64k, 128k, 256k or a number of tokens
#   PARALLEL      requests answered at the same time, 1 to 8 (empty: fit to the card)
#   VISION        off | on | gpu | cpu  (pictures; the picture reader file is downloaded with the model)
#   KV            int8 | q4_0 | k8v4 | fp16  (conversation memory format)
#   GPU / GPUS    card numbers, e.g. 0 or 0,1 (empty or "all": the engine's choice)
#   TUNE          quick | thorough | skip   (done once; RETUNE=1 tunes again)
#   PORT          port inside the container (default 8080)
#   API_KEY / API_KEY_FILE   the key clients must send. The server listens on every address inside the container, and Strata does
#                 not serve beyond loopback without a key, so one is always on: yours, or one made on the first start and kept in /data.
#   PRESET        preset name in /data (default "docker"). Settings above are saved into it; with no settings given a saved preset is reused.
#   STRATA_SET    advanced engine settings, space separated KEY=VALUE (like `strata launch --set`)
set -u

APP=/opt/strata
STRATA="$APP/terminal/bin/strata"
PYBIN="$APP/terminal/python/bin/python3"; [ -x "$PYBIN" ] || PYBIN=$(command -v python3 || echo python3)
STATE_DIR="${XDG_STATE_HOME:-/data/state}/strata"
KEY_FILE="${STRATA_KEY_FILE:-/data/state/api-key}"
PHASE_FILE=/run/strata-phase
PRESET="${PRESET:-docker}"
PORT="${PORT:-8080}"

say()  { printf '%s\n' "$*"; }
warn() { printf '[!] %s\n' "$*" >&2; }
die()  { printf '[X] %s\n' "$*" >&2; exit "${2:-1}"; }
phase() { printf '%s\n' "$1" > "$PHASE_FILE" 2>/dev/null || true; }

# ---- other commands: `docker run image strata ...`, `docker run image sh`
if [ "$#" -gt 0 ] && [ "$1" != "run" ] && [ "$1" != "serve" ]; then
  case "$1" in strata) shift; exec "$STRATA" "$@" ;; *) exec "$@" ;; esac
fi

mkdir -p /data/state /data/home /data/install /data/Strata-data /run 2>/dev/null || die "/data is not writable. Mount a volume there: -v strata-data:/data"
# llama.cpp at the pinned commit is part of the image: link it into the install folder, so the first start fetches nothing from GitHub
if [ -d /opt/strata/seed/third_party/llama.cpp ] && [ ! -e /data/install/third_party/llama.cpp ]; then
  mkdir -p /data/install/third_party && ln -s /opt/strata/seed/third_party/llama.cpp /data/install/third_party/llama.cpp
fi
if ! grep -q " /data " /proc/self/mountinfo 2>/dev/null; then
  warn "/data is not a volume, so the model and your settings are lost when this container is removed. Add: -v strata-data:/data"
fi

# ---- plain warnings (never refusals)
cpuinfo="${STRATA_CPUINFO:-/proc/cpuinfo}"
if [ -r "$cpuinfo" ] && grep -q '^flags' "$cpuinfo" && ! grep -m1 '^flags' "$cpuinfo" | grep -qw avx2; then
  warn "This computer's CPU has no AVX2. The engine is built for CPUs with AVX2 and may stop with 'illegal instruction'."
fi
limit=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
case "$limit" in max|"") ;; *[!0-9]*) ;; *)
  gb=$((limit / 1073741824))
  warn "This container has a memory limit of ${gb} GB. Strata plans from the host's RAM (/proc/meminfo) and loads tens of GB, so a low limit can end in the container being killed. Raise or remove the limit."
esac
if [ -n "${API_KEY:-}" ] && [ "${#API_KEY}" -lt 16 ]; then
  warn "API_KEY is short (${#API_KEY} characters). Use 16 or more; 'openssl rand -hex 24' makes a good one."
fi
for legacy in HOST LOW_RAM REINSTALL RESIDENT_BUDGET_GIB KV_STREAMING LAYER_SPLIT CONFIG; do
  eval "v=\${$legacy:-}"
  [ -n "$v" ] && warn "$legacy is from the older image and is ignored here. See docs/DOCKER.md (\"From the older image\")."
done

# ---- is there an engine for this card?
phase checking
engine_json=$("$STRATA" components engine --output json 2>/dev/null || true)
if ! printf '%s' "$engine_json" | grep -q '"usable": *true'; then
  echo "[X] The engine in this image does not match this computer's graphics card, or the card is not visible to the container." >&2
  "$STRATA" components engine 2>&1 | sed 's/^/    /' >&2 || true
  cat >&2 <<'MSG'
    Check, in this order:
      NVIDIA  docker run --gpus all ... (needs nvidia-container-toolkit on the host; driver 580+ for the cuda13 image, 575+ for cuda12)
      AMD     docker run --device /dev/kfd --device /dev/dri --group-add video --group-add render ...
      Intel   docker run --device /dev/dri ... (add --group-add for the host's render group)
    Then pick the image for your card: docs/DOCKER.md.
MSG
  exit 3
fi

# the card is found through sysfs even when no device file was passed in: say so now instead of failing when the engine starts
backend=$(printf '%s' "$engine_json" | sed -n 's/.*"backend": *"\([a-z0-9]*\)".*/\1/p' | head -1)
case "$backend" in
  hip)  [ -e /dev/kfd ] || { echo "[X] This is an AMD card, but /dev/kfd is not in the container. Add: --device /dev/kfd --device /dev/dri --group-add video --group-add <render group number> (docs/DOCKER.md)" >&2; exit 3; } ;;
  sycl) ls /dev/dri/renderD* >/dev/null 2>&1 || { echo "[X] This is an Intel card, but /dev/dri is not in the container. Add: --device /dev/dri --group-add <render group number> (docs/DOCKER.md)" >&2; exit 3; } ;;
esac

# ---- the API key
key="${API_KEY:-}"
if [ -z "$key" ] && [ -n "${API_KEY_FILE:-}" ]; then
  [ -r "$API_KEY_FILE" ] || die "API_KEY_FILE=$API_KEY_FILE cannot be read."
  key=$(tr -d '\r\n' < "$API_KEY_FILE")
fi

# ---- settings -> `strata launch` flags
set -- launch --no-tui --access lan --port "$PORT" --data-dir /data/Strata-data
have=""

preset_exists=""
if "$STRATA" launch --list-presets --output json 2>/dev/null | grep -q "\"name\": *\"$PRESET\""; then preset_exists=1; fi

model_path="${MODEL_PATH:-${GGUF_DIR:-}}"
[ -n "${MODEL:-}" ]       && { set -- "$@" --model "$MODEL"; have=1; }
[ -n "${FAMILY:-}" ]      && { set -- "$@" --family "$FAMILY"; have=1; }
[ -n "$model_path" ]      && { set -- "$@" --model-path "$model_path"; have=1; }
[ -n "${CONTEXT:-}" ]     && { set -- "$@" --context "$CONTEXT"; have=1; }
[ -n "${PARALLEL:-}" ] && [ "${PARALLEL}" != "auto" ] && { set -- "$@" --parallel "$PARALLEL"; have=1; }
[ -n "${KV:-}" ]          && { set -- "$@" --kv "$KV"; have=1; }
if [ -n "${VISION:-}" ]; then
  case "$(printf '%s' "$VISION" | tr 'A-Z' 'a-z')" in
    off|no|false|0) v=off ;; on|yes|true|1) v=on ;; gpu) v=gpu ;; cpu) v=cpu ;;
    *) die "VISION=$VISION is not one of off, on, gpu, cpu." 2 ;;
  esac
  set -- "$@" --vision "$v"; have=1
fi
gpus="${GPUS:-${GPU:-}}"
case "$gpus" in ""|all|ALL) ;; *)
  set -- "$@" --gpu "$gpus"; have=1
  # `launch --gpu` is applied when a model is first set up; a model that is already set up keeps its card. So that GPU / GPUS take effect on
  # a volume that already has the model, run settings that name another card are dropped: setup writes them again from the files that are there.
  want=$(printf '%s' "$gpus" | tr -d ' ')
  for c in /data/install/strata-*.json; do
    [ -f "$c" ] || continue
    cur=$("$PYBIN" -c 'import json, sys; g = json.load(open(sys.argv[1])).get("gpu"); print(",".join(map(str, g)) if isinstance(g, list) else ("" if g is None else g))' "$c" 2>/dev/null)
    if [ "$cur" != "$want" ]; then
      say "The saved run settings use graphics card '${cur:-default}', GPU/GPUS asks for '$want': making them again (model files are kept)."
      rm -f "$c"
    fi
  done ;;
esac
[ -n "${STRATA_SET:-}" ] && { for kv in $STRATA_SET; do set -- "$@" --set "$kv"; done; have=1; }
[ -n "$key" ] && { set -- "$@" --api-key-env STRATA_LAUNCH_KEY; }

# tune once: ask only when this preset has not been tuned yet (a restart must not tune again for 20 minutes)
tune_mode=$(printf '%s' "${TUNE:-}" | tr 'A-Z' 'a-z')
if [ -n "$tune_mode" ] && [ "$tune_mode" != "skip" ]; then
  tuned=""
  if [ -n "$preset_exists" ] && "$STRATA" launch --list-presets 2>/dev/null | grep -E "^$PRESET[[:space:]]" | grep -qv "not tuned"; then tuned=1; fi
  if [ -n "$tuned" ] && [ "${RETUNE:-0}" != "1" ]; then
    say "This preset is already tuned; keeping the tuned settings (set RETUNE=1 to tune again)."
  else
    set -- "$@" --tune "$tune_mode"; have=1
    say "Tuning is on (TUNE=$tune_mode): the first start takes longer (up to about $([ "$tune_mode" = quick ] && echo 20 || echo 120) minutes), once."
  fi
fi
[ "${TUNE_ALLOW_SMALL:-0}" = "1" ] && set -- "$@" --tune-allow-small

if [ -n "$preset_exists" ]; then
  set -- "$@" --preset "$PRESET"
  [ -n "$have" ] && set -- "$@" --save-preset "$PRESET"
elif [ -n "$have" ]; then
  set -- "$@" --save-preset "$PRESET"
else
  warn "No MODEL was set, so Strata picks the model it recommends for this card. Set MODEL (for example MODEL=IQ3_XXS) to choose."
  set -- "$@" --save-preset "$PRESET"
fi

# ---- start; `docker stop` during the download or the load ends it cleanly, after serving it drains and stops the model
stopping=""
launch_pid=""
on_term() {
  stopping=1
  phase stopping
  say "Stop requested: shutting down."
  if [ -n "$launch_pid" ] && kill -0 "$launch_pid" 2>/dev/null; then
    kill -INT "$launch_pid" 2>/dev/null     # the launch is still downloading or loading: it stops what it started
    i=0; while kill -0 "$launch_pid" 2>/dev/null && [ "$i" -lt 30 ]; do sleep 1; i=$((i+1)); done
    kill -TERM "$launch_pid" 2>/dev/null || true
  fi
  "$STRATA" launch --stop --preset "$PRESET" || true
  exit 0
}
trap on_term TERM INT

# A command started with & by a non-interactive sh ignores SIGINT, so `kill -INT` would do nothing. The launch must see it (it is how it
# stops a download or a load cleanly), so it is started through a one-line Python that puts SIGINT back before it runs the Terminal.
RESET_INT='import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execv(sys.argv[1], sys.argv[1:])'

phase preparing
say "Starting Strata: ${MODEL:-recommended model}${model_path:+ from $model_path} (preset '$PRESET')."
if [ -n "$key" ]; then
  STRATA_LAUNCH_KEY="$key" "$PYBIN" -c "$RESET_INT" "$STRATA" "$@" &
else
  "$PYBIN" -c "$RESET_INT" "$STRATA" "$@" &
fi
launch_pid=$!
wait "$launch_pid"
rc=$?
launch_pid=""
[ -n "$stopping" ] && exit 0
if [ "$rc" -ne 0 ]; then
  phase failed
  die "strata launch stopped with code $rc (the lines above say why). The container ends; a restart continues where it stopped (downloads resume)." "$rc"
fi

# keep the key where `docker exec ... strata monitor` and the health check can read it (private file on the volume)
( umask 077; "$STRATA" launch --show-key --preset "$PRESET" > "$KEY_FILE" 2>/dev/null ) || true
[ -s "$KEY_FILE" ] || rm -f "$KEY_FILE"
if [ -z "$key" ]; then
  warn "No API_KEY was set, so Strata made one (the server listens on all addresses in the container and will not run without a key)."
  warn "Show it with: docker exec <container> strata launch --show-key --preset $PRESET   Set your own with -e API_KEY=..."
fi
phase serving
say "Serving on port $PORT. Watch it: docker exec -it <container> strata monitor    Stop: docker stop <container>"

# ---- stay up; leave (so a restart policy can act) if the server is gone
bad=0
while [ -z "$stopping" ]; do
  sleep 10 &
  wait $!
  [ -n "$stopping" ] && break
  if "$STRATA" launch --status --output json 2>/dev/null | grep -q '"state": *"ready"'; then bad=0; else bad=$((bad+1)); fi
  if [ "$bad" -ge 3 ]; then
    phase failed
    die "The server is no longer running. Last lines of its log are in /data/install and 'docker logs'." 1
  fi
done
exit 0
