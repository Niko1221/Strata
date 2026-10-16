#!/usr/bin/env bash
#
# Run the Strata container.  A thin wrapper around `docker run`: it supplies the flags the
# container needs (the GPU, the model mounted read-only, the data volume, the port published on
# every interface) and hands the rest to you.
#
# The first word is the mode and everything after it belongs to that mode.  A mode is named, not
# guessed from the shape of an argument:
#
#   docker/run.sh serve                    the API server and web app (OpenAI- and Anthropic-compatible)
#   docker/run.sh engine --bench-pp 512    the engine in the foreground, once
#   docker/run.sh --help                   this text
#
# MODEL_FILE=/path/to/model.gguf names the model; it is bind-mounted read-only at
# /models/model.gguf.  There is no built-in default - every machine has its own model file.
#
#   MODEL_FILE=model.gguf docker/run.sh serve
#   CONTEXT=32768 MODEL_FILE=model.gguf docker/run.sh serve
#   MODEL_FILE=model.gguf docker/run.sh engine --bench-pp 512 --bench-tg 8
#
# `serve` takes no arguments: the container's entry point builds the server's command line
# itself, so its settings are environment variables (PORT, CONTEXT, API_KEY, CONFIG,
# ENGINE_ARGS_EXTRA, DATA_DIR, BIND_HOST, NAME, FOREGROUND, START_TIMEOUT, SMOKE).
#
# Engine SWITCHES are the engine's own STRATA_* variables (STRATA_Q36_NO_MMQ, STRATA_MMQ_KQ, ...):
# every STRATA_* variable set here is passed to the container unchanged, whichever mode you run.
# Engine ARGUMENTS go on the command line in `engine` mode; on the server they go through
# ENGINE_ARGS_EXTRA.
#
# Nothing is invented here: there is no MTP, POOL_WORKERS or EXPERT_VRAM_GIB variable and no
# auto-detected setting - the engine's own defaults apply unless you set a STRATA_* variable or
# pass an argument.  The model is never copied into the image.  A symlinked model is resolved
# with readlink -f first, because Docker mounts the directory a symlink points into, not the file
# it points at.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-strata:local}"

die() { echo "error: $*" >&2; exit 1; }

usage() {
  cat <<'EOF'
docker/run.sh serve                    the API server and web app on 0.0.0.0:11434
docker/run.sh engine [ARGS...]         the engine in the foreground, once
docker/run.sh --help                   this text

The model is MODEL_FILE=/path/to/model.gguf, mounted read-only at /models/model.gguf.
In `engine` mode --native /models/model.gguf is added for you unless you name a model
yourself, and `engine --help` prints the engine's own usage.

serve settings are environment variables:
  PORT (11434)  CONTEXT  API_KEY  CONFIG  ENGINE_ARGS_EXTRA  DATA_DIR
  BIND_HOST (0.0.0.0)  NAME (strata)  FOREGROUND (0)  START_TIMEOUT (1800)  SMOKE (0)

serve stays in the foreground and follows the log - Ctrl-C stops the container.  FOREGROUND=1
attaches to the container directly instead (no health check or model summary).

  MODEL_FILE=model.gguf docker/run.sh serve
  CONTEXT=32768 MODEL_FILE=model.gguf docker/run.sh serve
  MODEL_FILE=model.gguf docker/run.sh engine --bench-pp 512 --bench-tg 8
EOF
}

# The first word is the mode: named, never guessed from the shape of an argument.  `--help` as
# the first word prints the text above rather than falling through into the engine's `strata
# generate` usage, and a model path is not read as a mode.
MODE="${1:-serve}"
if [ "$#" -gt 0 ]; then shift; fi
case "$MODE" in
  -h|--help|help) usage; exit 0;;
  serve|engine) ;;
  *) usage >&2; die "unknown mode '$MODE' (expected serve or engine)";;
esac

# `serve` needs no arguments of its own: the container's entry point builds the server's command
# line, so those settings are environment variables.  Said out loud rather than ignored.
if [ "$MODE" = serve ] && [ "$#" -gt 0 ]; then
  if [ "$#" = 1 ]; then
    case "$1" in -h|--help) usage; exit 0;; esac
  fi
  usage >&2
  die "serve takes no arguments: its settings are environment variables (PORT, CONTEXT, API_KEY, CONFIG, ENGINE_ARGS_EXTRA)"
fi

MODEL_FILE="${MODEL_FILE:-}"

PORT="${PORT:-11434}"
# CONTEXT, API_KEY and CONFIG pass through when set; the container picks its own defaults
# otherwise (CONTEXT from the card's VRAM, no API key = reachable only on the host).
API_KEY="${API_KEY:-${STRATA_API_KEY:-}}"
NAME="${NAME:-strata}"
DATA_VOLUME="${DATA_VOLUME:-strata-data}"
FOREGROUND="${FOREGROUND:-0}"
SMOKE="${SMOKE:-0}"
# Where the published port listens on the host.  0.0.0.0 by default: the container's own HOST
# is 0.0.0.0, and publishing on every interface makes the API reachable from other machines,
# which needs an API key (the guard below).  Set BIND_HOST=127.0.0.1 to keep it loopback-only.
BIND_HOST="${BIND_HOST:-0.0.0.0}"

command -v docker >/dev/null 2>&1 || die "docker is not installed"
docker info >/dev/null 2>&1 || die "the docker daemon is not reachable (is it running?)"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE not found - build it first: docker/build.sh"

# `engine --help`, `--version` and `--families` answer without a model and without a GPU: they are
# how a caller asks what an image holds, so they pass before the mount is looked for.  A `--help`
# as the FIRST word (`docker/run.sh --help`) is this script's own help, printed above.
if [ "$MODE" = engine ]; then
  for a in "$@"; do
    case "$a" in
      --help|-h|--version|-V|--families)
        exec docker run --rm --entrypoint /opt/strata/engine/strata "$IMAGE" "$@";;
    esac
  done
fi

# The model: MODEL_FILE, resolved now so the bind mount gets a real file.  Docker mounts the
# directory a symlink points into, not the file it points at, which is why the link is followed
# here rather than left to Docker.
if [ -z "$MODEL_FILE" ]; then
  die "no model given - set MODEL_FILE=/path/to/model.gguf   (docker/run.sh --help)"
fi
if [ ! -e "$MODEL_FILE" ]; then
  case "$MODEL_FILE" in
    /models/*)
      die "$MODEL_FILE does not exist - /models is the mount point inside the container, not a host path.  MODEL_FILE is the host path to the file (e.g. /home/you/models/model.gguf), which is mounted read-only at /models/model.gguf";;
    *) die "$MODEL_FILE does not exist";;
  esac
fi
MODEL_REAL="$(readlink -f "$MODEL_FILE")" || die "cannot resolve $MODEL_FILE"
[ -f "$MODEL_REAL" ] || die "$MODEL_REAL is not a regular file"
if [ "$MODEL_REAL" != "$MODEL_FILE" ]; then
  echo "note: $MODEL_FILE -> $MODEL_REAL (mounting the resolved file)"
fi

# Engine switches: any STRATA_* variable set for this script is handed to the engine (the
# server passes the config's "env" to it; a direct engine run reads the variables itself).
strata_env_args=()
while IFS='=' read -r name _; do
  case "$name" in
    STRATA_*) eval "strata_env_args+=(-e \"$name=\${$name}\")";;
  esac
done < <(env)

# ---- engine: the engine in the foreground, once, with the arguments exactly as given ------
if [ "$MODE" = engine ]; then
  args=("$@")
  # The mounted model is what --native wants, unless the caller named a model of their own.
  named=0
  for a in "${args[@]}"; do
    case "$a" in --native|--model) named=1;; esac
  done
  if [ "$named" = 0 ]; then
    args=(--native /models/model.gguf "${args[@]}")
  fi
  echo "engine: /opt/strata/engine/strata ${args[*]}"
  exec docker run --rm --gpus all --ulimit memlock=-1 --shm-size 1g \
    -v "$MODEL_REAL:/models/model.gguf:ro" \
    "${strata_env_args[@]}" \
    --entrypoint /opt/strata/engine/strata \
    "$IMAGE" "${args[@]}"
fi

# ---- serve: the container's own entry point, which prepares /data and starts the API -------
# The container must bind 0.0.0.0 to be reachable through the published port, so the publish is
# what decides who can reach it: publishing it further without a key would hand the model to
# anything that can reach the port.
if [ "$BIND_HOST" != "127.0.0.1" ] && [ -z "$API_KEY" ]; then
  die "BIND_HOST=$BIND_HOST exposes the API beyond this machine; set API_KEY first"
fi

# Model files can be large; mount the file itself, read-only.
mounts=(-v "$MODEL_REAL:/models/model.gguf:ro")
if [ -n "${DATA_DIR:-}" ]; then
  mkdir -p "$DATA_DIR"
  mounts+=(-v "$(readlink -f "$DATA_DIR"):/data")
else
  mounts+=(-v "$DATA_VOLUME:/data")
fi

env_args=(-e "PORT=$PORT" -e "HOST=0.0.0.0")
[ -n "${CONTEXT:-}" ] && env_args+=(-e "CONTEXT=$CONTEXT")
[ -n "$API_KEY" ] && env_args+=(-e "API_KEY=$API_KEY")
[ -n "${CONFIG:-}" ] && env_args+=(-e "CONFIG=$CONFIG")
[ -n "${ENGINE_ARGS_EXTRA:-}" ] && env_args+=(-e "ENGINE_ARGS_EXTRA=$ENGINE_ARGS_EXTRA")

# The engine's switches (STRATA_*) collected above also reach the server's environment.
env_args+=("${strata_env_args[@]}")

# --gpus all: the engine needs the card.
# --ulimit memlock=-1: the engine pins RAM for the GPU; the project's own instructions set
#   the same limit.  Without it a large model can fail to lock its memory.
# --shm-size: the server passes prompt data between threads through shared memory.
run_args=(--gpus all --ulimit memlock=-1 --shm-size 1g -p "$BIND_HOST:$PORT:$PORT" "${mounts[@]}" "${env_args[@]}")

if [ "$FOREGROUND" = "1" ]; then
  echo "running $IMAGE in the foreground (Ctrl-C stops it)"
  exec docker run --rm --name "$NAME" "${run_args[@]}" "$IMAGE"
fi

# Replace a container of the same name left over from an earlier run.
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "removing the earlier container named $NAME"
  docker rm -f "$NAME" >/dev/null
fi

echo "starting $NAME from $IMAGE"
docker run -d --name "$NAME" "${run_args[@]}" "$IMAGE" >/dev/null

# From here the script owns the running container: Ctrl-C (or SIGTERM) stops it rather than
# leaving it running in the background.
stop_on_interrupt() {
  if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo
    echo "stopping $NAME"
    docker stop "$NAME" >/dev/null 2>&1 || true
  fi
}
trap stop_on_interrupt INT TERM

echo "  model:    $MODEL_REAL"
echo "  data:     ${DATA_DIR:-volume $DATA_VOLUME}"
echo "  api:      http://$BIND_HOST:$PORT/v1"
echo "  logs:     docker logs -f $NAME"
echo "  stop:     Ctrl-C (or docker stop $NAME)"
[ -n "${CONTEXT:-}" ] && echo "  context:  $CONTEXT"

# ---- wait for the model, then say what the API serves ---------------------------------------
# The port only opens after the model is loaded, which takes minutes (longer on a cold page
# cache).  Poll /health rather than the port: it is answered before the API key gate.
printf '\nwaiting for the model to load'
deadline=$(( $(date +%s) + ${START_TIMEOUT:-1800} ))
until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    printf '\n'
    echo "the container exited before it was healthy. Last lines:" >&2
    docker logs --tail 40 "$NAME" >&2 || true
    exit 1
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    printf '\n'
    echo "still not healthy after ${START_TIMEOUT:-1800}s; the model may be larger than this PC can hold." >&2
    echo "follow it with: docker logs -f $NAME" >&2
    exit 1
  fi
  printf '.'
  sleep 5
done
printf ' up\n\n'

auth=()
[ -n "$API_KEY" ] && auth=(-H "Authorization: Bearer $API_KEY")

models="$(curl -fsS "http://127.0.0.1:$PORT/v1/models" "${auth[@]}")" || { echo "GET /v1/models failed" >&2; exit 1; }
echo "models: $(echo "$models" | python3 -c 'import json,sys; print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))')"

# The end-to-end check is opt-in (SMOKE=1): it puts the model through a real request and costs
# GPU time, which is not a wrapper's job.  It stays one variable away because it is the one thing
# that proves the whole path - tokenizer, engine and API - works.
if [ "$SMOKE" = "1" ]; then
  # max_tokens is generous on purpose: this model thinks before it answers, and a small budget is
  # spent entirely on the reasoning block, leaving the visible answer empty.
  echo
  echo "request: POST /v1/chat/completions"
  response="$(curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' "${auth[@]}" \
    -d '{"model":"strata","messages":[{"role":"user","content":"Reply with exactly: hello world"}],"max_tokens":256,"temperature":0}')" \
    || { echo "the request failed. Last lines:" >&2; docker logs --tail 40 "$NAME" >&2 || true; exit 1; }

  echo "$response" | python3 -c '
import json, sys
d = json.load(sys.stdin)
c = d["choices"][0]
m = c["message"]
content = m.get("content")
reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
print("  finish:", c.get("finish_reason"))
print("  answer:", repr(content))
if reasoning:
    print("  thinking:", repr(reasoning[:120]) + ("..." if len(reasoning) > 120 else ""))
print("  usage: ", d.get("usage"))
if not content and not reasoning:
    sys.exit("the answer had neither content nor reasoning")
'
fi

echo
echo "done: $NAME is serving."
echo "      browse http://$BIND_HOST:$PORT   |   base URL for apps: http://$BIND_HOST:$PORT/v1"
echo
echo "following $NAME's log - Ctrl-C stops the container"
docker logs -f "$NAME"
