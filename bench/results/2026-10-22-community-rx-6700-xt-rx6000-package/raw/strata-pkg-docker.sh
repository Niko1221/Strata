#!/usr/bin/env bash
# Strata release v0.2.2's OFFICIAL image (engine package for the RX 6000, gfx1030/gfx1031, ROCm 7.13 inside the package) on the
# RX 6700 XT, to compare with strata-docker (the source build on ROCm 10.1).
#   strata-pkg-docker [start]   stop the other servers that hold the card, (re)create the container, wait until it serves
#   strata-pkg-docker stop      stop and remove it
#   strata-pkg-docker status    container and /health
#   strata-pkg-docker logs      follow the container log (Ctrl-C stops following, not the server)
#   strata-pkg-docker key       print the API key (this image always requires one)
#
# It does NOT mount your shared models folder as /data: the image's entrypoint moves /data/models, /data/packs and /data/mtp into
# /data/Strata-data (see docs/DOCKER.md, "From the older image"), which would break strata-docker and ub-docker. It uses its own
# data folder ($PKGDATA), mounts only the model's GGUF folder read-only, and publishes the port on 127.0.0.1 only.
# The pack (1.5 GB) and the MTP layer (0.8 GB) are copied once from your existing ones, so nothing is prepared or downloaded.
# Settings: MODEL, FAMILY, CONTEXT, KV, PARALLEL and STRATA_SET (advanced KEY=VALUE engine settings) are passed to the entrypoint.
set -euo pipefail

IMAGE=${IMAGE:-strata-hip:v0.2.2-pkg-rx6000}
NAME=strata-pkg-gfx1031
PORT=${PORT:-8081}
SRC=${STRATA_MODELS:-$HOME/code/Strata/models}
MODEL=${MODEL:-IQ2_XS}
FAMILY=${FAMILY:-qwen}
PKGDATA=${PKGDATA:-$HOME/strata-pkg-data}
KEYFILE=${KEYFILE:-$HOME/.config/strata-pkg/api-key}
GGUF_DIR=$SRC/models/$MODEL
TAG=$(printf '%s' "$MODEL" | tr 'A-Z' 'a-z')

[ -f "$KEYFILE" ] || { mkdir -p "$(dirname "$KEYFILE")"; ( umask 077; openssl rand -hex 24 > "$KEYFILE" ); }
KEY=$(tr -d '\r\n' < "$KEYFILE")

case "${1:-start}" in
  stop)   docker rm -f "$NAME" >/dev/null 2>&1 || true; exit 0 ;;
  status) docker ps --filter "name=^/$NAME$" --format '{{.Names}}\t{{.Status}}'
          curl -s -m 5 -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/health" || echo "(no answer on $PORT)"; echo; exit 0 ;;
  logs)   docker logs -f --tail 50 "$NAME" 2>&1 | tr '\r' '\n'; exit 0 ;;
  key)    echo "$KEY"; exit 0 ;;
  start)  ;;
  *)      echo "usage: strata-pkg-docker [start|stop|status|logs|key]" >&2; exit 2 ;;
esac

docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE not found" >&2; exit 1; }
ls "$GGUF_DIR"/*.gguf >/dev/null 2>&1 || { echo "no .gguf files in $GGUF_DIR" >&2; exit 1; }

# the data folder: the pack and the MTP layer are copied once (never moved) from the shared folder
mkdir -p "$PKGDATA/Strata-data/packs" "$PKGDATA/Strata-data/mtp"
[ -d "$PKGDATA/Strata-data/packs/$TAG" ] || { echo "copying the $TAG pack (once) ..."; cp -a "$SRC/packs/$TAG" "$PKGDATA/Strata-data/packs/$TAG"; }
[ -d "$PKGDATA/Strata-data/mtp/rt" ]     || { echo "copying the MTP layer (once) ...";  cp -a "$SRC/mtp/rt" "$PKGDATA/Strata-data/mtp/rt"; }

VIDEO_GID=$(getent group video | cut -d: -f3); RENDER_GID=$(getent group render | cut -d: -f3)
ENV_ARGS=(-e "MODEL=$MODEL" -e "FAMILY=$FAMILY" -e "MODEL_PATH=/models" -e "PARALLEL=${PARALLEL:-1}" -e "KV=${KV:-int8}"
          -e "CONTEXT=${CONTEXT:-196608}" -e "VISION=${VISION:-off}" -e "API_KEY=$KEY" -e "PORT=8080")
[ -n "${STRATA_SET:-}" ] && ENV_ARGS+=(-e "STRATA_SET=$STRATA_SET")
[ -n "${TUNE:-}" ] && ENV_ARGS+=(-e "TUNE=$TUNE")
for kv in ${STRATA_ENV:-}; do ENV_ARGS+=(-e "$kv"); done

# every other server that holds the card or the port
docker rm -f "$NAME" strata-gfx1031-test unbiased-strata-gfx1031 unbiased-poly-gfx1031 strata >/dev/null 2>&1 || true
docker run -d --name "$NAME" --ulimit core=0 --ulimit memlock=-1 \
  --device=/dev/kfd --device=/dev/dri --group-add "$VIDEO_GID" --group-add "$RENDER_GID" --ipc=host \
  --security-opt label=disable -p "127.0.0.1:$PORT:8080" \
  -v "$PKGDATA:/data" -v "$GGUF_DIR:/models:ro" \
  "${ENV_ARGS[@]}" "$IMAGE" >/dev/null
echo "$NAME started on image $IMAGE; loading (watch: strata-pkg-docker logs) ..."

for _ in $(seq 1 120); do
  sleep 5
  if curl -s -m 3 -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/health" | grep -q '"loaded": *true'; then
    echo "ready: http://127.0.0.1:$PORT/v1  (needs the key: strata-pkg-docker key)"; exit 0
  fi
  [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" = true ] || { echo "container stopped:"; docker logs --tail 30 "$NAME"; exit 1; }
done
echo "still loading - check: strata-pkg-docker logs" >&2; exit 1
