#!/usr/bin/env bash
#
# Build the Strata container image.
#
#   docker/build.sh                     build strata:local for the GPUs in this machine
#   docker/build.sh strata:v1           build with that tag
#   ARCHS=86 docker/build.sh            override the CUDA architectures (see below)
#   NO_CACHE=1 docker/build.sh          rebuild every layer
#
# The engine bakes one cubin per CUDA architecture listed, and picks one at run time.  The
# default is read from nvidia-smi here, so the image carries the code for the card it was
# built on and builds in the least time; a card outside the set needs a rebuild with its own
# arch (86 = RTX 30, 89 = RTX 40, 120 = RTX 50, 75 = RTX 20, 80 = A-series).  Strata refuses
# anything below 75.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-${IMAGE_TAG:-strata:local}}"
DOCKERFILE="$ROOT/docker/Dockerfile"

command -v docker >/dev/null 2>&1 || { echo "docker is not installed" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "the docker daemon is not reachable (is it running?)" >&2; exit 1; }
[ -f "$DOCKERFILE" ] || { echo "missing $DOCKERFILE" >&2; exit 1; }

# CUDA architectures, from the cards in this machine unless ARCHS says otherwise.
archs="${ARCHS:-}"
if [ -z "$archs" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    # 8.9 -> 89; several different cards -> "86;89"
    archs="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
      | tr -d ' ' | sed '/^$/d' | sed 's/\.//' | sort -u | paste -sd';' -)"
  fi
  if [ -z "$archs" ]; then
    echo "note: nvidia-smi did not answer; building for 89 (RTX 40). Override with ARCHS=<archs>." >&2
    archs="89"
  fi
fi

# The lowest arch in the set is what the engine must at least support; failing here beats
# failing 10 minutes into the build.
for a in ${archs//;/ }; do
  case "$a" in *[!0-9]*) echo "ARCHS='$archs' is not a ';'-separated list of CUDA arch numbers" >&2; exit 1;; esac
  if [ "$a" -lt 75 ]; then
    echo "CUDA arch $a is below 75: Strata needs an RTX 20 series or newer card." >&2
    exit 1
  fi
done

echo "building $TAG"
echo "  context:   $ROOT"
echo "  dockerfile: docker/Dockerfile"
echo "  CUDA archs: $archs"

# shellcheck disable=SC2086
docker build \
  -f "$DOCKERFILE" \
  -t "$TAG" \
  --build-arg "CUDA_ARCHITECTURES=$archs" \
  ${NO_CACHE:+--no-cache} \
  "$ROOT"

echo
echo "built $TAG"
docker image inspect "$TAG" --format '  size: {{.Size}} bytes' 2>/dev/null || true
echo "  run it:  MODEL_FILE=<model.gguf> docker/run.sh serve"
