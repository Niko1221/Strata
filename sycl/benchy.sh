#!/usr/bin/env bash
# benchy v1: the standard benchmark of Strata's SYCL engine on an Intel Arc, so every card runs the same bench.
#
#   sycl/benchy.sh [--configs strata-a.json,...] [--sizes 20,2185,...] [--warm] [--force] [--out DIR]
#
# What it runs:
# - Every model you set up with sycl/setup_intel.py (each strata-*.json next to the checkout with "backend": "sycl";
#   steering variants are left out), each with its own serve config, unchanged.
# - The v1 prompts (sycl/bench/v1): 20, 2,185, 8,000, 40,000, 128,000 and 256,000 tokens, each followed by 256
#   greedy tokens. A size that does not fit the config's --max-context is skipped and listed.
# - Each run is a fresh engine process, after the page cache is dropped (root, so it re-runs itself with sudo; --warm
#   skips the drop and the sudo, and the table says it was warm).
#
# The engine runs in the strata-sycl-dev image (STRATA_SYCL_RUNNER=docker, or podman); distrobox:<name> runs it in a
# distrobox with oneAPI,
# where /work is the data root (the directory that holds the checkout), as in the image.
#
# Stop the served model first: benchy refuses to start while more than 2 GB of VRAM is in use (--force overrides).
# A full run takes 30-60 minutes on a B70 (the 128K and 256K prompts are most of it).
#
# Post sycl/benchy-results/v1-<date>/matrix.md and its config-*.txt files with your results (docs/INTEL_PERFORMANCE.md,
# "Submitting numbers"). The bench's prompts, sizes and columns stay fixed within a version; a change is a new version.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
warm=0
for a in "$@"; do [ "$a" = --warm ] && warm=1; done
case "${STRATA_SYCL_RUNNER:-docker}" in
docker)
    command -v docker >/dev/null || { echo "benchy: docker is not installed (the engine runs in the oneAPI image;" \
        "STRATA_SYCL_RUNNER=distrobox:<name> runs it in a distrobox with oneAPI instead)" >&2; exit 2; }
    docker image inspect "${STRATA_SYCL_IMAGE:-strata-sycl-dev}" >/dev/null 2>&1 \
        || { echo "benchy: the image ${STRATA_SYCL_IMAGE:-strata-sycl-dev} is missing (sycl/tools/Dockerfile)" >&2; exit 2; } ;;
podman)
    command -v podman >/dev/null || { echo "benchy: podman is not installed" >&2; exit 2; }
    as_user=()   # rootless images are the user's who ran sudo, not root's
    [ "$(id -u)" = 0 ] && [ -n "${SUDO_USER:-}" ] && as_user=(runuser -u "$SUDO_USER" -- env "XDG_RUNTIME_DIR=/run/user/$SUDO_UID")
    "${as_user[@]}" podman image inspect "${STRATA_SYCL_IMAGE:-strata-sycl-dev}" >/dev/null 2>&1 \
        || { echo "benchy: the image ${STRATA_SYCL_IMAGE:-strata-sycl-dev} is missing (podman build -t strata-sycl-dev" \
             "-f sycl/tools/Dockerfile sycl/tools)" >&2; exit 2; } ;;
distrobox:?*)
    command -v distrobox >/dev/null || { echo "benchy: distrobox is not installed" >&2; exit 2; } ;;
*)  echo "benchy: STRATA_SYCL_RUNNER is docker, podman or distrobox:<name>" >&2; exit 2 ;;
esac
if [ "$(id -u)" != 0 ] && [ $warm = 0 ]; then
    echo "benchy v1: re-running with sudo (it drops the page cache before each run; --warm to run without)" >&2
    exec sudo --preserve-env=STRATA_SYCL_ROOT,STRATA_SYCL_IMAGE,STRATA_SYCL_RUNNER,STRATA_SYCL_BIN,ONEAPI_DEVICE_SELECTOR "$0" "$@"
fi
rc=0
python3 "$here/tools/perf_matrix.py" "$@" || rc=$?
# the results belong to the user who ran sudo, not root
[ -n "${SUDO_UID:-}" ] && [ -d "$here/benchy-results" ] && chown -R "$SUDO_UID:${SUDO_GID:-$SUDO_UID}" "$here/benchy-results" || true
exit $rc
