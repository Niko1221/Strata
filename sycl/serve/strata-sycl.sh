#!/usr/bin/env bash
# strata-sycl.sh: the SYCL-built engine as a drop-in `exe` for serve/server.py (--engine strata). The port's binary
# needs the oneAPI runtime of the dev image, so this runs it there with stdin/stdout attached (the serve protocol is
# lines on those pipes) and stderr going to the server's log. Paths in the config's args are the container's:
# the data root is mounted at /work.
#   STRATA_SYCL_ROOT   host directory mounted at /work        (default: two levels above this script's repo)
#   STRATA_SYCL_IMAGE  the runtime image                      (default: strata-sycl-dev)
#   STRATA_SYCL_BIN    the engine binary, relative to the repo (default: build-sycl-aot/strata)
#   STRATA_SYCL_NAME   the container's name                   (default: strata-sycl-serve)
#   ONEAPI_DEVICE_SELECTOR  passed in when set; otherwise level_zero:gpu (every card) for a --layer-split, and the
#                      image's level_zero:0 for one card (#423)
#   STRATA_SYCL_RUNNER docker (the image, default); native - already inside a toolbox with oneAPI where the data root
#                      is /work (the server runs there too); distrobox:<name> - from the host, in that distrobox
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)                 # the repo
root=${STRATA_SYCL_ROOT:-$(dirname "$here")}
repo_in=/work/$(basename "$here")
name=${STRATA_SYCL_NAME:-strata-sycl-serve}
args=""
split=""
for a in "$@"; do args+=" $(printf '%q' "$a")"; [ "$a" = --layer-split ] && split=1; done
selv=${ONEAPI_DEVICE_SELECTOR:-${split:+level_zero:gpu}}
sel=()
[ -n "$selv" ] && sel=(-e "ONEAPI_DEVICE_SELECTOR=$selv")
# the port's run-time switches: the device-built verify plan without host handshakes (docs/INTEL.md)
env="STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1 STRATA_STAGER_THREADS=12 ONEAPI_DEVICE_SELECTOR=${selv:-level_zero:0}"
run="set +u; . /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; export $env; cd $repo_in && exec ${STRATA_SYCL_BIN:-build-sycl-aot/strata}$args"
case "${STRATA_SYCL_RUNNER:-docker}" in
native) exec bash -c "$run" ;;
distrobox:?*) exec distrobox enter "${STRATA_SYCL_RUNNER#distrobox:}" -- bash -c "$run" ;;
esac
docker rm -f "$name" >/dev/null 2>&1 || true              # a container left behind by a killed server
exec docker run --rm -i --name "$name" --device /dev/dri --oom-score-adj 1000 --stop-timeout 30 --no-healthcheck \
    -v "$root:/work" \
    -e STRATA_VERIFY_DEVICE_PLAN=1 -e STRATA_VERIFY_NO_HOST=1 -e STRATA_STAGER_THREADS=12 \
    "${sel[@]}" \
    "${STRATA_SYCL_IMAGE:-strata-sycl-dev}" \
    "cd $repo_in && exec ${STRATA_SYCL_BIN:-build-sycl-aot/strata}$args"
