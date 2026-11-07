#!/usr/bin/env bash
# strata-sycl-native.sh: the SYCL engine as the `exe` for serve/server.py, run on the host's own oneAPI (no Docker).
# Made for an integrated Arc (measured on the Arc 140T, Core Ultra 9 285H, Fedora 44, oneAPI 2026.0). The server
# runs it as `strata-sycl-native.sh --serve <args>` and talks to it over stdin/stdout; stderr goes to its log.
#   STRATA_SYCL_BIN    the engine binary, relative to the repo or absolute (default: build-sycl/strata)
#   ONEAPI_ROOT        the oneAPI install         (default: /opt/intel/oneapi)
# The port's run-time switches below are defaults: the config's "env" block overrides them (a value of 0 turns one off).
set -eo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)                # the repo
# setvars.sh reads unset variables, so -u only after it; its output would corrupt the serve protocol on stdout
source "${ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh" >/dev/null 2>&1 || true
set -u
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-level_zero:0}
export SYCL_CACHE_PERSISTENT=${SYCL_CACHE_PERSISTENT:-0}         # docs/INTEL.md: the persistent JIT cache crashed on Xe2
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1                  # the expert cache is one 23 GiB allocation (> 4 GiB)
# as strata-sycl.sh: the device-built verify plan and no host handshakes in it (xe); STRATA_NO_MULTI_GR: the generic
# GR kernels (the multi-token ones decoded 15.1 vs 18.3 tok/s on the 140T in 0.1.40, same output)
for kv in STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1 STRATA_STAGER_THREADS=12 STRATA_NO_MULTI_GR=1; do
    k=${kv%%=*}
    [ -n "${!k+x}" ] || export "$kv"
    # the engine tests some switches by presence: 0 or empty means "not set"
    case "${!k}" in 0|"") unset "$k" ;; esac
done
# The iGPU's memory is system RAM: if the machine runs out, the kernel should kill the engine, not the desktop
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
bin=${STRATA_SYCL_BIN:-build-sycl/strata}
case "$bin" in /*) ;; *) bin=$here/$bin ;; esac
exec "$bin" "$@"
