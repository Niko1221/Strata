#!/usr/bin/env bash
# strata-vision-sycl.sh: the image encoder as a drop-in `vision.exe` for serve/server.py.  The engine has a launcher
# of its own, strata-sycl.sh (the dev image); the encoder is started by the server itself, on the host, and needs
# one for the same reason: a binary built against oneAPI does not resolve libsycl.so, oneMKL or UMF from the
# loader's default path (`ldd` prints "not found").  This composes LD_LIBRARY_PATH, preflights the binary so a
# missing runtime fails with a named fix instead of the loader's message, and execs it with every argument
# unchanged (the server's --mmproj/--model/--gpu/... are passed through as they come).
#
#   STRATA_VISION_BIN    the encoder binary                            (default: the first of
#                                                                     build-vision-sycl/bin/strata-vision and
#                                                                     engine/strata-vision beside this checkout;
#                                                                     setup_intel.py stages the second)
#   STRATA_ONEAPI_ROOT   the oneAPI install to take the runtime from  (default: the one the icpx on PATH belongs
#                                                                     to, else /opt/intel/oneapi)
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)                 # the checkout

bin=${STRATA_VISION_BIN:-}
if [ -z "$bin" ]; then
    for p in "$here/build-vision-sycl/bin/strata-vision" "$here/engine/strata-vision" \
             "$here/build-vision/bin/strata-vision"; do
        [ -x "$p" ] && { bin=$p; break; }
    done
fi
if [ -z "$bin" ]; then
    echo "strata-vision: the image encoder is not built (sycl/tools/build.sh strata-vision, or tools/vision" >&2
    echo "  configured with -DSTRATA_VISION_SYCL=ON); images cannot be read.  STRATA_VISION_BIN=<binary> names" >&2
    echo "  one somewhere else." >&2
    exit 127
fi
case "$bin" in /*) ;; *) bin="$here/$bin" ;; esac
[ -x "$bin" ] || { echo "strata-vision: not executable: $bin" >&2; exit 127; }

# the runtime directories: the compiler's, oneMKL's, UMF's (the Level Zero adapter dlopens it, so the loader
# search still needs it), and hwloc/tbb/redist.  The caller's own entries keep working, after these.
root=${STRATA_ONEAPI_ROOT:-}
if [ -z "$root" ] && command -v icpx >/dev/null 2>&1; then
    root=$(cd "$(dirname "$(command -v icpx)")/../../.." && pwd)     # <root>/compiler/<ver>/bin/icpx
fi
[ -n "$root" ] || root=/opt/intel/oneapi
dirs=""
for d in "$root"/compiler/*/lib "$root"/compiler/*/opt/compiler/lib "$root"/umf/*/lib \
         "$root"/mkl/*/lib "$root"/tcm/*/lib "$root"/tbb/*/lib "$root"/redist/lib; do
    [ -d "$d" ] && dirs="${dirs:+$dirs:}$d"
done
[ -n "${LD_LIBRARY_PATH:-}" ] && dirs="${dirs:+$dirs:}$LD_LIBRARY_PATH"
[ -n "$dirs" ] && export LD_LIBRARY_PATH="$dirs"

missing=$(ldd "$bin" 2>&1 | awk '/not found/{print $1}' | sort -u)
if [ -n "$missing" ]; then
    echo "strata-vision: the image encoder cannot start: missing runtime libraries: $missing" >&2
    echo "strata-vision: install Intel oneAPI (the DPC++ compiler and oneMKL components), then either" >&2
    echo "strata-vision:   source <oneapi>/setvars.sh        # before starting the server, or" >&2
    echo "strata-vision:   export STRATA_ONEAPI_ROOT=<oneapi-install-root>" >&2
    exit 1
fi

exec "$bin" "$@"
