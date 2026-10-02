#!/usr/bin/env bash
# Build Strata for this machine's AMD GPU in Docker, and produce the runnable image.
#
#   ./build.sh                 # strata-hip-builder:<arch> -> compile -> strata-hip:<arch>
#   ./build.sh --tests         # ... and run the HIP test set on the GPU afterwards
#   ./build.sh --arch gfx1100  # build for the other supported RDNA3 card
#   ./build.sh --runtime-only  # repackage ./build-hip/strata without recompiling
#
# What it does:
#   1. builds strata-hip-builder:<arch>   (ROCm toolchain, ~7.4 GB base)
#   2. runs it with the repo bind-mounted and compiles the engine into ./build-hip (ccache volume)
#   3. builds strata-hip:<arch>           (runtime: engine + serve/ + tools/ + model bootstrap)
#   4. gates the result: every shared library resolves, serve/ imports
# Then start it with ./run.sh.  Plan and rationale: docs/DOCKER_GFX1101_PLAN.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------- defaults
ARCH="gfx1101"                                    # this machine: RX 7700 XT
BUILDER_BASE="rocm/dev-ubuntu-24.04:7.2.1-complete"
RUNTIME_BASE="rocm/dev-ubuntu-24.04:7.2.1-complete"
JOBS="$(nproc)"
SKIP_COMPILE=0 RUNTIME_ONLY=0 BUILDER_ONLY=0 WITH_TESTS=0 DRY=0 NO_CACHE=""
CCACHE_VOL="${STRATA_CCACHE_VOLUME:-}"            # default: strata-ccache-<arch>
VERSION="$(sed -n 's/^[[:space:]]*project(strata VERSION \([0-9.]*\).*/\1/p' CMakeLists.txt 2>/dev/null | head -1)"
VERSION="${VERSION:-dev}"                          # CMakeLists.txt:11 is the engine's source of truth

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
  cat <<'USAGE'

Options:
  -a, --arch ARCH        gfx1101 (RX 7700 XT, default) or gfx1100 (RX 7900 XT/XTX)
      --builder-base IMG Base image for the builder    (default: rocm/dev-ubuntu-24.04:7.2.1-complete)
      --runtime-base IMG Base image for the runtime    (default: same; :7.2.1 alone has no math libs)
  -j, --jobs N           compile jobs (default: nproc)
      --skip-compile     keep ./build-hip/strata as it is
      --runtime-only     same as --skip-compile
      --builder-only     stop after the builder image
      --tests            after building, run the HIP CTest set on this GPU
      --no-cache         docker build --no-cache
  -n, --dry-run          print the commands instead of running them
  -h, --help             this text

Environment: STRATA_CCACHE_VOLUME, DOCKER, GIT_SHA.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    -a|--arch)        ARCH="${2:?--arch needs a value}"; shift 2 ;;
    --builder-base)   BUILDER_BASE="${2:?}"; shift 2 ;;
    --runtime-base)   RUNTIME_BASE="${2:?}"; shift 2 ;;
    -j|--jobs)        JOBS="${2:?}"; shift 2 ;;
    --skip-compile|--runtime-only) SKIP_COMPILE=1; shift ;;
    --builder-only)   BUILDER_ONLY=1; shift ;;
    --tests)          WITH_TESTS=1; shift ;;
    --no-cache)       NO_CACHE="--no-cache"; shift ;;
    -n|--dry-run)     DRY=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    *)                usage >&2; die "unknown option '$1'" ;;
  esac
done

BUILDER_IMAGE="strata-hip-builder:${ARCH}"
RUNTIME_IMAGE="strata-hip:${ARCH}"
GIT_SHA="${GIT_SHA:-$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)}"
RUNTIME_IMAGE_SHA="strata-hip:${ARCH}-${GIT_SHA}"
CCACHE_VOL="${CCACHE_VOL:-strata-ccache-${ARCH}}"
ENGINE="$ROOT/build-hip/strata"

xrun() {   # run, or print in dry-run mode
  if [ "$DRY" = 1 ]; then printf '  +'; printf ' %q' "$@"; printf '\n'; else "$@"; fi
}

# ---------------------------------------------------------------- preflight
[ -f "$ROOT/CMakeLists.txt" ] && [ -f "$ROOT/docker/Dockerfile.hip" ] \
  || die "run this from the Strata repository root (CMakeLists.txt and docker/ must be here)"
[ -f "$ROOT/.dockerignore" ] || die ".dockerignore is missing; the build context would upload models/ and build-hip/"
DOCKER="${DOCKER:-docker}"
command -v "$DOCKER" >/dev/null || die "docker not found on PATH"
xrun "$DOCKER" info >/dev/null 2>&1 || die "the docker daemon is not reachable (is it running, and are you in the docker group?)"

case "$ARCH" in
  gfx1101|gfx1100) ;;
  *) die "--arch must be gfx1101 (RX 7700 XT / 7800 XT) or gfx1100 (RX 7900 XT / XTX).
       Your card's real arch is $(python3 "$ROOT/docker/hipinfo.py" --arch 2>/dev/null || echo 'unknown')." ;;
esac

# cmake/hip_backend.cmake aborts the configure for an arch it does not know. Finding that out after a
# 7.4 GB pull costs patience; finding it out now costs nothing.
if ! grep -q "$ARCH" "$ROOT/cmake/hip_backend.cmake"; then
  die "cmake/hip_backend.cmake has never heard of $ARCH and would fail the configure with
       'Strata HIP currently supports only gfx1100 wave32'.
       Apply the gfx1101 patch set first: docs/DOCKER_GFX1101_PLAN.md Part C items 1-4
       (cmake/hip_backend.cmake, src/core/device.cu, include/strata/hip_compat/intrinsics.hpp)."
fi
if [ "$ARCH" = gfx1101 ] && ! grep -q '__gfx1101__' "$ROOT/include/strata/hip_compat/intrinsics.hpp"; then
  warn "intrinsics.hpp has no __gfx1101__ branch: the RDNA3 signed-dot would be compiled out and
       __dp4a would silently fall back to a scalar loop (correct, much slower). Gate P3 in
       docs/DOCKER_GFX1101_PLAN.md is exactly about catching that."
fi

# Where images and layers land: a ROCm build needs tens of GB of headroom.
ROOT_DIR_FREE_GB="$("$DOCKER" info --format '{{.DockerRootDir}}' 2>/dev/null | xargs -r df -BG --output=avail 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)"
if [ "${ROOT_DIR_FREE_GB:-0}" != 0 ] && [ "${ROOT_DIR_FREE_GB:-0}" -lt 40 ]; then
  warn "docker's data-root has only ${ROOT_DIR_FREE_GB} GB free; two ROCm images plus a build tree"
  warn "want ~40 GB.  Move it:  echo '{\"data-root\":\"/mnt/storage/docker\"}' | sudo tee /etc/docker/daemon.json"
  warn "then  sudo systemctl restart docker   (plan Phase P0)"
fi
if [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
  warn "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION is in your environment. It does not affect"
  warn "this build (containers get a clean env), but it must NOT be set when running: on gfx1101 it"
  warn "makes the code object fail to load.  ./run.sh refuses it."
fi

log "building Strata $VERSION for $ARCH ($JOBS jobs) - builder $BUILDER_IMAGE, runtime $RUNTIME_IMAGE"

# ---------------------------------------------------------------- 1. the builder image
xrun "$DOCKER" build $NO_CACHE -f docker/Dockerfile.hip-builder \
     --build-arg ROCM_BASE="$BUILDER_BASE" --build-arg STRATA_HIP_ARCH="$ARCH" \
     --build-arg STRATA_VERSION="$VERSION" \
     -t "$BUILDER_IMAGE" .
log "builder image: $BUILDER_IMAGE"
[ "$BUILDER_ONLY" = 1 ] && { log "stopping after the builder image (--builder-only)"; exit 0; }

# ---------------------------------------------------------------- 2. compile inside it
if [ "$SKIP_COMPILE" = 1 ]; then
  log "--skip-compile: using the existing $ENGINE"
else
  xrun "$DOCKER" volume create "$CCACHE_VOL" >/dev/null
  # No --device: compiling needs no GPU.  The build runs as root (ROCm images have no usable non-root
  # user), so the tree is chowned back to whoever invoked this.
  xrun "$DOCKER" run --rm --name "strata-build-${ARCH}" \
       -v "$ROOT":/src -v "$CCACHE_VOL":/ccache \
       -e JOBS="$JOBS" -e CMAKE_TESTS=1 \
       -e CHOWN_UID="$(id -u)" -e CHOWN_GID="$(id -g)" \
       --ulimit memlock=-1:-1 \
       --entrypoint bash "$BUILDER_IMAGE" -lc '
         set -e
         /usr/local/bin/build-engine.sh '"$ARCH"'
         if [ "${CHOWN_UID:-0}" != "0" ]; then chown -R "${CHOWN_UID}:${CHOWN_GID}" /src/build-hip; fi'
fi
if [ "$DRY" != 1 ]; then
  [ -x "$ENGINE" ] || die "the compile produced no $ENGINE - see the compiler output above"
  log "engine: $(du -h "$ENGINE" | cut -f1), sha256 $(sha256sum "$ENGINE" | cut -c1-16)..."
fi

# ---------------------------------------------------------------- 3. the runtime image
# gguf-py from the pinned llama.cpp checkout that was just built against (tools/_paths.py:16 wants
# <dir>/gguf/).  Fallback: an empty staging dir with a note, so the COPY always has something to
# copy and the container degrades to --spec 0 with a clear message instead of failing to build.
GGUFPY_SRC="$(find "$ROOT/build-hip/_deps" -maxdepth 3 -type d -name gguf-py 2>/dev/null | head -1 || true)"
GGUFPY_STAGE="$ROOT/build-hip/.ggufpy-stage"
if [ -n "$GGUFPY_SRC" ]; then
  note "gguf-py: $GGUFPY_SRC"
  xrun rm -rf "$GGUFPY_STAGE"; xrun mkdir -p "$GGUFPY_STAGE"
  xrun cp -a "$GGUFPY_SRC/." "$GGUFPY_STAGE/"
else
  warn "no gguf-py under build-hip/_deps: the MTP draft layer cannot be prepared at runtime"
  warn "(the container starts with --spec 0). Re-run the compile step to fetch the pinned llama.cpp."
  xrun rm -rf "$GGUFPY_STAGE"; xrun mkdir -p "$GGUFPY_STAGE"
  printf 'no gguf-py staged; build with the pinned llama.cpp to enable the MTP draft layer\n' | xrun tee "$GGUFPY_STAGE/MISSING.txt"
fi

xrun "$DOCKER" build $NO_CACHE -f docker/Dockerfile.hip \
     --build-arg RUNTIME_BASE="$RUNTIME_BASE" --build-arg STRATA_VERSION="$VERSION" \
     --build-arg STRATA_HIP_ARCH="$ARCH" \
     --build-context "engine=$ROOT/build-hip" \
     --build-context "ggufpy=$GGUFPY_STAGE" \
     -t "$RUNTIME_IMAGE" -t "$RUNTIME_IMAGE_SHA" -t "strata-hip:${ARCH}-latest" .
log "runtime image: $RUNTIME_IMAGE, $RUNTIME_IMAGE_SHA"

# ---------------------------------------------------------------- 4. gates over the result
if [ "$DRY" != 1 ]; then
  log "gate: shared libraries of /usr/local/bin/strata inside the runtime image"
  MISSING_LIBS="$("$DOCKER" run --rm --entrypoint bash "$RUNTIME_IMAGE" -lc \
                    'ldd /usr/local/bin/strata | sed -n "s/^\s*\(.*\)\s=>\snot found.*/\1/p"' || true)"
  if [ -n "$MISSING_LIBS" ]; then
    printf '%s\n' "$MISSING_LIBS" | sed 's/^/    missing: /' >&2
    die "the runtime base has no ROCm math libraries. Rebuild the runtime with the complete tag:
         ./build.sh --runtime-base "$BUILDER_BASE" --runtime-only"
  fi
  log "gate: serve/ imports and the model helpers load inside the image"
  "$DOCKER" run --rm --workdir /opt/strata --entrypoint /opt/venv/bin/python "$RUNTIME_IMAGE" -c \
    'import sys; sys.path[:0] = ["/opt/strata", "/opt/strata/docker"]
import serve.server, jinja2, numpy, regex, psutil, hfmodel, hipinfo
print("   imports ok")'
fi

# ---------------------------------------------------------------- 5. optional tests on the GPU
if [ "$WITH_TESTS" = 1 ]; then
  RENDER="$(python3 "$ROOT/docker/hipinfo.py" --render-node)"
  TEST_ARCH="$(python3 "$ROOT/docker/hipinfo.py" --arch)"
  [ "$TEST_ARCH" = "$ARCH" ] || die "this machine has $TEST_ARCH but the image was built for $ARCH"
  log "gate: HIP tests on $TEST_ARCH ($RENDER)"
  # memlock: platform_memory_test asks for 256 MiB of locked memory and fails on the host shell's
  # 8 MiB default.  ple_parity needs an external fixture and stays excluded by name - never counted
  # as a pass (docs/AMD_HIP.md:145-152).
  "$DOCKER" run --rm --name "strata-test-${ARCH}" \
    --device /dev/kfd --device "$RENDER" --group-add video --group-add render \
    --ulimit memlock=-1:-1 -e HIP_VISIBLE_DEVICES=0 -e HSA_ENABLE_SDMA=1 \
    -v "$ROOT":/src -v "$CCACHE_VOL":/ccache \
    --entrypoint bash "$BUILDER_IMAGE" -lc '
      cd /src/build-hip && ctest --output-on-failure --timeout 60 -E "^(ple_parity)$"'
fi

# ---------------------------------------------------------------- summary
log "done"
"$DOCKER" image ls --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}' \
  | grep -E "^(strata-hip|REPOSITORY)" | sed 's/^/  /' || true
cat <<NEXT

Next:
  ./run.sh                      start the server (models come from your Hugging Face cache)
  ./run.sh --model IQ2_XS       a smaller quant
  docker logs -f strata-gfx1101 watch the engine
NEXT
