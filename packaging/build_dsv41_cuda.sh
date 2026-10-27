#!/bin/bash
# strata-dsv41 (the DeepSeek-V4.1 engine) for NVIDIA on Linux, built next to the release engine so package_engine.py can ship it
# beside `strata` (setup.py dsv41_exe() looks for strata-dsv41 in the engine folder).  Same toolchain and RUNPATH rule as the
# engine build: the result finds cudart/cublas beside itself ($ORIGIN), never the toolkit it was built with.
#
#   VARIANT=cuda13| cuda12  SRC=<checkout> B=<build dir> [CUDA=<toolkit root>] GG=<llama.cpp checkout> [ARCH=...] [JOBS=6]  packaging/build_dsv41_cuda.sh
#
# VARIANT=cuda13 (default) is the 0.2.7 release shape: SASS for sm_86 (RTX 3090 / 3060), sm_89 (RTX 40) and sm_120 (RTX Blackwell) + PTX sm_120,
# and needs CUDA 13 (the sm_120 code).  VARIANT=cuda12 is the package for cards CUDA 13 has no code for: sm_70 (V100 / Titan V), sm_75 (RTX 20),
# sm_80 (A100), sm_86 and sm_89; it needs a CUDA 12.x toolkit and builds with -DSTRATA_EXPERIMENTAL_SM60=ON, the same flag the cuda12 Qwen engine
# uses (packaging/build_cuda_linux.sh).  Pack the result as the feed's dsv41 member with '--variant cuda12' in package_engine.py / feed_tool.py.
# ARCH overrides the per-variant default ("70-real;86-real;89-real" for cuda12, "86-real;89-real;120" for cuda13); a cuda12 build must not name 120.
# Prints the binary path on the last line; collect its libraries with packaging/collect_libs.py --binary <it> --search $CUDA/lib64.
set -eu
SRC=${SRC:-$(cd "$(dirname "$0")/.." && pwd)}
B=${B:?B=<build dir>}
GG=${GG:?GG=<llama.cpp checkout at the pinned commit>}
variant=${VARIANT:-cuda13}
case "$variant" in
  cuda13) CUDA=${CUDA:-/usr/local/cuda-13.0}; ARCH=${ARCH:-86-real;89-real;120}; sm60=OFF ;;
  cuda12) CUDA=${CUDA:-/usr/local/cuda-12.9}; ARCH=${ARCH:-70-real;86-real;89-real}; sm60=ON ;;
  *) echo "build_dsv41_cuda.sh: VARIANT must be cuda13 or cuda12" >&2; exit 2 ;;
esac
if [ "$variant" = cuda12 ]; then
  case "$ARCH" in *120*) echo "build_dsv41_cuda.sh: a cuda12 build cannot name sm_120 (it needs CUDA 13)" >&2; exit 2 ;; esac
fi
JOBS=${JOBS:-6}
cudafull=$("$CUDA/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
case "$variant:$cudafull" in
  cuda12:12.*|cuda13:13.*) ;;
  *) echo "build_dsv41_cuda.sh: $variant needs the matching CUDA toolkit; nvcc here is ${cudafull:-missing}" >&2; exit 2 ;;
esac
export PATH=$CUDA/bin:$PATH
# the build id: the commit, 9 characters.  A git checkout gives it to CMake itself; a tarball has no .git, so COMMIT=<hex> is required there and goes to BUILD_ID.
# (The id is baked in when cmake configures: a stale build dir is configured again, and a stale binary is deleted first.)
if [ ! -e "$SRC/.git" ] && [ ! -f "$SRC/BUILD_ID" ]; then
  [ -n "${COMMIT:-}" ] || { echo "build_dsv41_cuda.sh: $SRC has no .git and no BUILD_ID: set COMMIT=<hex> so the engine can print its commit" >&2; exit 2; }
  printf '%s\n' "${COMMIT:0:9}" > "$SRC/BUILD_ID"
fi
rm -f "$B/CMakeCache.txt" "$B/strata-dsv41"
cmake -G Ninja -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DSV41=ON \
  -DSTRATA_EXPERIMENTAL_SM60=$sm60 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON "-DCMAKE_CUDA_ARCHITECTURES=$ARCH" -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA" -DSTRATA_GGML_DIR="$GG" -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_SKIP_BUILD_RPATH=OFF \
  '-DCMAKE_INSTALL_RPATH=$ORIGIN'
ninja -j "$JOBS" -C "$B" strata-dsv41
[ -x "$B/strata-dsv41" ] || { echo "no $B/strata-dsv41" >&2; exit 1; }
if [ -n "${COMMIT:-}" ]; then
  grep -aq "${COMMIT:0:9}" "$B/strata-dsv41" || { echo "build_dsv41_cuda.sh: strata-dsv41 does not carry the build id ${COMMIT:0:9}" >&2; exit 1; }
  echo "build id ${COMMIT:0:9} is in $B/strata-dsv41" >&2
fi
echo "$B/strata-dsv41"
