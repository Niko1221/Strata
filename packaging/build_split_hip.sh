#!/bin/bash
# strata-split-4 (the all-cards split engine, src/split, from sFNX, MIT, John Fink) for AMD RDNA4 (gfx1201: Radeon AI PRO R9700, RX 9070) on Linux, built next to the HIP
# engine package (packaging/build_hip_linux.sh, VARIANT=rx9070) so package_engine.py ships it with `--include 'strata-split-*'` beside `strata`. One executable per card
# count, named strata-split-<N>-gfx1201 (the launcher, strata_cli/launch/runner.py, looks for exactly that name on AMD): the AMD Linux package ships N = 2, 3 and 4. Which counts
# run which model is the launcher's rule (strata_control/split_engine.py: all the experts must fit in the cards' memory), not the binary's. RUNPATH $ORIGIN: it finds the ROCm libraries the engine package bundles (libamdhip64, hipblas, rocblas and its
# Tensile data for gfx1201) beside itself, the same set `strata` uses.
#
#   SRC=<checkout> B=<build dir> HP=<ROCm devel root> [LIBW=<ROCm libraries root>] GG=<llama.cpp checkout at the pinned commit> [CARDS="2 3 4"] [JOBS=12]  packaging/build_split_hip.sh
#
# After the build the script FAILS if the binary
#   - links librccl (RCCL is loaded at run time with dlopen when FNX_PF_RCCL=1 and is optional: the split must start without it),
#   - can call hipMemcpyPeer* (the engine's peer traffic is kernel stores only; src/split/src/hip/peer.h is the one file that touches peer access),
#   - or is missing the gfx1201 code object.
# Prints the binary paths (the renamed copies), one per line, last.
set -eu
SRC=${SRC:-$(cd "$(dirname "$0")/.." && pwd)}
B=${B:?B=<build dir>}
HP=${HP:?HP=<ROCm devel root>}
LIBW=${LIBW:-$HP}
GG=${GG:?GG=<llama.cpp checkout at the pinned commit>}
CARDS=${CARDS:-2 3 4}
JOBS=${JOBS:-12}
ARCH=gfx1201
export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr ROCM_PATH=$HP HIP_PATH=$HP PATH=$HP/bin:$PATH
SYSDEPS=""; [ -d "$HP/lib/rocm_sysdeps" ] && SYSDEPS=";$HP/lib/rocm_sysdeps"
BC=$HP/lib/llvm/amdgcn/bitcode; [ -d "$BC" ] || BC=$HP/amdgcn/bitcode
rm -f "$B/CMakeCache.txt"
cmake -G Ninja -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_SPLIT=ON \
  "-DSTRATA_SPLIT_CARDS=${CARDS// /;}" "-DCMAKE_HIP_ARCHITECTURES=$ARCH" -DSTRATA_PREFILL_MMQ=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_GGML_DIR="$GG" \
  -DCMAKE_HIP_COMPILER="$HP/llvm/bin/clang++" -DCMAKE_HIP_COMPILER_ROCM_ROOT="$HP" "-DCMAKE_PREFIX_PATH=$HP;$LIBW$SYSDEPS" \
  "-DCMAKE_HIP_FLAGS=--rocm-path=$HP --rocm-device-lib-path=$BC" -DSTRATA_PORTABLE=ON \
  "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,'\$ORIGIN' -Wl,--enable-new-dtags"
out=""
for n in $CARDS; do
  ninja -j "$JOBS" -C "$B" "strata-split-$n"
  exe=$(find "$B" -name "strata-split-$n" -type f | head -1)
  [ -x "$exe" ] || { echo "build_split_hip.sh: no strata-split-$n" >&2; exit 1; }
  if readelf -d "$exe" | grep -qi 'NEEDED.*rccl'; then
    echo "build_split_hip.sh: $exe links librccl (it must be dlopen'ed only):" >&2
    readelf -d "$exe" | grep -i needed >&2
    exit 1
  fi
  if nm -D --undefined-only "$exe" | grep -qiE 'hipMemcpy(3D)?Peer'; then
    echo "build_split_hip.sh: $exe can call hipMemcpyPeer*:" >&2
    nm -D --undefined-only "$exe" | grep -iE 'hipMemcpy(3D)?Peer' >&2
    exit 1
  fi
  if ! grep -aq "$ARCH" "$exe"; then
    echo "build_split_hip.sh: $exe has no $ARCH code object" >&2
    exit 1
  fi
  named="$(dirname "$exe")/strata-split-$n-$ARCH"
  cp -f "$exe" "$named"
  out="$out$named"$'\n'
done
printf '%s' "$out"
