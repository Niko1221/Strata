#!/usr/bin/env bash
# The CPU image encoder for the SYCL port: strata-vision (tools/vision, no CUDA) built inside strata-sycl-dev, like
# sycl/tools/build.sh builds the engine.  sycl/tools/build-vision.sh     (docs/INTEL.md, "Images")
# Output in $BUILD_DIR (default <repo>/build-vision): bin/strata-vision, lib/libgomp.so.1 (the image's OpenMP runtime;
# the host has no libgomp) and strata-vision.sh, the launcher setup_intel.py writes as the config's vision "exe".
# It runs on the host (the engine runs in the container, the encoder does not), so it must build against the image's
# libc and be started with its own libgomp.  Needs llama.cpp's sources in third_party/llama.cpp (setup.py fetches them).
set -uo pipefail
# the host runs the encoder, so its glibc must be at least as new as this one (docs/INTEL.md, "Images")
echo "image glibc: $(ldd --version | head -1)"
repo=${REPO:-$(git rev-parse --show-toplevel)}
b=${BUILD_DIR:-$repo/build-vision}
llama=$repo/third_party/llama.cpp
st=0
mkdir -p "$b"
if [ ! -f "$llama/ggml/CMakeLists.txt" ] || [ ! -d "$llama/gguf-py" ]; then
    echo "llama.cpp is not in $llama (the encoder is built against it)."
    echo "Run ./setup.sh --backend sycl once on the host: its engine step fetches the pinned llama.cpp (setup.py's"
    echo "get_llama_cpp()) into third_party/llama.cpp. Then run this again."
    echo "BUILD EXIT 1"
    exit 1
fi
# gcc/g++ (not icx): the encoder is plain C++ and the host runs it, so it is built like the host's own binaries
cmake -S "$repo/tools/vision" -B "$b" -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR="$llama" -DSTRATA_VISION_CUDA=OFF -DSTRATA_PORTABLE=OFF \
    2>&1 | tee "$b/configure.log" | tail -15
st=${PIPESTATUS[0]}
if [ $st -eq 0 ]; then
    cmake --build "$b" --target strata-vision -j "${JOBS:-$(nproc)}" 2>&1 | tee "$b/build.log" \
        | grep -E '^FAILED|error:|^ninja: build stopped|Linking' | tail -40
    st=${PIPESTATUS[0]}
fi
if [ $st -eq 0 ]; then
    omp=$(g++ -print-file-name=libgomp.so.1)
    if [ ! -f "$omp" ]; then
        echo "libgomp.so.1 not found (g++ -print-file-name): the image has no OpenMP runtime"
        st=1
    else
        mkdir -p "$b/lib"
        cp -L "$omp" "$b/lib/libgomp.so.1"                # -L: the image's file is a symlink to the versioned library
    fi
fi
if [ $st -eq 0 ]; then
    cat > "$b/strata-vision.sh" <<'EOF'
#!/bin/sh
# strata-vision built in the strata-sycl-dev image: its OpenMP runtime (libgomp) is not installed on this host
here=$(dirname "$(readlink -f "$0")")
LD_LIBRARY_PATH="$here/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" exec "$here/bin/strata-vision" "$@"
EOF
    chmod +x "$b/strata-vision.sh"
    [ -x "$b/bin/strata-vision" ] || st=1
fi
echo "BUILD EXIT $st"
grep -c 'error:' "$b/build.log" 2>/dev/null | sed 's/^/errors: /'
grep -E '^FAILED' "$b/build.log" 2>/dev/null | sed 's#.*/##' | sort | uniq | head -20
exit $st
