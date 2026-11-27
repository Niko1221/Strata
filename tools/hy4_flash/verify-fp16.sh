#!/usr/bin/env bash
# Compare the fixed Hy4-Flash Q4_K_M cases with llama.cpp v1.1.0.
set -euo pipefail
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'HELP'
Usage: verify-fp16.sh [MODEL.gguf [PACK_DIR]]
Defaults to the hy4-flash launch preset. Missing packs are built from the GGUF.
With one model, the default pack directory is pack/ beside the GGUF.
Requires Linux, Python with NumPy, a configured CUDA Strata build, and a
CUDA llama.cpp v1.1.0 build with compile_commands.json. The checked
profile uses a 37 GiB expert pool on one SM89 GPU and AVX-512 CPU experts.

GPU              Strata GPU (default: first idle card with 46 GiB free)
REFERENCE_GPUS   GPUs for generating golden logits (default: up to five idle cards)
LLAMA_DIR        llama.cpp checkout (default: sibling llama.cpp_v1.1.0)
LLAMA_BUILD      llama.cpp build directory (default: LLAMA_DIR/build)
STRATA_BUILD     Strata build directory (default: repository build/)
REFERENCE_DIR    Override the default cache/ beside the GGUF (about 583 MiB per model)

llama.cpp generates missing reference logits. Later runs reuse the cache
using shard file metadata and only need one GPU. Exit 0 requires all 1,263
positions to pass top-1, mean KL <= 1e-10 per case, and max |logit delta|
<= 1e-5. Exit 1 indicates a comparison failure.
HELP
    exit 0
fi
if (( $# > 2 )); then echo 'Use --help for usage.' >&2; exit 2; fi
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
llama=$(realpath "${LLAMA_DIR:-$root/../llama.cpp_v1.1.0}")
llama_build=$(realpath "${LLAMA_BUILD:-$llama/build}")
build=$(realpath "${STRATA_BUILD:-$root/build}")
python=${PYTHON_BIN:-python3}
if [[ $(git -C "$llama" rev-parse HEAD) != 171527522610996f5f0acf12334e185156ca8f44 ]]; then
    echo 'LLAMA_DIR must be checked out at v1.1.0 (171527522).' >&2; exit 2
fi
if [[ ! -f "$llama_build/compile_commands.json" ]] ||
   ! grep -Fxq "CMAKE_HOME_DIRECTORY:INTERNAL=$llama" "$llama_build/CMakeCache.txt"; then
    echo 'Configure LLAMA_BUILD from LLAMA_DIR with CUDA and CMAKE_EXPORT_COMPILE_COMMANDS=ON.' >&2; exit 2
fi
if ! grep -Fxq "CMAKE_HOME_DIRECTORY:INTERNAL=$root/engine" "$build/CMakeCache.txt"; then
    echo 'Configure STRATA_BUILD from the engine/ directory of this checkout before validating.' >&2; exit 2
fi
"$python" -c 'import numpy'
flock "$build/.hy4-parity-build.lock" cmake --build "$build" --target strata libstrata_kq.so
flock "$llama_build/.hy4-parity-build.lock" cmake --build "$llama_build" --target llama
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
if ! flock "$llama_build/.hy4-parity-build.lock" "$python" "$here/build_reference_ops.py" \
    "$llama_build" "$work/reference.so" >"$work/build-reference.log" 2>&1; then
    cat "$work/build-reference.log" >&2
    exit 2
fi
"$python" -u "$here/verify_fp16.py" "$here" "$build" "$llama" "$llama_build" "$work" "${1:-${MODEL:-}}" "${2:-${PACK:-}}"
