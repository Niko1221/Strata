#!/usr/bin/env bash
# remote-build.sh - build strata-vision on the remote vision host.
#
# Run on the vision host, usually over ssh from the machine that runs the server.  Everything it
# needs is already in this directory:
#   tools/vision/            the source (copied from the same Strata checkout as the engine)
#   third_party/llama.cpp/   the SAME pinned revision as the engine's ggml
#   mmproj-*.gguf            the vision encoder + projector
#   models/text-vocab-stub.gguf   metadata+vocabulary-only stub of the text shard
#
# The llama.cpp revision is not a variable to manage: it is copied from the Strata checkout, so
# it always matches the engine that will talk to this process.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# CUDA: nvcc is not on PATH by default on this machine, and CUDA_HOME is unset.
pick_cuda() {
  for v in "${CUDA_VERSION:-}" 13.3 13.1 13.0 12.9 12.8; do
    [ -n "$v" ] || continue
    if [ -x "/usr/local/cuda-$v/bin/nvcc" ]; then echo "/usr/local/cuda-$v"; return 0; fi
  done
  if command -v nvcc >/dev/null 2>&1; then dirname "$(dirname "$(command -v nvcc)")"; return 0; fi
  return 1
}
CUDA_DIR="$(pick_cuda)" || { echo "ERROR: no nvcc found under /usr/local/cuda-*"; exit 1; }
export CUDA_HOME="$CUDA_DIR"
export CUDACXX="$CUDA_DIR/bin/nvcc"
export PATH="$CUDA_DIR/bin:$PATH"
echo "== CUDA: $CUDA_DIR  ($("$CUDACXX" --version | tail -1))"

ARCH="${STRATA_VISION_ARCH:-86}"
echo "== CUDA arch: $ARCH    jobs: $(nproc)"

# The text model only has to exist and hold metadata+vocabulary; see make_vocab_stub.py.
STUB="$(ls models/text-vocab-stub.gguf 2>/dev/null || true)"
MMPROJ="$(ls mmproj-*.gguf 2>/dev/null | head -1 || true)"
[ -n "$STUB" ]   || { echo "ERROR: models/text-vocab-stub.gguf is missing"; exit 1; }
[ -n "$MMPROJ" ] || { echo "ERROR: mmproj-*.gguf is missing"; exit 1; }

cmake -S tools/vision -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_DIR="$HERE/third_party/llama.cpp" \
      -DSTRATA_VISION_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$ARCH"

cmake --build build --target strata-vision -j"$(nproc)"

# llama.cpp's CMake puts executables in build/bin (the Windows build in the Strata checkout
# uses build/Release); accept either.
BIN=""
for c in "$HERE/build/bin/strata-vision" "$HERE/build/strata-vision" "$HERE/build/Release/strata-vision"; do
  if [ -x "$c" ]; then BIN="$c"; break; fi
done
[ -n "$BIN" ] || { echo "ERROR: strata-vision was not produced under $HERE/build"; ls -R "$HERE/build" | head -20; exit 1; }
echo "== built: $BIN"

# Smoke test: start it the way the server does and require the READY line.  This is the same
# handshake the server waits for, so a pass here means the wrapper can succeed.
echo "== smoke test (expect a READY line within 120 s)"
set +e
out="$(printf 'QUIT\n' | timeout 120 "$BIN" --mmproj "$MMPROJ" --model "$STUB" --gpu --max-tokens 1024 2>/tmp/strata-vision-smoke.err | head -1)"
rc=$?
set -e
if [ $rc -ne 0 ] && [ -z "$out" ]; then
  echo "ERROR: the smoke test failed (rc=$rc); last stderr lines:"
  tail -5 /tmp/strata-vision-smoke.err
  exit 1
fi
echo "== smoke test says: $out"
case "$out" in
  READY*) echo "== OK"; ;;
  *) echo "ERROR: expected READY, got: $out"; tail -5 /tmp/strata-vision-smoke.err; exit 1 ;;
esac
