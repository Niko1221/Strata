#!/usr/bin/env bash
# Set up agentionai Gyro (Qwen3.8-Flash-Next rotor quants) on Strata: build, download, pack, draft head, config.
# Linux only. See docs/GYRO.md for what each step does.
#   tools/gyro_setup.sh --model S|M --backend cuda|hip [--cuda-arch 120] [--hip-arch gfx1201]
#                       [--models DIR] [--llama DIR] [--ctx 131072] [--port 8080]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=S BACKEND=cuda CUDA_ARCH=120 HIP_ARCH=gfx1201 MODELS=models LLAMA=../agention-llama.cpp CTX=131072 PORT=8080
while [ $# -gt 0 ]; do case "$1" in
  --model) MODEL=$2; shift 2;; --backend) BACKEND=$2; shift 2;; --cuda-arch) CUDA_ARCH=$2; shift 2;;
  --hip-arch) HIP_ARCH=$2; shift 2;; --models) MODELS=$2; shift 2;; --llama) LLAMA=$2; shift 2;;
  --ctx) CTX=$2; shift 2;; --port) PORT=$2; shift 2;; -h|--help) sed -n 2,5p "$0"; exit 0;;
  *) echo "unknown option $1"; exit 2;; esac; done
case "$MODEL" in S) F=Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf; P=gyro-s;; M) F=Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf; P=gyro-m;;
  *) echo "--model must be S or M"; exit 2;; esac
step() { printf '\n== %s\n' "$*"; }
for t in git python3; do command -v $t >/dev/null || { echo "missing: $t"; exit 1; }; done

step "python environment (also provides cmake >= 3.24 and ninja)"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt huggingface_hub hf_transfer
export PATH="$PWD/.venv/bin:$PATH"

step "agentionai llama.cpp fork (GGML with the trellis types) -> $LLAMA"
[ -d "$LLAMA/ggml" ] || git clone --depth 1 https://github.com/agentionai/llama.cpp "$LLAMA"
LLAMA=$(cd "$LLAMA" && pwd)

step "build Strata ($BACKEND)"
if [ "$BACKEND" = cuda ]; then
  NV=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p' || true)
  if [ "${CUDA_ARCH}" -ge 120 ] && [ "${NV:-0}" -lt 13 ]; then
    echo "warning: RTX 50 (sm_${CUDA_ARCH}) with CUDA ${NV:-?}: use CUDA 13 (12.8 crashes the prompt path, issues #220/#224)"; fi
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DSTRATA_GGML_DIR="$LLAMA"
elif [ "$BACKEND" = hip ]; then
  echo "note: AMD is experimental for Gyro (kernel parity validated on gfx1151, no end-to-end run yet)"
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON \
    -DCMAKE_HIP_ARCHITECTURES="$HIP_ARCH" -DSTRATA_GGML_DIR="$LLAMA" \
    -DCMAKE_PREFIX_PATH="${ROCM_PATH:-/opt/rocm}"
else echo "--backend must be cuda or hip"; exit 2; fi
cmake --build build -j

step "download $F -> $MODELS (keep it on a fast local SSD: the n-gram table is read per token)"
mkdir -p "$MODELS"
if [ -s "$MODELS/$F" ]; then echo "already there: $MODELS/$F (delete it to download again)"
else HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download agentionai/Qwen3.8-Flash-Next-Gyro-GGUF "$F" --local-dir "$MODELS"; fi
G=$(cd "$MODELS" && pwd)/$F

step "pack -> packs/$P"
STRATA_GGUF_PY="$LLAMA/gguf-py" .venv/bin/python tools/iq_pack.py --gguf "$G" --out "packs/$P" --compat-bf16
LC=$(build/native_expert_parity --load-check "packs/$P" "$G" 2>&1) || true; echo "$LC"
echo "$LC" | grep -q " 0 failures" || { echo "load check failed: see above"; exit 1; }

step "draft head (Strata's MTP runtime)"
if [ ! -f mtp/rt/draft_vocab.bin ]; then
  .venv/bin/python tools/mtp_fetch.py fetch --out mtp
  .venv/bin/python tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf
  .venv/bin/python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
  cp data/draft_vocab.bin mtp/rt/draft_vocab.bin
fi

step "config strata-$P.json"
cat > "strata-$P.json" <<JSON
{
  "exe": "build/strata",
  "args": [
    "--pack", "packs/$P",
    "--native", "$G",
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
    "--prefill", "512", "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", "mtp/rt", "--max-context", "$CTX", "--kv", "int8"
  ],
  "cwd": ".",
  "tokenizer": "packs/$P/tokenizer",
  "model_name": "$P-strata",
  "log": "strata-$P.log",
  "host": "127.0.0.1",
  "port": $PORT
}
JSON

step "done. Start the server with:"
echo "  STRATA_APR=1 .venv/bin/python -m serve.server --engine strata --config strata-$P.json --port $PORT"
echo "  web: http://127.0.0.1:$PORT   API: http://127.0.0.1:$PORT/v1"
