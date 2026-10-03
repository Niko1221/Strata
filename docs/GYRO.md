# Gyro (agentionai rotor quants) on Strata

Validated on Linux with an RTX 5090 (32 GB), 16 CPU cores and 30 GB RAM on 2026-10-02 (CUDA 13.0, sm_120).

The targets are the published files of `agentionai/Qwen3.8-Flash-Next-Gyro-GGUF`, used unchanged:

| File | Experts | GPU memory for the experts | Status |
|---|---|---:|---|
| `Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf` (54.5 GB) | TQK6 gate/up, TQK7 down, block-128 Hadamard | 24.0 GiB | validated end to end |
| `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf` (92.0 GB) | TQ2_T, block-128 Hadamard | 29.9 GiB | packs and passes `--load-check`; not run end to end |

Both files carry their own experts in the trellis types TQ2_T/TQK6/TQK7 and record the activation rotation in
`prism.hadamard.*` metadata. Strata reads both natively (set `STRATA_APR=1`). About half of each file is the
n-gram (PLE) table, which stays on disk and is read row by row; it is not loaded into RAM or VRAM.

The draft head is Strata's own MTP runtime, prepared from the original model's MTP weights as for the other
Qwen3.8-Flash-Next models. It is not the GGUF draft file from the Gyro repository.

## Quick setup

Gyro support lives on the `feat/apr-types` branch of the agentionai fork:

```sh
git clone -b feat/apr-types https://github.com/agentionai/Strata && cd Strata
```

`tools/gyro_setup.sh` does every step below on Linux:

```sh
tools/gyro_setup.sh --model S --backend cuda --cuda-arch 120       # RTX 50 series
tools/gyro_setup.sh --model S --backend cuda --cuda-arch 89        # RTX 40 series
tools/gyro_setup.sh --model S --backend hip  --hip-arch gfx1201    # Radeon RX 9070 / AI PRO R9700 (experimental)
```

It prints the command that starts the server. The steps, by hand:

## 1. Build

Strata's GGML dependency must be the agentionai llama.cpp fork, which has the trellis types:

```sh
git clone --depth 1 https://github.com/agentionai/llama.cpp ../agention-llama.cpp
# NVIDIA. RTX 50 series: use CUDA 13 (CUDA 12.8 crashes the prompt path on sm_120: issues #220, #224)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_GGML_DIR=$PWD/../agention-llama.cpp
# AMD (experimental: kernel parity validated on gfx1151, not run end to end)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DSTRATA_GGML_DIR=$PWD/../agention-llama.cpp
cmake --build build -j
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## 2. Download and pack

```sh
.venv/bin/pip install huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download agentionai/Qwen3.8-Flash-Next-Gyro-GGUF \
  Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf --local-dir models
STRATA_GGUF_PY=../agention-llama.cpp/gguf-py .venv/bin/python tools/iq_pack.py \
  --gguf models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf --out packs/gyro-s --compat-bf16
build/native_expert_parity --load-check packs/gyro-s models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf
```

The pack is 1.5 GB (dense weights, tokenizer, the Hadamard and expert tables); the experts are read from the
GGUF. Keep the GGUF on a fast local SSD: the n-gram table is read from it for every token.
`--load-check` must end with `0 failures`.

## 3. Draft head

```sh
.venv/bin/python tools/mtp_fetch.py fetch --out mtp
.venv/bin/python tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
cp data/draft_vocab.bin mtp/rt/draft_vocab.bin
```

## 4. Serve

Save as `strata-gyro-s.json` at the repository root:

```json
{
  "exe": "build/strata",
  "args": [
    "--pack", "packs/gyro-s",
    "--native", "models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf",
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
    "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", "mtp/rt", "--max-context", "131072", "--kv", "int8"
  ],
  "cwd": ".",
  "tokenizer": "packs/gyro-s/tokenizer",
  "model_name": "gyro-s-strata",
  "log": "strata-gyro-s.log",
  "host": "127.0.0.1",
  "port": 8080
}
```

```sh
STRATA_APR=1 .venv/bin/python -m serve.server --engine strata --config strata-gyro-s.json --port 8080
```

The web interface is at `http://127.0.0.1:8080`; API clients use `http://127.0.0.1:8080/v1` (OpenAI) or
`/v1/messages` (Anthropic). `--expert-cache auto` sizes the expert cache to the free VRAM: on a 32 GB card at
128k context it holds 23,464 of 24,576 experts (22.9 GiB). On smaller cards it holds fewer and streams the rest
from RAM; nothing else changes.

## Measured (RTX 5090, CUDA 13.0, greedy, same prompt tokens as llama.cpp)

Decode tokens/s, single prompts of 34-177 tokens, 384 new tokens, `--max-context 16384`:

| Expert cache | prose | JSON | code |
|---|---:|---:|---:|
| 24,576 slots (all experts) | 116.4 | 194.3 | 173.5 |
| 22,000 slots | 111.4 | 175.7 | 166.1 |
| 9,000 slots (about a 16 GB card's room) | 81.4 | 74.0 | 96.8 |
| 3,000 slots (about an 8 GB card's room) | 48.0 | 52.4 | 57.9 |
| llama.cpp (agentionai main), best of no draft / cost-aware MTP | 106.0 | 172.9 | 142.2 |

The smaller caches were emulated on the 5090 (fast GPU, PCIe 5.0 x16 at ~51 GB/s). A real 8 or 16 GB card is a
slower GPU, often on a slower link: read those rows as "it stays usable", not as a prediction for a specific card.
The decoded prose output was read and is fluent and correct; token-level parity against llama.cpp has not been
checked. A real chat at 128k context worked through the web interface.

## Known limits

- **Not measured yet:** long-prompt prefill, token parity against llama.cpp, Gyro-M end to end.
- **Benchmark mode only:** `strata --tokens-file ... --max-new N` decodes N tokens past the end of the answer;
  compare speeds only up to the answer's end.
- **AMD:** the trellis kernels pass parity on gfx1151 (ROCm 7.2.1); no end-to-end run on AMD yet.
- **Gyro-M:** 29.9 GiB of experts do not all fit a 32 GB card next to the KV cache; `auto` streams the rest.
