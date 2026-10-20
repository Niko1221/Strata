# Gyro (agentionai rotor quants) on Strata

Gyro-S and Gyro-M are Qwen3.8-Flash-Next quantizations by agentionai
([agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)): the routed
experts are stored as trellis-coded blocks (ggml types TQ2_T, TQK6, TQK7) in a Hadamard-rotated basis. Strata reads the
published files unchanged. **Setup lists them as a model family of their own** ("Qwen3.8-Flash-Next (Gyro)"); neither
is anybody's default.

| | Gyro-S | Gyro-M |
| --- | --- | --- |
| File | `Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf` | `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf` |
| **Download** (the n-gram table is inside the file) | **58.5 GB** | **92.0 GB** |
| **Memory it needs** (VRAM + RAM together) | **27.6 GB** | **35 GB** |
| Experts (trellis) | TQK6 gate/up, TQK7 down | TQ2_T |
| In plain words | the smallest option, for PCs with little memory; lower quality than the 2-3-bit sizes | about IQ3_XXS quality with about 20% less GPU memory; decodes about 20% slower than Q2_0 on 12 GB cards |

About half of each file is the n-gram (PLE) table: it stays on the SSD and is read row by row for every token, so keep
the file on a fast local drive. The memory figure is what the weights take once loaded.

Not defaults, by measurement (our evaluation, 2026-10): against Q2_0 at the same GPU bytes Gyro-M halves the KL
divergence (0.159 against 0.312 on 128 chunks) and is level with IQ3_XXS (0.157), but decodes 17-21% slower on a 12 GB
card (RTX 3060) and an R9700, and its reasoning set scored lower than Q2_0's (49/60 against 57/60). Gyro-S is 13% worse
in KL than IQ2_XS (0.268 against 0.236) with 24% fewer bytes. So: pick Gyro-S where nothing bigger fits, Gyro-M when
the 20% memory saving over IQ3_XXS matters. Setup and the MCP server mention them only where the other sizes do not
fit the PC; they warn, never refuse. Gyro-L (agentionai's 50 GiB file, 2026-10-12) is not supported: it uses another
trellis family (types 147-149) that the engine does not have.

## Install

```sh
./setup.sh --setup --family gyro --model Gyro-S      # or Gyro-M;  Windows: START-HERE.bat --setup ...
```

Setup downloads the pinned revision of the file (size and SHA-256 checked), prepares the pack (`tools/iq_pack.py
--compat-bf16`: the Hadamard table and the dense weights, 1.5 GB), the draft head (Strata's own MTP head, prepared from
the original model as for every Qwen3.8-Flash-Next file) and the start script. Needs engine 0.1.45.1 or newer: the pack is native_experts.txt v5 (Hadamard-folded), which the engine reads since the
APR port, and the released 0.1.45 stops at v4. A ready-made engine that is older is replaced by one compiled from this source
(setup), or the app says the model needs a newer engine and offers Update engine.

**Supported:** NVIDIA (CUDA, Linux and Windows) and AMD (HIP, experimental: the authors validated gfx1151; we ran
gfx1201 with identical output to rc2). **Not supported:** Intel Arc (SYCL) and Apple Metal: the engines there have no
trellis kernels; the Metal engine refuses the file at load with the reason.

**Pictures.** Gyro reads pictures with the original model's picture reader file (the same one every Qwen3.8-Flash-Next
model uses; setup downloads it with the model). The reader opens only the vocabulary and the header of the Gyro file, so
the trellis tensors do not matter to it. Pictures are on by default, with the CPU reader under 24 GB of graphics memory and
the card's reader from 24 GB. This needs engine 0.1.45.1 or newer, like the Gyro files themselves.

**RTX 50 cards (sm_120).** Gyro gives correct output on them with engine 0.1.45.3. The fused gate/up SwiGLU kernel, which
is the default on compute capability 12.0 cards, handles only rows of 80 dot products; Gyro's trellis rows have 160, and
before the fix they went through it anyway. Gyro-S then answered with loops, garbage or nothing: on an RTX PRO 5000, 0 of
8 test questions were answered right before the fix and 8 of 8 after it (greedy and with the app's sampling). Now those
rows use the full-warp kernel, and the identity gate (10 prompts, 4K and 20K prompts) is identical to the run with the fused
kernel off. If Gyro misbehaves on an RTX 50 card, update the engine.

**Prompt speed and the ggml build.** The trellis prompt kernels (MMQ) are agentionai's, from their llama.cpp fork (MIT).
Strata builds against upstream ggml, which does not have them, so it adds them itself at configure time:
`cmake/strata_ggml_tq_overlay.cmake` writes a copy of the pinned ggml-cuda headers into the build folder with the three
trellis types routed like IQ2_XS (the tile loader is the fork's, `third_party/ggml-tq/`); only Strata's own MMQ library sees
the copy, every other type is compiled from the untouched files (identity gate below). The ready-made engines read Gyro
prompts through it as fast as rc2 does, and bit for bit the same. If the pinned ggml changes and an anchor of the script
no longer matches, the overlay switches itself off with a message and prompts take the slower FP16 path. A build against the
fork (`-DSTRATA_GGML_DIR=<agentionai/llama.cpp>`, as rc2 does) uses the fork's own instances instead.

## Credit and licenses

The trellis types, the Hadamard rotation, the CPU and GPU kernels and the pack step were written by Laurent Zuijdwijk
(agentionai) in agentionai/Strata `rc2` (9ff02c9, MIT, Strata v0.1.40.3 plus Gyro) and ported here commit by commit with
his authorship kept; `third_party/ggml/ggml-common-tq.h` and the MMQ instances come from agentionai/llama.cpp (MIT, the
ggml authors). The weights are agentionai's quantization of Qwen3.8-Flash-Next under the Qwen Community License 1.0
(<https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF/blob/main/LICENSE>).

## How it runs (by hand)

The rest of this page is the authors' own guide for a manual install and their measurements on their machines (RTX 5090,
3090, 4090 and a Strix Halo, with Strata rc2). The numbers are theirs; ours are in the next section.

## 1. Build

Setup does these steps for you; by hand, from this checkout. For the fast prompt path Strata's GGML dependency is the agentionai llama.cpp fork, which has the trellis types:

```sh
git clone --depth 1 https://github.com/agentionai/llama.cpp ../agention-llama.cpp
# NVIDIA. RTX 50 series: use CUDA 13 (CUDA 12.8 crashes the prompt path on sm_120: issues #220, #224)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_GGML_DIR=$PWD/../agention-llama.cpp
# AMD (experimental: kernel parity and an end-to-end run validated on gfx1151; in a non-root container add
# --group-add $(getent group render | cut -d: -f3), otherwise HIP finds no device)
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
.venv/bin/python tools/iq_pack.py \
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
.venv/bin/python -m serve.server --engine strata --config strata-gyro-s.json --port 8080
```

The web interface is at `http://127.0.0.1:8080`; API clients use `http://127.0.0.1:8080/v1` (OpenAI) or
`/v1/messages` (Anthropic). `--expert-cache auto` sizes the expert cache to the free VRAM: on a 32 GB card at
128k context it holds 23,464 of 24,576 experts (22.9 GiB). On smaller cards it holds fewer and streams the rest
from RAM; nothing else changes.

## Measured here (the port, 2026-10-15/16)

One AMD Radeon AI PRO R9700 (gfx1201, ROCm 7), one card of four, exclusive box lock, `--prefill auto --spec 4 --spec-min-p 0.5
--mtp --kv int8`, 22K context, interleaved pairs (the arm that goes first alternates), medians, the files of the first
Gyro release (2026-10-07). A = agentionai/Strata rc2 9ff02c9, B = this port.

| B, ggml of | file | decode tok/s A / B (B/A) | pairs B faster | prompt 4K ms A / B | prompt 20K ms A / B |
| --- | --- | ---: | --- | ---: | ---: |
| agentionai fork | Gyro-S | 85.4 / 89.7 (1.051) | 5/5 | 5418 / 5313 (0.98) | 14358 / 13859 (0.97) |
| agentionai fork | Gyro-M | 72.7 / 82.7 (1.136) | 5/5 | 5183 / 5520 (1.07) | 13421 / 14164 (1.06) |
| upstream + overlay (the ready-made engine) | Gyro-S | 85.3 / 89.6 (1.050) | 5/5 | 5413 / 5298 (0.98) | 14372 / 13854 (0.96) |
| upstream + overlay | Gyro-M | 72.6 / 82.6 (1.138) | 5/5 | 5682 / 5524 (0.97) | 15005 / 14160 (0.94) |
| upstream, no overlay (FP16 prompt path) | Gyro-S | 85.3 / 89.6 (1.050) | 5/5 | 5425 / 6939 (1.28) | 14364 / 21677 (1.51) |

Decode is faster than rc2's because the port sits on the newer Strata (0.1.45); the same engine change makes IQ3_XXS 7% faster
(rc2 86.9, port 93.1 tok/s, 2 pairs). Prompts of the ISTA files gained 11-16% from it too (IQ3_XXS 4K 6184 to 5480 ms, 20K
16587 to 13864 ms). Gyro-M's first run against rc2 showed its prompts 5-7% slower; in the second rc2's own Gyro-M prompts
took 15.0 s instead of 13.4 s (the 92 GB file's page-cache state), the port's stayed at 14.2 s, and Gyro-M's prompts are 3-6%
faster than rc2's: that difference was rc2's baseline, not a regression. Without the overlay (plain upstream ggml) the trellis
prompt kernels are missing and prompts go through the FP16 path: 28% (4K) to 51% (20K) slower than rc2, and a different
rounding. Decode is unaffected either way.

Identity (the fixed gate flags, 10 prompts, exact_pp 4096 and 20000 tokens, same card for both arms):

- Q2_0 and IQ3_XXS, f-split against the port (upstream ggml): 10/10 identical, residual dumps and outputs IDENTICAL.
- Gyro-S and Gyro-M, rc2 against the port with the fork's ggml and with upstream + overlay: 10/10 identical; exact_pp 4096 /
  20000 residual dumps and outputs IDENTICAL (run alone; a run beside other engines on the box once showed a different dump).
- Q2_0 and IQ3_XXS, f-split against the port with the overlay on, on AMD gfx1201 and on NVIDIA sm_86: IDENTICAL.
- The pack step (`tools/iq_pack.py --compat-bf16`) writes byte-identical files to rc2's tools for Gyro-S.

## Measured (the authors', rc2)

All runs: greedy or the stated sampling, same prompt tokens as llama.cpp, MTP draft `--spec 4 --spec-min-p 0.5`,
int8 KV. "llama.cpp" is the agentionai fork with the CUDA trellis fixes (in its `main` since 2026-10-04).

### RTX 5090 (32 GB), Gyro-S, all experts on the GPU (2026-10-03)

Decode tokens/s, single prompts of 34-177 tokens, 384 new tokens, `--max-context 16384` (all 24,576 experts cached):

| | prose | JSON | code |
|---|---:|---:|---:|
| Strata | 139.9 | 221.3 | 209.0 |

Server, rc2's `gyro_setup.sh` config (128k context, 23,463 experts cached), thinking off, temperature 0.7,
top-k 20, top-p 0.95, min-p 0.05, 512 new tokens, 3 repetitions:

| | prose | JSON | code | copy |
|---|---:|---:|---:|---:|
| Strata | 131.7 | 200.1 | 195.2 | 201.8 |
| llama.cpp, MTP draft | 116.8 | 213.0 | 179.0 | 220.7 |
| llama.cpp, no draft | 121.8 | 123.2 | 122.5 | 121.2 |

Long real-text prompts, prefill tokens/s: 4k tokens 3,456 (llama.cpp 2,520), 16k tokens 4,883 (llama.cpp 2,471).

### RTX 5090, Gyro-M (2026-10-03)

The experts (29.9 GiB) do not all fit next to the KV cache; `auto` caches 18,007 of 24,576. Same server protocol:
prose 113.8, JSON 170.8, code 153.0, copy 175.7 tokens/s; prefill 4k 3,369, 16k 4,854 tokens/s. llama.cpp needs
`-ncmoe 8` for this file on a 32 GB card and decodes at 70.8 tokens/s without a draft.

### RTX 3090 (24 GB), 2026-10-03

Same server protocol, 128k context, `auto` cache:

| | experts cached | prose | JSON | code | copy | prefill 4k | prefill 16k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gyro-S | 15,756 | 31.8 | 45.2 | 42.2 | 45.1 | 1,402 | 1,863 |
| Gyro-M | 11,811 | 27.8 | 39.4 | 34.4 | 37.7 | 1,363 | 1,910 |
| llama.cpp, Gyro-S, `-ncmoe 10`, 32k context, no draft | (10 layers on the CPU) | 16.0 | 16.2 | 16.2 | 16.5 | 468 (`-ncmoe 11`) | 479 |

On a 24 GB card, cache misses run on the CPU, so the CPU matters: this box had a Zen 2 EPYC (AVX2, no AVX-512).
The llama.cpp row is the build before the CUDA trellis fixes; those added about 3 % at 24 GB, where the CPU layers
dominate.

The Strata rows predate Strata's own CPU kernels for the trellis types (`src/kernels/cpu/tq_*.cpp`, AVX2 and
AVX-512, picked at run time). Before them a cache miss ran ggml-cpu's `vec_dot`, one token at a time; the new kernels
take about 45 % of its time per expert on AVX2 and decode a block once for all the tokens of a verify window
(`build/tq_cpu_parity --bench` measures them on your CPU). `STRATA_NO_TQ_KERNELS=1` goes back to ggml-cpu;
`STRATA_TQ_ISA=avx2` or `scalar` caps the instruction set (for comparisons).

### Smaller caches (RTX 5090, 2026-10-02, before the int8 decode kernels)

| Expert cache | prose | JSON | code |
|---|---:|---:|---:|
| 24,576 slots (all experts) | 116.4 | 194.3 | 173.5 |
| 22,000 slots | 111.4 | 175.7 | 166.1 |
| 9,000 slots (about a 16 GB card's room) | 81.4 | 74.0 | 96.8 |
| 3,000 slots (about an 8 GB card's room) | 48.0 | 52.4 | 57.9 |

The smaller caches were emulated on the 5090 (fast GPU, PCIe 5.0 x16 at ~51 GB/s). A real 8 or 16 GB card is a
slower GPU, often on a slower link: read those rows as "it stays usable", not as a prediction for a specific card.
The decoded output was read and is fluent and correct; token-level parity against llama.cpp has not been
checked. A real chat at 128k context worked through the web interface.

### Two GPUs (2026-10-04, 2× RTX 4090, no NVLink/P2P)

- **Layer split** (Strata splits the layers across both cards automatically) works with Gyro, including the
  idle-card prompt helper (`STRATA_PREFILL_HELP=1`). Repeated runs give identical output.
- **Decode with a second card** (`--peer-device 1` or `--expert-cache-device1 auto`) works and matches single-GPU output.
- **Prompt processing on the peer card** needs GPU-to-GPU access (NVLink or data-centre GPUs); without it Strata keeps
  the prompt on the first GPU and logs `no P2P`.

Single RTX 4090 smoke run (Gyro-S, MTP draft, greedy): code 107.5, prose 84.3, JSON 76.3 tokens/s.

## Numerics

- **Decode** (Strata's own expert kernels) reads the trellis weights through an int8 copy of the codebook with one
  global scale (`tq_lut_i8`, relative RMS error 0.85 % of the codebook's standard deviation, far below the
  quantization error itself), against q8_1 activations.
- **Prompt processing** decodes them otherwise: the FP16 path through the fp16 codebook (`tq_lut_f16`, what GGML's
  own dequantizer reads); the MMQ path through the llama.cpp fork's kernels, which read the int8 codebook as well
  on the fork's current `main`.
- So decode and prompt results agree within tolerance, not bit for bit. `native_expert_parity` checks one expert
  on the decode kernels against GGML's float dequantizer and a float SwiGLU: relative L1 error below 3e-2, the
  bound every expert type gets there. Its "bitwise equal" lines compare Strata's decode-once and per-entry decode
  kernels with each other, not with a float reference.

## Known limits

- **Not in this port** (rc2 has them; see the list at the end of this page): the prompt helper on a second GPU for a Gyro
  pack. Setup of two cards still runs the layer split and the decode peer; the prompt peer says it is not used for a
  Hadamard-folded pack and the prompt stays on the first card. Gyro-L, Intel Arc (SYCL) and Apple Metal are not supported.
- **Source builds:** the SYCL tree builds with these sources (P1006, 0 errors) but has no trellis GPU kernels; the Metal
  engine builds and refuses the file at load (`native_expert_supported`).

- **Not measured yet:** token parity against llama.cpp.
- **Benchmark mode only:** `strata --tokens-file ... --max-new N` decodes N tokens past the end of the answer;
  compare speeds only up to the answer's end.
- **AMD:** the trellis kernels pass parity on gfx1151 (ROCm 7.2.1) and a short end-to-end run there generated
  correctly (~28-31 tokens/s); no speed tuning or RDNA4 run yet.
- **Gyro-M:** 29.9 GiB of experts do not all fit a 32 GB card next to the KV cache; `auto` streams the rest.
- **Not validated end to end yet** (Strata runs them and logs a warning): an AMD GPU other than gfx1151, and
  prompt processing on a peer GPU over P2P.
