# GLM-5.3 (glm-dsa) - the second model: how it runs, what was measured, what is missing

Strata was built for one model. This page is the second one: **[GlmMoeDsaForCausalLM]**, GLM-5.3, measured on
`D:\models\GLM-5.3-colibri-int4-g64` (141 safetensors shards, 116,915 tensors, 419,282,314,240 B of data).
Nothing here is estimated: every number is either read out of a shard header, derived from the geometry, or
measured on the PC named next to it.

GLM-5.3 runs on its own engine, **`strata-glm`** (`src/glm/`, `include/strata/glm/`), behind the same server, web
app and API as the Qwen models. The Qwen engine (`strata`) is not changed by it.

> **On this page:** [Run it](#run-it) · [The geometry](#the-geometry-measured) ·
> [The container's format](#the-containers-format) · [What the container does not carry](#what-the-container-does-not-carry) ·
> [The engine](#the-engine) · [Checked against colibri](#checked-against-colibri) · [Speed](#speed-measured) ·
> [The drive](#the-drive) · [What is missing](#what-is-missing)

## Run it

You need the int4-g64 container (419 GB, `Justvugg/GLM-5.3-colibri-int4-g64`), 64 GB of RAM (about 20 GB free is
the floor: 11.6 GB of weights stay in RAM) and an NVMe SSD. An NVIDIA card is optional: it reads prompts about 3x
faster and holds a second expert cache.

**With the installer:** `START-HERE.bat` (`./setup.sh`) offers GLM-5.3 as a model choice, or directly:

```text
SETUP.bat --family glm                                   download (resumable, a pinned revision), build, start
SETUP.bat --family glm --glm-model-dir D:\models\GLM-5.3-colibri-int4-g64     use a folder you already have
          [--backend cpu | --gpu N]  [--context N]  [--glm-kv f32|bf16|fp8]
```

It checks for 64 GB of RAM and the free disk space first, builds `strata-glm` (with CUDA when a GPU is asked for and
nvcc is found; on Windows it fetches NVIDIA's checksum-verified cuBLAS when the toolkit lacks it), then writes the
same three files as `tools/glm_setup.py`. GLM-5.3 runs one request at a time, on one GPU, text only.

**By hand**, for a model folder you already have:

```text
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release [-DSTRATA_ENABLE_CUDA=ON]
cmake --build build --target strata-glm
python tools/glm_setup.py --model D:\models\GLM-5.3-colibri-int4-g64 --exe build\strata-glm.exe [--gpu 0]
          [--max-context 32768] [--kv bf16]
run-glm53.bat                         (Linux: ./run-glm53.sh)
```

Both write the tokenizer folder (`data/glm53/tokenizer`, with `serve/glm/chat_template.jinja`), the run config
`strata-glm53.json` (`"family": "glm"`) and the launcher. The server is the usual one, on port 8080: chat,
`/v1/chat/completions`, `/v1/messages`, tool calls and thinking (`reasoning_effort` low / medium / high / none). The
server refuses what GLM-5.3 does not do (images, parallel requests, slot save/restore, `POST /v1/vram`).

**Context.** The KV cache holds 576 values per token per layer. Per token: f32 180 KB, bf16 90 KB, fp8 46 KB; at
32,768 tokens that is 5.9 / 2.9 / 1.5 GB of RAM, taken from the expert cache. The engine accepts up to 131,072, but
reading a prompt is the limit: about 10 tokens/s on the GPU and 3.7 on the CPU at 1,536 tokens (see Speed), and the
attention grows with the context. 32,768 with bf16 is the largest that is still practical.

**The engine alone**, from token ids: `strata-glm --model DIR --tokens 154822,154824,785,6722,315,9621,374 --gen 16`
(or `--tokens-file`); `--tf` prints the argmax and top-5 at every position, `--logits-out FILE` writes the logits.

| Option | Default | |
| --- | --- | --- |
| `--max-context N` | 8192 | up to 131,072 |
| `--kv f32\|bf16\|fp8` | f32 | the KV format (see "Checked against colibri" for what bf16 and fp8 change) |
| `--gpu N` | off | CUDA device: VRAM expert cache, prompts on the GPU |
| `--vram-gb G`, `--vram-reserve-mib N` | all free VRAM less 700 MiB and 1.5 GB of working memory | the VRAM expert cache |
| `--promote-after N` | 2 | uses before an expert is copied to VRAM |
| `--cpu-prefill` | off | with `--gpu`: read prompts on the CPU |
| `--prefill layer\|chunk`, `--prefill-chunk N` | layer, 1024 | the whole prompt per layer, or chunks of N tokens |
| `--prefetch` | off | start reading the next layer's likely experts early |
| `--ram-gb G`, `--expert-ram-gb G` | 85% of free RAM | the RAM budget / the expert cache alone |
| `--threads N`, `--io-threads N` | physical cores, 8 | |
| `--features` | | prints what the build has (`{"family":"glm","cuda":true}`) |

## The geometry (measured)

| | Value | How it was measured |
| --- | ---: | --- |
| Layers | 78 (75 sparse, 3 dense) | `mlp_layer_types`, confirmed by the tensors present |
| Hidden | 6,144 | `input_layernorm.weight` shape |
| Routed experts | 256 per sparse layer, 8 active | tensor count per layer, `num_experts_per_tok` |
| Heads | 64 | `num_attention_heads`, confirmed by `q_b_proj` and `o_proj` shapes |
| MLA | `q_lora_rank` 2,048, `kv_lora_rank` 512 | the two layernorm shapes |
| MLA head | 192 nope + 64 rope, `v_head_dim` 256 | `kv_a_proj_with_mqa` is [576, 6144]: 576 − 512 = 64 |
| MoE intermediate | 2,048 (dense layers 12,288) | expert tensor row count |
| Router | sigmoid, `noaux_tc`, top-8, scaling 2.5, `norm_topk_prob` true | `config.json` |
| RoPE | θ = 8,000,000, interleaved, on the 64-wide slice | `config.json` (`rope_parameters`, `rope_interleave`) |
| Indexer | top-k 2,048, key 128, 32 heads, 21 `full` layers | `config.json`; the tensors are absent (see below) |

The shape table in `tools/glm53_geometry.py` is a measurement, not a guess: the container stores every tensor
**flat**, so the header records a byte count rather than `[rows, cols]`. The table reproduces all 116,915 recorded
spans, and every F32 tensor's flat length equals `rows * cols`.

`config.json`'s own `quantization_config` says fp8 e4m3 with 128×128 blocks. That is the **original** checkpoint's
config and it is wrong for this container — trusting it would make the code plane 4x too small and the scale plane
2x too small.

## The container's format

A quantized tensor is two header entries: `name` (U8 codes) and `name.qs` (F32 scales).

- Codes: 4 bits per element, two per byte, **LSB-first** (byte j holds element 2j in the low nibble and 2j+1 in the
  high nibble), stored as `q + 8` so the bias is −8. Verified by dequantising the first group of a real tensor:
  the largest absolute code is 7 and `7 × scale` is the group's maximum.
- Scales: one f32 per group of 64, so `scales == rows * ceil(cols / 64)`. Group 64 holds for every weight tensor.
- Exception: `model.embed_tokens.weight` and `lm_head.weight` are **signed int8 with one scale per row**: each byte
  is an `int8_t` in two's complement (0x80 = −128), not `code − 128`. An earlier version of this page said "bias
  −128"; reading them that way gives garbage logits (measured: the argmax wrong at every position). colibri, which
  wrote the container, reads them as `int8_t` (its fmt 1), and with that the engine matches colibri token for token.
- Norms, `mlp.gate.weight` (the router, 256 × 6144) and `e_score_correction_bias` are F32.

Per expert that is 3 × (6,291,456 code bytes + 786,432 scale bytes) = **21,233,664 B**.

**Where an expert lies.** safetensors orders a shard's tensors by dtype and then by name, so an expert's three code
planes (down, gate, up) are normally one contiguous 18,874,368 B span and its three scale planes one contiguous
2,359,296 B span: two reads. Experts that straddle a shard boundary take more (layer 3 expert 27 does); the engine
reads whatever runs of planes are adjacent. This is why the engine reads the container **in place**: the repacked
`experts.bin` that `tools/glm53_pack.py` describes would be a 408 GB copy of the same bytes.

## What the container does not carry

Counted over all 141 shard headers:

| | Tensors present |
| --- | ---: |
| indexer (`indexer.*`, `indexers_proj`, `index_kpool`) | 0 |
| MTP / NextN (`nextn`, `eh_proj`, `hnorm`) | 0 |
| hyper-connection (`hc_*`) | 0 |
| shared-expert gate | 0 |
| chat template | none (no `chat_template` in tokenizer_config.json, no .jinja file) |

Shared experts are present as `mlp.shared_experts.*` (450 tensors) and are added with weight 1, and the router as
`mlp.gate.weight` plus `mlp.gate.e_score_correction_bias` (75 each). Without the indexer, attention is dense over the
whole context. That is exactly what DSA computes up to `index_topk` = 2,048 tokens; beyond that the model was trained
to look at its top 2,048 tokens and here sees all of them (colibri does the same with this container).

## The engine

`strata-glm` runs on the CPU, with an optional CUDA backend:

| Part | File | What it does |
| --- | --- | --- |
| container | `src/glm/container.cpp` | config.json (checked against ranges), the 141 shard headers, every expert's plane runs |
| kernels | `src/glm/kernels.cpp` | int4-g64 matrix products (AVX2; the activation is permuted once per call into the even/odd nibble order), signed-int8 rows, RMSnorm, the interleaved RoPE, the prompt attention |
| expert cache | `src/glm/experts.cpp` | an LRU of 21 MB slots in RAM; misses are unbuffered reads from 8 I/O threads, each with its own file handles |
| KV | `include/strata/glm/kv.hpp` | the MLA cache rows in f32, bf16, or fp8 (E4M3 with one scale for the latent and one for the roped key) |
| CUDA | `src/glm/cuda/expert_gemv.cu` | a VRAM expert cache (filled with the experts used most often), prompt matrix products (cuBLAS SGEMM on dequantized weights), the prompt attention |
| model | `src/glm/model.cpp` | the forward pass: absorbed MLA, the sigmoid router, the shared and routed experts, the dense MLP |
| program | `src/glm/main.cpp` | `--tokens` / `--tf` / `--serve` (the line protocol `serve/server.py` speaks to `strata --serve`); a watchdog ends a forward pass that makes no progress for 30 minutes |

The forward pass, per layer: `h = rmsnorm(x)`; MLA with `kv_b_proj` absorbed (the cache holds the 512-wide normed
latent and the 64-wide roped key; the score is `(W_k^T q_nope · L + rope(q_rope) · R) / 16`); the router picks 8 of
256 by `sigmoid + bias` and weighs them by the sigmoids, renormalised and × 2.5; the shared expert computes while the
routed experts' reads arrive, and the routed ones are computed in the order their bytes are ready. RoPE pairs
(2j, 2j+1) and writes them to (j, j+32), as colibri and the checkpoint's `rope_interleave: true` do.

**A prompt goes through one layer at a time, whole** (`--prefill layer`, the default): every layer's routed experts
are read once per prompt instead of once per chunk. The engine reports progress after every layer (`PP` lines), so
the server sees a long prompt advancing. With the 1,536-token prompt below, chunks of 256 read 1,756 GB from the SSD
and a whole-prompt pass 360 GB.

**Writing answers** goes token by token: the CPU (or a VRAM-resident expert on the GPU) computes each of the 8
experts; an expert used `--promote-after` times is copied to VRAM. The conversation's KV stays between requests: a
request that starts with the last one's tokens reads only its new ones. `STOP` from the server is checked between
layers and experts, also while a prompt is read.

The server side: `tools/strata_tokenizer.py` has the `glm` pre-tokenizer (digits in runs of 1–3, no `\p{M}` class)
and reads `tokenizer.json`; `serve/glm/chat_template.jinja` is the GLM-5.2 format; `serve/frontend.py`'s
`GlmOutputParser` reads `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>`; the server ends a
reply at `<|user|>`, `<|observation|>` and `<|endoftext|>`. A config with `"family": "glm"` (or a tokenizer folder
written with the `glm` pre-tokenizer) selects all of this.

Tests without the model: `glm_kernels_test` (15 checks: the formats, the kernels against scalar or double-precision
definitions, RoPE, the prompt attention, the pool), `glm_kv_test` (the three KV formats), `glm_cuda_test` (with an
NVIDIA card: the GPU matrix products, experts and attention against the CPU, error under 1e-6),
`python -m unittest serve.test_glm` (12: the parser whole and streamed, values containing the closing tags, the
forced call, the template), `python tools/test_setup_glm.py` (8: the installer's checks, config and launcher). With
the model and the `tokenizers` package: `tests/glm/test_glm_tokenizer.py` (33 strings); with the model:
`python tools/glm_validate.py --exe build/strata-glm.exe --model DIR --out DIR` (the 12-position check on CPU and GPU
and in every KV format). The engine and its tests also build and pass on Linux (g++ 14, WSL Debian); the model was
not run there.

## Checked against colibri

colibri (`colibri.exe`, the engine that wrote this container) is the reference. On this PC (Ryzen 9 8940HX, 64 GB,
RTX 5070 Ti Laptop 12 GB, Kingston NV3 1 TB), prompt `[gMASK]<sop>The capital of France is`
(`154822 154824 785 6722 315 9621 374`):

- colibri greedy: `12089 11 264 3283 3881 369 1181 9077 3840 11 19812 17621` (" Paris, a city known for its rich
  history, stunning architecture").
- strata-glm, teacher-forced over those 19 tokens (`--tf`): the argmax at positions 6–17 is that sequence:

  | | Positions that match colibri |
  | --- | ---: |
  | CPU, f32 KV | **12 of 12** |
  | GPU (`--gpu 0`), f32 KV | **12 of 12** |
  | CPU, bf16 KV | 11 of 12 |
  | CPU, fp8 KV | 11 of 12 |

  bf16 and fp8 miss the same position (13): they pick `1947`, which f32 ranks second, 20.18 against 20.47. Greedy
  generation (`--gen 24`) gives colibri's 12 tokens and goes on (" and vibrant culture").
- The tokenizer gives the same ids as Hugging Face `tokenizers` on 33 of 33 test strings (CJK, Arabic, emoji,
  combining marks, control bytes, digit runs, every special token).
- The chat template renders byte for byte what colibri's GLM-5.2 renderer does (itself checked against zai-org's
  `chat_template.jinja`) on 10 of 10 conversations: thinking on and off, history, tools, calls, tool results.
- Through the server: "What is the capital of France?" was answered "Paris is the capital of France." (stop); a
  request with a `get_weather` tool came back as the call `get_weather {"city": "Oslo", "unit": "celsius"}`
  (finish_reason `tool_calls`).

**Not yet checked on the model:** the prompt-attention kernel (`mla_head_prompt`) and the two-rows-by-four multi-row
kernel in `src/glm/kernels.cpp` came after these runs. They pass `glm_kernels_test` against their definitions (error
under 2e-7), but the 12-position check above was not repeated with them.

## Speed (measured)

Same PC. **Writing answers**, 24 tokens after the 7-token prompt, CPU, 33.8 GB expert cache, started cold, the drive
in its fast state (below):

| | colibri 1.12 (cap 11/layer, RAM_GB 36) | strata-glm |
| --- | ---: | ---: |
| 7-token prompt | 27.2 s | 17.9 s |
| writes answers | 0.31 tokens/s | **0.49 tokens/s** |
| expert hit rate | 41% | 39% |

47 of the 60 s the routed experts took went to waiting for reads, at 3.4 GB/s; 16 or 32 I/O threads instead of 8
changed nothing. With nothing cached a token reads 12.74 GB, about 3 s on this drive in its fast state; the cache
brings it to about 2 s. The VRAM expert cache (`--gpu`, 431 slots = 9.15 GB on this card) was not measured for
answer speed: the run that was to measure it met the drive's slow state.

**Reading a prompt**, 1,536 tokens, 28 GB expert cache, before the two kernels above
(`bench/results/2026-10-06-glm53-port/`):

| | Time | Tokens/s | Read from the SSD | Waiting for reads |
| --- | ---: | ---: | ---: | ---: |
| CPU, chunks of 256 | 604.5 s | 2.5 | 1,756 GB | 250 s |
| CPU, chunks of 1024 | 508.5 s | 3.0 | 662 GB | 6 s |
| CPU, whole prompt per layer | 417.6 s | 3.7 | 359 GB | 0 s |
| GPU, whole prompt per layer | **146.5 s** | **10.5** | 360 GB | 44 s |
| GPU, chunks of 256 | 1,496 s | 1.0 | 1,752 GB | 1,138 s |

On the CPU the whole-prompt pass is compute: attention 186.5 s, experts 222.8 s. The two kernels added after these
runs target exactly that: on synthetic weights at an expert's prompt shape (48 rows, 2048 x 6144) the multi-row
product went from 5.2 to 2.7 ms, and the prompt attention unpacks each head's `kv_b` once per prompt instead of once
per query. Their effect on this prompt was not measured yet. On the GPU the attention takes 25 s and the experts
119.5 s, 44 s of it waiting for the SSD.

### The drive

The Kingston NV3 (a QLC drive without DRAM, 70% full) reads in two states. Fast: colibri's `iobench`, 20 MB random
direct reads, 3.69 GB/s with one thread and 4.54 GB/s with eight (and 4.15 GB/s later). Slow: after about 2.5 TB of
reads in an hour, the same test gave **0.11 GB/s**, Windows reported 260 ms per read, and writing answers fell from
0.49 to **0.03 tokens/s**. The drive was at 43 °C (limit 75 °C), so it was not heat. It came back to 4.15 GB/s after
about 4 minutes without reads. An earlier version of this page said "an NVMe SSD measured at about 200 MB/s, about 64
seconds per token": that was this slow state, not the drive's speed. A drive with DRAM, or a model folder on a less
full drive, is likely to stay in the fast state; this was not measured.

## What is missing

- **The two new CPU kernels on the model** (see "Not yet checked"): the 12-position check, and the 1,536-token CPU
  prompt for their speed.
- **Answer speed with the VRAM expert cache and with `--prefetch`**: built and unit-tested, not measured.
- **The GPU prompt path's own costs:** every call copies its weights over PCIe from pageable memory and dequantizes
  them to f32 for SGEMM, and the GPU attention copies the layer's whole KV each call (why chunked prompts are 10x
  slower on the GPU). An int4 GEMM and a KV kept in VRAM are the next steps.
- **DSA, MTP, mHC**: no weights in this container (see above). `--spec` does not apply.
- **The model on Linux**: the engine and its tests build and pass there, the model was not run.
- `include/strata/plan/plan.hpp`'s glm-dsa planner and `tools/glm53_pack.py` describe a repacked pack that the engine
  does not use; they stay as the measured geometry (their tests still pass).
