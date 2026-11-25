# Run hy4-flash on Apple Silicon (Metal)

[简体中文](README_MAC-Metal.zh-CN.md) · [NVIDIA](README.md)

This is the Metal backend: a native hy4-flash engine for Apple Silicon, not a port of the CUDA one. It brings the architecture, a set of Metal kernels and fusions, MTP speculative decoding, pictures, and a serve entry point that `strata launch` drives directly. Decode output is bit-for-bit identical to the reference; prefill uses simdgroup MMA with half operands and is accepted by a separate numerical gate.

## Models

The same GGUF files and pack as the CUDA engine. Expert quantizations including IQ2_XXS are recognised on Apple Silicon.

| | |
|---|---|
| Model | first GGUF shard, `--model-path` |
| Pack | tokenizer, chat template, stop ids, dense weights in the engine's layout, `--pack-path` |
| Vision shard | an mmproj GGUF for the `hy_v4_flash` projector, beside the model or named by `STRATA_HY4_FLASH_VISION_SHARD` |

The pack is **not** produced by the build. Either it arrives with the model, or you build it from the shards:

```sh
python terminal/tools/strata_pack.py build  --gguf <first shard> --out <pack dir>
python terminal/tools/strata_pack.py verify --gguf <first shard> --out <pack dir>
```

`verify` re-reads the written pack and requires `decode(pack) == dequant_ggml(source)` bit for bit, for every tensor. A pack that has not passed it should not be used.

## Hardware and memory

Apple Silicon with unified memory; measured on an M4 Pro with 64 GB. Run on AC power — on battery the GPU clocks down. Keep the model on local SSD: expert weights are read on demand, so an external drive becomes the bottleneck.

Memory for an IQ2 model, in GB:

| Component | Size | Notes |
|---|---|---|
| Dense weights | ~10.8 | Resident |
| Embeddings | ~0.7 | Resident |
| Expert weights (all) | ~53.4 | On disk; enter the arena when routing hits them |
| Expert arena | computed | LRU plus cross-layer read-ahead |

The arena is not a fixed default. It is the smallest of the GPU working set, the RAM macOS must be left with, the total expert bytes, and a conservative cap — and the engine prints which of the four bound it. You do not need to raise any GPU memory limit by hand.

**Do not pin the arena size.** Setting `STRATA_METAL_ARENA_GB` replaces the plan and switches off the live shrink/grow guards, so on a smaller machine the result is an out-of-memory kill rather than a smaller arena. If memory runs short, let the engine shrink first and only then reduce context length.

## Build

metal-cpp headers are the one prerequisite: download the archive for your macOS version from <https://developer.apple.com/metal/cpp/> and unzip it. The directory must contain `Metal/Metal.hpp`.

```sh
cmake -S engine/metal -B build-metal -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_METAL_ENGINE=ON \
  -DMETAL_CPP_DIR=$PWD/metal-cpp
cmake --build build-metal -j
```

Two of the resulting binaries are entry points: `strata` (the server, used by `strata launch`) and `strata_hy4_flash_metal` (an offline single-shot and benchmarking driver). The first build needs an accepted Xcode licence.

## Launch and query

```sh
strata launch --no-tui --yes \
  --model-path <first shard> \
  --pack-path <pack dir> \
  --engine-path build-metal/strata \
  --port 8099
```

| Flag | Meaning |
|---|---|
| `--model-path` | First GGUF shard |
| `--pack-path` | Pack directory |
| `--engine-path` | The Metal `strata` built above |
| `--port` | HTTP listen port |

`/v1/models` then lists the model as `loaded` and `/v1/chat/completions` takes ordinary requests.

Note that hy4-flash **thinks by default**: with no `reasoning_effort` set, the text goes to `reasoning_content` and `content` stays `null` until the reasoning block closes. A client that reads only `content` will see nothing until then.

### Pictures

Add `--vision on`. The vision shard is looked for beside the model, or named directly:

```sh
strata launch --no-tui --yes --vision on \
  --model-path <first shard> --pack-path <pack dir> \
  --engine-path build-metal/strata --context 32k --port 8099
```

Send pictures as an `image_url` content part with a `data:` URL. A picture of `h × w` (both rounded to multiples of 32) occupies `(h/32) · (w/32 + 1)` positions — the `+1` is one image-newline row per merged row — so a large picture is worth thousands of context positions. Give the server enough `--context`.

Geometry is pinned to the HuggingFace image processor rather than inferred: the target size, the patch grid and the trunk's three-axis rotary positions are each checked against it as integers. Malformed picture bundles are refused by name rather than decoded into noise.

Text-only decode is bit-for-bit identical whether or not the vision path is built, so enabling pictures costs a text request nothing.

### Speculative decoding

MTP is on by default and drafts 2 tokens per round, verified by the trunk in one pass. It changes no output: the accepted tokens are exactly those plain decoding would have produced.

## Launch options

`strata launch` picks sensible values on Apple Silicon, so you usually need none of these.

| Switch | Default | What it does |
|---|---|---|
| `--vision` | off | `on` enables pictures |
| `--prefill-chunk` | 4096 on Mac | Chunk size for chunked prefill |
| `--draft-tokens` | 2 | Tokens the MTP head drafts per round |
| `--context` | model default | Raise it before sending large pictures |

### Engine settings

| Variable | Default | What it does |
|---|---|---|
| `STRATA_METAL_ARENA_GB` | unset | Pins the arena and disables its guards; see above |
| `STRATA_HY4_FLASH_VISION_SHARD` | unset | Path to the mmproj GGUF |
| `HY4_FLASH_MTP_EXPERT_GB` | engine default | Expert budget during drafting |

Behaviour on non-Darwin platforms is unaffected by this backend.

## Measured results

Mac mini, M4 Pro (16-core GPU), 64 GB unified memory; IQ2 model, MTP k=2, 8K context.

| Metric | Start | Now |
|---|---|---|
| Prefill | 35.8 tok/s | **125–127 tok/s** |
| Decode (256 generated) | 7.31 tok/s | **17.300 tok/s** |
| Decode (1024 generated) | — | 18.979 tok/s |

Decode is the geometric mean of four prompts' medians. MTP acceptance rises with generation length, so the 1024 figure flatters; 256 is the figure the protocol specifies.

The decode ceiling is structural: every token reads about 11.7 GB of weights, 86% of it dense, while the arena can only cache the 14% that is experts. At a measured 252 GB/s that caps plain decoding near 21.6 tok/s — MTP raises the ceiling by amortising the dense read across several verified rows.
