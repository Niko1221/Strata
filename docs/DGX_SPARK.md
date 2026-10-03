# Experimental: NVIDIA DGX Spark (GB10, aarch64)

The DGX Spark pairs an ARM CPU (10 Cortex-X925 + 10 Cortex-A725 cores) with a Blackwell GPU (GB10, sm_121); the CPU
and the GPU share 128 GB of unified memory. Strata runs there from a source build. It is community-tested, on one Spark,
with Qwen3.8-Flash-Next **IQ2_XS** and Unsloth's **UD-Q4_K_XL** installed by `setup.sh`. The other setup models use
expert formats that have an ARM code path (below) but have not been run on a Spark. There is no ready-made ARM engine.

## Install with setup (recommended)

```sh
./setup.sh
```

The same questions as on a PC. What setup does differently on a Spark:

- **GPU:** the GB10 reports no memory of its own (`nvidia-smi` shows `[N/A]`), so setup counts the system's RAM as
  the GPU's.
- **Engine:** compiled on the Spark, once (the ready-made engine is x86-64). It needs the CUDA 13 toolkit and a C++
  compiler; both were already installed on the Spark it was tested on. ggml's CPU code is compiled with the
  features read from `/proc/cpuinfo` (`-march=armv8.6-a+dotprod+i8mm+fp16` on the GB10): gcc's `-mcpu=native`
  finds none on this CPU, which would leave ggml without its dot-product instructions.
- **Start settings:** `--mmap-experts` and no KV streaming (see below).

A manual build, if you do not use setup:

```sh
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121 -DSTRATA_GGML_DIR=third_party/llama.cpp \
  -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm+fp16
cmake --build build --target strata
```

## How it runs with unified memory

- **Every expert on the GPU.** With `--expert-cache auto` the cache holds all 24,576 experts (33.0 GiB for IQ2_XS, 71.7 GiB for
  UD-Q4_K_XL), so
  the CPU computes none: the per-token expert work is the GPU's alone.
- **`--mmap-experts`:** the experts are read from the GGUF in place. A RAM arena of the experts (the default on a PC)
  would be a second copy of the expert cache in the same memory.
- **The cache's size counts reclaimable memory.** `cudaMemGetInfo` reports only free RAM, not the page cache, which
  here holds mostly the GGUF's own pages and gives way to the GPU's allocation. On a unified memory system the engine sizes
  the cache from `MemAvailable` less 6 GiB for everything else; `STRATA_UMA_HEADROOM_GIB=N` changes that margin.
- **No KV streaming:** it moves KV to RAM to free VRAM, and there is no separate VRAM.

## Speed

Qwen3.8-Flash-Next IQ2_XS on a DGX Spark (driver 580.173.02, CUDA 13.0, Ubuntu 24.04), started with setup's own
`run-iq2_xs.sh` (128K context, `--kv int8`). [llama-benchy](https://github.com/eugr/llama-benchy) 0.4.0, 3 runs per
row after a warm-up, `--no-cache`, one request at a time, 128 tokens written per request:

| Prompt | Context | Prompt tokens | Prefill | Decode | MTP acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 0 | 2,049 | 957 tok/s | 55.7 tok/s | 72% |
| 8K | 0 | 8,194 | 1,357 tok/s | 59.7 tok/s | 74% |
| 2K | 16K | 18,433 | 1,410 tok/s | 63.7 tok/s | 73% |
| 8K | 16K | 24,577 | 1,503 tok/s | 57.3 tok/s | 80% |
| 2K | 64K | 67,586 | 1,538 tok/s | 56.2 tok/s | 77% |
| 8K | 64K | 73,729 | 1,548 tok/s | 56.4 tok/s | 75% |

Prefill and decode are the server's own (`strata serve: prompt ... read in ... generated in ...` in the
log), averaged over the 3 runs. llama-benchy's `pp` column cannot time Strata's server: the first chunk of the stream
is sent before the prompt is read, so benchy's time to first response is a few milliseconds. Its time to the first
token is right (2.18 s at 2K, 6.11 s at 8K, 47.8 s at 8K after 64K).

Short chat answers decode at 68-79 tok/s (a coding, a factual and a maths question, `reasoning_effort` low).

**UD-Q4_K_XL** (`./setup.sh --family unsloth`, 262K context, `--kv int8`): all 24,576 experts in the GPU's cache,
82.2 GiB of GPU memory for the whole engine; the 28.8 GB PLE table stays on the SSD. On a unified memory system setup
keeps no RAM budget of experts (`--resident-budget-gib`, the 64 GB PC's setting): it would be a second copy of them
in the memory the expert cache uses. The same benchmark:

| Prompt | Context | Prompt tokens | Prefill | Decode | MTP acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 0 | 2,049 | 590 tok/s | 38.4 tok/s | 74% |
| 8K | 0 | 8,193 | 1,117 tok/s | 39.2 tok/s | 72% |
| 2K | 16K | 18,433 | 1,058 tok/s | 41.8 tok/s | 80% |
| 8K | 16K | 24,577 | 1,173 tok/s | 39.7 tok/s | 78% |
| 2K | 64K | 67,585 | 1,165 tok/s | 39.4 tok/s | 79% |
| 8K | 64K | 73,729 | 1,187 tok/s | 41.5 tok/s | 81% |

## When the CPU computes experts

With fewer cache slots than experts (a smaller `--expert-cache`), the CPU computes the rest. On aarch64 every expert
format, Q2_0 included, goes through ggml-cpu's NEON dot products; Strata's own multi-token kernels for the x86 CPUs
(AVX-512 / AVX2) have no ARM version yet. Measured on IQ2_XS with a 2,048-slot cache, where the CPU computed ~22 of the
~29 experts each layer routed: 19.7 tok/s, against 82.9 tok/s with every expert on the GPU, and the same greedy
answer token for token.

## Not covered

- Models other than IQ2_XS and UD-Q4_K_XL: not run (their formats are covered: IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S,
  IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL and Q2_0 experts all have ggml NEON paths).
- Images (`--vision`): built with the same CPU flags, not tested.
- ARM CPUs without a CUDA GPU, and other ARM boards with an NVIDIA GPU: not tested.

## Tests

`ctest` on the Spark: 51 of 52 pass. `ple_parity` needs the Q2_0 model's second shard, which was not present.
`expert_multi_test` checks the AVX-512 Q2_0 kernel and is registered on x86 only.
