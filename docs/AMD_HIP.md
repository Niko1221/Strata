# Experimental AMD HIP backend (gfx1100, gfx1101, gfx1200, gfx1201)

This is a Linux source build for the RX 7900 XT / XTX (RDNA3, gfx1100) and the
RX 9070 / 9070 XT / Radeon AI PRO R9700 (RDNA4, gfx1201; see [RDNA4](#rdna4-gfx1201)). The RX 7800 XT / 7700 XT
(gfx1101) and the RX 9060 XT (gfx1200) were validated by their owners (see [Community-validated
cards](#community-validated-cards)). It is opt-in; the NVIDIA installer and CUDA build remain the default. Other AMD
architectures, wave64, Windows HIP, and mixed AMD/NVIDIA execution are outside this contribution.

The backend maps the CUDA-shaped runtime and BLAS calls to HIP/hipBLAS, uses
RDNA3/RDNA4's signed integer dot instruction for quantized kernels, and supplies
wave32 shuffle/packed-byte operations. CUDA-only QSA matrix instructions have
an ordered FP32 fallback. Prefill supports both dequantization plus hipBLAS GEMM and opt-in HIP ggml MMQ.
An optional, calibrated hipBLASLt path accelerates dense projections (per-architecture tables in `tools/hip`).
This does not claim bit-identical model answers across backends. See
[performance settings and evidence](AMD_HIP_PERFORMANCE.md).

## Install with setup (recommended)

On Linux with an RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT or Radeon AI PRO R9700 and
the kernel's amdgpu driver (no ROCm install needed):

```sh
./setup.sh --backend hip
```

- **Detection:** setup finds the card through the kernel's KFD topology. Integrated Radeon GPUs are listed as not
  supported. On a PC without an NVIDIA card Strata can use, `--backend hip` is chosen automatically.
- **ROCm:** a system ROCm 7 in `/opt/rocm` (or `$ROCM_PATH`) with hipcc and hipBLAS is used when present. Otherwise
  (or when it is older than 7.0) ROCm is installed into `.venv` from AMD's TheRock wheels (~10 GB, no sudo), pinned
  to the version this backend was tested with, from the card family's index: `gfx110X-dgpu` for gfx1100 / gfx1101,
  `gfx120X-all` for gfx1200 / gfx1201 (`STRATA_ROCM_VERSION` / `STRATA_ROCM_INDEX` override them).
- **Engine:** compiled on your PC for the card's architecture (10-20 minutes, once; again after a `git pull` that
  changes it, or when you pick a card of another architecture). This needs a C++ compiler and git
  (`sudo apt install build-essential git`).
- **hipBLASLt tuning table:** setup uses `tools/hip/<arch>-hipblaslt-<version>.txt` only when it matches both the
  card's architecture and the installed hipBLASLt version (read from `hipblaslt-version.h`; 1.2.0 is `100200`).
  Otherwise it says so and the prompt's dense matrix products use plain hipBLAS (slower prompts, same answers).
  A table's solution ids are valid only for that pair, and the engine refuses any other table.
- **Several cards:** setup takes one card (the one with the most VRAM, or `--gpu N`) unless you name more:
  `./setup.sh --backend hip --gpus 1,0` splits the model's layers across them, the first one the main card (numbers
  as setup lists them; `--gpus all` = every supported card, the most VRAM first). Every chosen card must be one of the
  architectures above; the engine is compiled for each of them (cards of two families, e.g. gfx1100 + gfx1201, need
  a system ROCm 7: AMD's wheels hold one family). A split pays only when no single card holds the model's experts
  (see RDNA4 below).
- **Limits for now:** no images, no calibration. The Monitor shows no GPU statistics.

The rest of setup is the same as on NVIDIA: the model download, the start script, the server.

## Build

Requirements: a working ROCm driver/runtime, HIP development headers and
compiler, hipBLAS and (for tuned dense prefill) hipBLASLt development files, CMake 3.24+, a C++20 host compiler, and Git.
The model still needs sufficient system RAM and fast SSD storage for its PLE
table. VRAM occupancy alone is not a throughput measurement.

```sh
cmake -S . -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip --target strata -j2
```

`CMAKE_HIP_ARCHITECTURES` is `gfx1100`, `gfx1101`, `gfx1200`, `gfx1201`, or a list such as `"gfx1100;gfx1201"`
(one binary for both). gfx1102 (the same wave32, 64 KiB LDS and dot4 instruction) builds with a warning: it passed
ctest (#192) but no model run has been reported. At startup the engine and `strata-device` compare each GPU they use
(`gcnArchName` up to the `:` feature suffix) with the architectures the binary was compiled for, and require
wave32. A binary carried to another card stops with the card's name, its architecture and the build's list,
instead of failing later with "invalid device function".

If CMake cannot find the HIP compiler, add
`-DCMAKE_HIP_COMPILER=/path/to/rocm/llvm/bin/clang++`. On the Fedora-family test
host this was `/usr/lib64/rocm/llvm/bin/clang++`. Use the compiler and libraries
from the same ROCm installation. Do not enable both GPU backends in one build.

Native IQ experts are enabled by default. CMake fetches the llama.cpp revision
pinned by this repository. For an offline build, point `STRATA_GGML_DIR` at a
checkout of that exact revision; using an arbitrary newer checkout changes the
dependency being tested.

## Model and serving configuration

Prepare a supported model pack and its MTP runtime using the existing tools.
For OrcaRouter IQ3_XXS, follow [the explicit compatibility conversion](ORCA.md);
this backend does not change quantization, model licenses, or tokenizers.
Add `--experts-bin` to the `tools/iq_pack.py` command: mmap requires the pack's
`experts.bin`, which the default native packing command does not emit. This
consumes additional disk space. Original GSQ-RCO IQ3_XXS uses shard 2 for PLE;
Orca uses shard 1. Keep each model's own pack and tokenizer together.

Use `build-hip/strata` as the executable in the server JSON. Select the AMD
device with `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` if necessary. Start with
a modest context and prefill chunk size before measuring larger workloads.
Keep the PLE file on an SSD. Do not assume a configured context length proves
successful full-window inference.

Large pinned host allocations can fail on ROCm even when ordinary RAM is
available. The existing `--mmap-experts` path avoids allocating the full pinned
expert arena; it still depends on OS file-cache residency and may stall on
storage reads. It does not make SSD access equivalent to RAM.

Starting args for the original GSQ-RCO IQ3_XXS model (replace the paths):

```sh
build-hip/strata --serve \
  --pack packs/iq3xxs \
  --native /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --ple-gguf /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
  --mmap-experts --expert-profile data/expert-profile.bin --expert-cache auto \
  --prefill 512 --spec 4 --spec-min-p 0.5 --mtp mtp/rt \
  --max-context 4096 --kv int8 --pool-workers 15 \
  --adapt-every 0 --pcie-frac 0 --vram-reserve-mib 1024
```

`--serve` is the engine's internal token protocol. For a browser or OpenAI API,
put these args in the `args` array of the [server JSON example](ORCA.md#local-server),
set `exe` to `build-hip/strata`, and use the matching pack's `tokenizer` directory.
Launch with `.venv/bin/python -m serve.server --engine strata --config strata-hip.json --port 8080`.
The worker count above was used on a 16-core CPU; measure it for your CPU.
The 4K context is a smoke-test starting point, not a model limit. The expert cache
sizes itself automatically and leaves 1 GiB of VRAM headroom.

The installer supports this backend (see "Install with setup" above). The vision helper is NVIDIA-only for now.
Setup installs one AMD card, or several with `--gpus` (the engine's layer split; see RDNA4 below).

## RDNA4 (gfx1201)

The RX 9070 / 9070 XT and the Radeon AI PRO R9700 run the same kernels as gfx1100: wave32, 64 KiB of LDS per
workgroup, the signed dot4 instruction (`v_dot4_i32_iu8` through `__builtin_amdgcn_sudot4`) and a 100 MHz wall
clock. The first report and patch came from doplxyz (#178). Validated on 2026-09-30 on doplxyz's test machine:
an RX 9070 XT 16 GB and a Radeon AI PRO R9700 32 GB (both gfx1201), a Ryzen 9 3900X (16 threads, no AVX-512),
47 GB RAM, Ubuntu 24.04 in a KVM/VFIO guest.

- **Build:** complete HIP build with tests (`-DCMAKE_HIP_ARCHITECTURES=gfx1201 -DSTRATA_BUILD_TESTS=ON
  -DSTRATA_PREFILL_MMQ=ON`), with the system ROCm 7.14 (hipBLASLt 1.4.1) and with setup's own path: TheRock
  7.10.0a20251120 wheels from `gfx120X-all` (hipBLASLt 1.2.0) and `build_engine_hip`.
- **ctest** (all 45 registered tests, R9700): 42 pass, `hip_prefill_hipblaslt_gemm` skips (no table), and two
  fail for reasons outside the GPU: `ple_parity` needs the Q2_0 model fixture, `expert_multi_test` needs an
  AVX-512 CPU. `hip_handoff` needed the volatile ring store: on gfx1201 a plain store to mapped pinned memory
  stays in the GPU's L2 until the stream is synchronized.
- **Arch check:** a gfx1100-only build stops on the gfx1201 card at startup with the message above (engine and
  `strata-device`).
- **End to end** (Coder IQ1_M, setup's arguments: `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5`,
  MTP, `--kv int8`, `--max-context 32768`; 128 greedy tokens, system ROCm 7.14, no hipBLASLt table). The answers
  are coherent and the same on both cards. The live server smoke (`serve/server.py`: web page, models, props,
  chat, prefix reuse, streaming, the Anthropic endpoint, a sampled code answer) passed 9/9 on each card.

  | card | expert cache | peak VRAM | 4K prompt | 4K decode | 16K prompt | 16K decode |
  |---|---|---|---|---|---|---|
  | Radeon AI PRO R9700 32 GB | 12,288 slots, 23.4 GiB | 29.7 GiB | 982 tok/s | 45.5 tok/s | 1,402 tok/s | 48.3 tok/s |
  | RX 9070 XT 16 GB | 4,931 slots, 9.4 GiB | 15.6 GiB | 782 tok/s | 30.8 tok/s | 1,235 tok/s | 35.4 tok/s |
  | R9700, TheRock 7.10 wheels | 12,288 slots | | 957 tok/s | 44.2 tok/s | 1,284 tok/s | 43.9 tok/s |

  The engine's resident memory was about 26 GB in every run. Since 0.1.31 `__byte_perm` is one `v_perm_b32` and the
  packed byte subtracts/compare work on four lanes at once (#262, ttio2tech): decode +15% on the R9700 (46.0 -> 53.0
  tok/s on a 4K prompt, 52.0 -> 60.5 warm) and +5-7% on the 9070 XT, prompts unchanged, the same tokens.
- **hipBLASLt:** there is no gfx1201 table in `tools/hip`. A table calibrated on the R9700 at the engine's shapes
  (hipBLASLt 1.4.1; 0.98-1.76x per GEMM over hipBLAS) changed the end-to-end prompt speed by 0-3%, within noise,
  so none is shipped: on gfx1201 the plain hipBLAS path is already close.
- **Both cards in one run (layer split, engine 0.1.30):** the config's `"backend": "hip", "gpu": [1, 0]` (R9700
  first) runs through `serve/server.py` (setup writes it with `--gpus 1,0` since 0.1.31). Auto split put layers 0-27 on the R9700 and
  28-47 on the 9070 XT. With every expert on the GPUs the split gives exactly the tokens of the R9700 alone (4K and
  16K prompts); checkpoints on the split (second turn, rewind, a prompt sharing a prefix, a cancelled prompt
  retried) give exactly the tokens of a fresh read. The conversation cache refuses a split at start (exit 2).
  On this pair the split does not pay: the R9700 alone already holds all 12,288 expert pairs.

  | run (the same session, warm) | 4K prompt | 16K prompt | decode |
  |---|---|---|---|
  | R9700 alone | 1,794 tok/s | 1,804 tok/s | 51-52 tok/s |
  | R9700 + 9070 XT, auto split | 1,384 tok/s | 1,906 tok/s | 42-43 tok/s |
  | RX 9070 XT alone | 1,016 tok/s | 1,519 tok/s | 38-40 tok/s |

  A split is worth it when no single card holds the model's experts. Since 0.1.31 a split pins the whole expert
  arena on Linux (the 8 GiB cap is for Windows/WSL2 only, #253): with an expert cache of 1,500 on the R9700 (most
  experts streamed) the split read a 16K prompt at 1,803 tok/s instead of 1,287, with the same tokens in 5 + 5
  starts. `"split_skip_if_fits": true` in the config (0.1.31, opt-in) runs such a pair on the first card alone when it
  holds every profiled expert: with the R9700 first, 4K prompts 1,776 tok/s (split: 1,244) and decode ~60 tok/s
  (split: ~51), the tokens of the R9700 alone (docs/MULTI_GPU.md).
- **Known:** rarely (about 1 start in 10) a HIP run's greedy output differs from another start's at some token, on
  one card or two and on engine 0.1.29 as well; not yet explained.
- **Not validated:** images, long contexts beyond 16K, answer-quality benchmarks.

## Community-validated cards

Run by their owners, not on the maintainers' machines; setup accepts them like gfx1100 / gfx1201. No hipBLASLt table
is shipped for them (make one with [Tuning table](#tuning-table) and compare the prompt speed with and without it).

- **gfx1101, RX 7800 XT 16 GB** (jhohertz, #254; engine 0.1.29, Ryzen 9 5950X, 121 GiB RAM, system ROCm with
  hipBLASLt 1.4.1): `./setup.sh --backend hip` detected the card and compiled the engine; `strata-device --selftest`
  passed; ctest 30/32 (`ple_parity` needs the Q2_0 fixture, `platform_memory_test` the memlock limit). Coder IQ1_M,
  64K context, MTP, with a table the owner calibrated: fresh prompts of 4K-9K tokens at 898-953 tok/s, decode
  38-44 tok/s (128 tokens).
- **gfx1200, RX 9060 XT 16 GB** (Efeisot, #256 after #176; engine 0.1.29, Ryzen 9 7950X, 64 GiB RAM, ROCm 7.2 with
  hipBLASLt 1.2.2): ctest 32/32 (without `ple_parity` and `platform_memory_test`). Coder IQ1_M, greedy, MTP
  `--spec 4 --spec-min-p 0.5`, with a table the owner calibrated:

  | prompt | prompt speed | decode |
  |---|---|---|
  | 2,374 tokens | 540 tok/s | 27.1 tok/s (0.69 draft acceptance; 31.0 at 1.00) |
  | 65K (`--kv int8 --kv-resident 65536 --prefill 16384`) | 748-753 tok/s | 26-40 tok/s by acceptance |
  | 130K (the same flags) | 725 tok/s | - |

  Greedy output was the same across runs. For comparison, llama.cpp's HIP build measured 20 tok/s decode and
  450 tok/s prompt on that card.

## Tuning table

A hipBLASLt table holds solution ids that are valid only for one GPU architecture and one hipBLASLt version, so it
is calibrated on the card with the ROCm the engine runs with. The shipped gfx1100 table's rows are the engine's
dense GEMM shapes; to calibrate them for another card or version (a few minutes):

```sh
cmake --build build-hip --target tune_hipblaslt
CASES=$(awk 'NR>2 {printf " --case %s,%s,%s,%s,%s", $1, $5, $2, $3, $4}' tools/hip/gfx1100-hipblaslt-100200.txt)
./build-hip/tune_hipblaslt $CASES --tuning-out table.txt
```

The file's second line names the architecture and version (`STRATA_HIPBLASLT_TUNING_V1 gfx1201 100401`); save it
as `tools/hip/<arch>-hipblaslt-<version>.txt` for setup, or point `STRATA_HIPBLASLT_TUNING` at it. Compare the
prompt speed with and without it before keeping it.

## RDNA3 WMMA kernels for the gfx1100 prefill (opt-in)

gfx11 has matrix instructions (`v_wmma_*`) for fp16 and bf16 that this backend was not using: dense projections
reached BLAS, and the prompt attention used the ordered FP32 fallback written for CUDA-only QSA matrix
instructions.  Two native paths are available behind **opt-in** switches, so default output stays exactly what
hipBLAS produces today:

* `STRATA_WMMA_GEMM=1` - dense GEMM over the RDNA3 WMMA fragment layout (16x16x16, wave32 "doubled" inputs),
  templated for `_Float16` and `__bfloat16`, dispatched from `Gemm::f16` and `Gemm::bf16` ahead of the BLAS
  fallback.  `STRATA_WMMA_BF16=0` excludes just the bf16 path.
* `STRATA_PA_WMMA=1` - the prompt attention itself on WMMA (scores 16x32 in two waves, output four warps over 64
  dimensions) instead of the emulated scalar path.

Both paths gate themselves at **runtime** on the device's `gcnArchName` (gfx11 prefix): a gfx1201, a mixed-arch
build or any other device never executes them, whatever was compiled.  The build only supplies the
`STRATA_WMMA_GFX11` compile definition for gfx11 configurations (see `cmake/hip_backend.cmake`), because the host
pass of a HIP compile does not define `__gfx1100__` and a compiler-macro guard would silently build the
returning-false stubs.

Measured on one RX 7900 XTX with a 6-core host over PCIe 4.0 x16 (engine 0.1.30, the documented measured
configuration, two repetitions per cell, interleaved):

| tier | switch state | prefill (tok/s) | decode (tok/s) |
| --- | --- | ---: | ---: |
| 1K | off | 376 / 455 | 35.2 / 45.4 |
| 1K | GEMM | 471 / 551 | 39.7 / 51.5 |
| 1K | GEMM + PA | 619 / 471 | 50.1 / 50.3 |
| 32K | off | 790 / 800 | 57.3 / 55.8 |
| 32K | GEMM + PA | **1501 / 1484** | 55.1 / 57.1 |

Prefill is the clear win and grows with context (+31 % at the 1K median, +88 % at 32K); decode is unchanged within
this host's variance.

`tests/hip/prefill_wmma_gemm_parity.cpp` (ctest-registered like the neighbouring `hip_*` tests) covers both entry
points against a double-precision host reference over 440 shapes x 4 beta/ldy configurations x 2 dtypes, including
partial tiles, padding columns, end guards and the declined shapes.  Numerical note: the fp16 comparison allows
4 ULP because the matrix core accumulates in fp32 but rounds toward zero while the reference rounds to nearest -
every delta observed is <= 1 ULP and toward zero, bf16 compares exactly, and with the switches off the engine's
output is bit-identical to the hipBLAS path by construction.  `STRATA_WMMA_PARITY_EXACT=1` demands bit equality.

## Original backend validation (PR #94)

The following is historical validation of the original backend, not a fresh
test count for this replacement. Current build/test evidence and performance
limits are recorded in [AMD_HIP_PERFORMANCE.md](AMD_HIP_PERFORMANCE.md).

Tested against upstream `c1e903310f211e6630780c3bd2038778c071c68d` (0.1.20),
with pinned llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
The host has an RX 7900 XTX (24 GiB), Ryzen 9 7950X3D, 64 GiB system RAM,
and Fedora-family Linux with ROCm 7.1.52802 / Clang 20. The original model used
for the smoke check was on mechanical RAID0; cold page faults were slow. Keep
latency-sensitive data on SSD and measure warm residency separately.

- Complete HIP Release build, including the executable and parity targets.
- All 25 tests selected by the command below passed on 2026-09-29.
- Intrinsic parity includes signed packed-byte dot products and overflow.
- CPU/GPU mapped-memory handoff covers delayed publication, changing payloads,
  and repeated graph replay in both directions.
- GR coverage includes the maximum eight-token fused batch and graph replay
  after inputs change. Native QSA checks the scalar HIP fallback.
- Q8_K quantization is byte-exact against the existing reference. HIP's
  `__fmul_rn` can become ordinary multiplication; disabling contraction in
  `quantize_act.cu` preserves the separately rounded product before the magic
  rounding bias is added. CUDA's compile flags remain unchanged.
- The mmap regression covers canonical and differing native layer sizes,
  truncated/oversized files, lookup bounds, and reopening after errors.
- CUDA 13 / GCC 15 / sm_89 executable build passed; GR, sampler, quantization,
  and mmap tests passed on the same tree. This is a regression check, not a
  claim of complete CUDA inference validation.
- CPU-only `strata-plan` build passed without either GPU backend.
- Real GSQ-RCO IQ3_XXS server smoke with mmap, the shipped expert profile,
  automatic GPU cache, and MTP passed arithmetic, executable Python addition,
  system-marker recall, and 1,170-token batched-prefill recall. All four requests
  ended normally with correct answers; one engine process served them all.
  The cache held 11,142 experts (18.08 GiB), with 874 MiB of VRAM free after
  startup. These short checks do not establish a throughput or quality benchmark.

Build all targets before running the registered focused checks:

```sh
cmake --build build-hip -j2
ctest --test-dir build-hip --output-on-failure --timeout 60 \
  -E '^(ple_parity|platform_memory_test)$'
```

`ple_parity` requires an external model fixture. `platform_memory_test` requests
256 MiB of locked memory; the test shell's 8 MiB memlock limit prevented that
test from passing. These are explicit exclusions, not skipped tests counted as
passes. Additional unpublished upstream fixture suites are not covered.

The published performance table explicitly identifies the measured development
snapshot; it must not be read as a benchmark of every subsequent rebase. Vision, long-context stress, broad answer-quality
equivalence, other AMD cards, and mixed-vendor inference are not validated here.
