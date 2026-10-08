# Jetson AGX Orin 32 GB implementation plan

Target: Linux ARM64, Jetson AGX Orin 32 GB, JetPack 6.2, CUDA 12.6.
Branch: `feat/jetson-orin-arm64`. The engine builds and runs the pinned Coder
IQ1_M model on this device. The five phases below have been implemented and
validated for the tested Coder IQ1_M configuration; limits are recorded in the
hardware report. The initial `[N/A]` workaround in setup
has been replaced with CUDA driver capability discovery.

On Jetson, CPU and GPU allocations consume the same physical RAM. Unified
addressing, shared physical RAM, and CUDA managed allocations are distinct
properties. Do not replace every allocation with `cudaMallocManaged`.
[NVIDIA's CUDA for Tegra guide](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-for-tegra-appnote/index.html)
describes the memory access and coherency rules to use for this port.

## Phase 1: platform facts and small portability changes — implemented

- Added a standalone CUDA capability probe which builds without the x86 CPU
  kernels, ggml downloads, or model files. It queries memory and capabilities;
  it does not allocate a model or an expert cache.
- Added an ARM `yield` / x86 `pause` abstraction and used it in the CPU pool.
  This is a spin hint, not a replacement for atomic memory ordering.
- Fixed setup CPU detection when `/proc/cpuinfo` has ARM `Features` rather than
  x86 `flags`; added ARM and x86 regression cases.
- Runtime `DeviceInfo` exposes integrated, managed, concurrent managed and
  mapped-host capabilities. Setup uses the CUDA driver API on ARM64; the probe
  also supports `--json`.
- Exit check: setup and runtime agree on ARM64, sm_87, integrated memory,
  supported registration/mapping, and available physical memory without
  inventing a dedicated VRAM size. Check error handling with no usable GPU.

Build and run the diagnostic alone:

```sh
cmake -S . -B build-jetson-probe \
  -DSTRATA_BUILD_CUDA_PROBE_ONLY=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12
cmake --build build-jetson-probe -j2
./build-jetson-probe/strata-cuda-probe
python tools/test_setup_jetson.py
```

Measured here on 2026-10-07: Linux aarch64, NVIDIA Jetson AGX Orin Developer Kit,
GCC 11.4, nvcc 12.6.68; runtime and driver report 12060. The probe reports:

| Property | Value |
| --- | --- |
| GPU / compute architecture | Orin / sm_87 |
| Host physical and CUDA total bytes | 32,187,883,520 each |
| CUDA free bytes at probe time | 2,216,710,144 |
| Integrated / unified addressing | 1 / 1 |
| Managed / concurrent managed access | 1 / 0 |
| Pageable access | 0 |
| Mapped host / host registration / VMM support | 1 / 1 / 1 |

These are capability and memory observations, not inference benchmarks.
The diagnostic and full CUDA engine builds pass with `CMAKE_CUDA_ARCHITECTURES=87`.
Setup Jetson tests pass (15 cases), older CPU tests pass (six cases with one
Windows-only skip), and older GPU tests pass (19 cases).

## Hardware validation completed (2026-10-08)

The user stopped the separate ninfer service. Real MTP inference now passes
repeated greedy requests, OpenAI/Anthropic streaming and cancellation recovery.
Nine requests with 64 generated tokens each measured median decode throughput
of 18.6–21.6 tokens/s on this Orin; prompt throughput was 37.8–52.5 tokens/s
for 172–1069 prompt tokens. Prompts of 3105, 7234 and 31234 tokens passed at
context limits 4096, 8192
and 32768, followed by a successful short recovery request.
These numbers apply to the pinned Coder IQ1_M pack and this configuration.

Real requests exposed two additional issues: a one-token native prompt skipped
execution because its last position was also the sentinel, and late workspaces
could consume the physical headroom left by auto cache sizing. The sentinel is
now -1. Integrated-device auto sizing leaves an additional 3072 MiB workspace
reserve; setup applies the same default with vision, and startup checks fresh
available physical RAM after initialization. Explicit reserves remain honored.
The corrected 4K run retained at least 7.43 GiB available RAM; the 32K run
retained at least 7.33 GiB. Swap was in use.

[Commands and raw results](../bench/results/2026-10-08-jetson-orin/README.md)
include actual CPU reference comparisons, finite CPU/CUDA image embeddings,
repeated image API requests, and checksum-checked memory-path measurements.
Formatted chat prompts selected the CPU reference's top token with small
selected-token log-probability differences; raw-token inputs had larger
differences, so universal numerical parity is not established. High-resolution
vision, other model quantizations, sustained thermal throughput and desktop GPU
runtime validation remain outside the measured scope. The checkpoint
below records the earlier state before the separate service was stopped.

## Implementation checkpoint (2026-10-07)

- Phase 2: ARM generic expert operations and ggml's ARM backend compile. Actual
  ARM multi-token, worker task, affinity, router and 20-second pool stress tests
  pass. Synthetic CUDA parity passes for q4_0/q4_0, q4_K/q5_1 and q8_0/q8_0;
  The K/Q native quantization suite passes 11 cases, and four additional IQ
  expert pairs pass; Q2_0 GGUF rows agree with
  an independent FP64 reference for 1–8 tokens and 1/3/10/40 blocks. These tests
  do not establish correctness for every model quantization. Real Coder weights
  also pass parity for expert 7 in each of all 48 layers. The optional Q8_0
  activation contract matches pinned ARM ggml for 5120 blocks after correcting
  its reciprocal arithmetic order.
- Phase 3: integrated-device cache sizing uses OS/cgroup available physical RAM
  with a CUDA allocation ceiling and six GiB default headroom. Ten deterministic
  budget cases pass. Managed sequential ownership, registered host mapping,
  512 CPU/GPU mapped handshake rounds and a two MiB VMM allocation pass on Orin.
  Mapped publication uses ARM system barriers. The actual expert cache passes
  VMM shrink/grow, address stability, data retention, cudaMalloc and excessive
  headroom refusal tests. Injected whole-arena and sliced host registration
  failures recover through pageable storage, clear CUDA errors and preserve
  copied data. File-source/cgroup and VMM range tests pass; resident expert
  rotation retains exact bytes with pinned and ordinary storage. Startup
  refuses exhausted shared budgets before loading model weights, and checks
  canonical and native dense allocation sizes against fresh budgets. Repeated
  real model requests remain outstanding.
- Phase 4: ARM setup selects local builds, retains JetPack libraries, rejects
  CUDA 13 on this Jetson, records CPU architecture in local build metadata and
  caps its initial context recommendation at 32768. ARM Linux build parallelism
  uses available RAM, leaving two GiB aside and allowing two GiB per compiler
  job as conservative heuristics; exhausted or unknown availability selects
  one job. Desktop build parallelism is preserved. Native ARM CPU and CUDA
  vision encoders build and reach argument parsing; image inference still
  needs validation. The CUDA encoder has 143 embedded SM87 images. Desktop
  setup fixtures pass 110 regression cases, and the x86 CPU kernels and router
  cross-build with GCC 11.4. Local ARM64 engine and GPU-capable vision artifacts
  are installed in `engine-cuda12/`, with the CPU-only encoder preserved.
  Setup recognizes their source and CPU architecture metadata without
  rebuilding, and upgrades CPU-only vision when GPU vision is requested.
  All pinned Python 3.12 requirements have compatible ARM64 wheels, install
  into the local environment, and pass `pip check`. Mock-engine
  API/security/cancellation tests pass 290 cases on ARM. Actual
  model/API integration remains outstanding.
- Phase 5: no inference throughput or supported model claims yet. No compatible
  Flash-Next pack was present initially; both pinned Coder IQ1_M shards are now
  downloaded and SHA256-verified in `/home/calvin/models/strata-orin-validation/IQ1_M`.
  A 23.42 GiB file-backed expert pack is prepared (1.37 GiB canonical dense
  arena); all 31 MTP tensors are downloaded and SHA256-verified. MTP packing
  and runtime-file conversion are complete. All 512 prepared MTP experts pass
  ARM CPU parity against the scalar INT8 activation oracle. MTP inference
  validation remains outstanding. Actual startup correctly refuses the current
  insufficient memory budget while the existing server runs. The initial probe observed only
  about two GiB CUDA free memory. Read-only nvmap diagnostics identify the
  existing `ninfer-qwen38-27b.service` as owning 26,603,488 KiB (about 25.4 GiB)
  of GPU/shared allocations. Temporarily stopping this separate server requires
  user approval; model validation will need that memory released.

Reproduce the full engine build:

```sh
cmake -S . -B build-orin -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc
cmake --build build-orin --target strata -j2
```

Enable `STRATA_CUDA_MEMORY_SELFTEST=ON` in the standalone probe build to run
`strata-cuda-memory-test`. `STRATA_UMA_HEADROOM_GIB` overrides runtime shared
memory headroom; it accepts an integer from zero through 1024.

## Phase 2: ARM64 CPU correctness and engine build — implemented

- Split generic expert operations from x86 implementations in
  `src/kernels/cpu/expert.cpp`; preserve Q2_0 scalar quantization and dot
  contracts before introducing NEON optimizations.
- Guard CPU probes in `expert_layout.cpp`: ARM must return false for AVX
  capabilities without executing CPUID or XGETBV. Give ARM topology and CPU
  naming a Linux path independent of Intel hybrid-core assumptions.
- In `CMakeLists.txt`, select CPU source files and compiler flags by target
  architecture. Omit AVX files and AVX-VNNI compile checks on ARM; provide
  generic implementations for any symbols still referenced by the pool.
  Reject x86-only portable/ISA-floor options on incompatible targets.
- Use pinned ggml's ARM CPU backend for native IQ/K-quant experts. Audit
  dispatches in `pool.cpp` and `native_expert.cpp` so no x86-only symbols or
  activation layouts are assumed on ARM.
- Audit remaining intrinsics, prefetches, alignment, SIMD rounding, and platform
  helpers across the engine and vision build. Compile CUDA for sm_87 with the
  installed CUDA 12.6; preserve desktop builds.
- Exit check: build the full engine on this device; synthetic expert parity,
  multi-token expert parity, affinity and pool stress tests pass. Run independent
  CPU references with tolerances justified by quantization, not guessed changes.

## Phase 3: one physical memory budget — implemented

- Trace startup planning, `DeviceArena`, `ExpertCache`, `ExpertSource`, pinned
  arenas, KV storage, dense weights, prompt scratch, and vision reserves.
  Extend existing integrated-device handling rather than adding a second policy.
- Derive one budget from OS available physical RAM, cgroup constraints, CUDA
  allocation limits and explicit headroom. Never add reported GPU capacity to
  host RAM. Account for allocations already made when taking a new snapshot.
- Bound GPU expert cache and host complement together. Start with existing
  `cudaMalloc` plus bounded pinned staging and file-backed experts; make
  duplicate host/GPU expert copies visible in the accounting.
- Retain mapped host memory only where capability and synchronization contracts
  support it. Audit doorbell publication and completion ordering on ARM and
  Orin before enabling persistent CPU/GPU concurrent paths. Managed buffers
  require a separate policy for devices with `concurrentManagedAccess=0`.
- Keep VMM capability checks (this Orin reports support). Test actual reserve,
  map, resize and fallback behavior before relying on segmented cache growth.
- Exit check: deterministic budget tests cover 32 GB shared RAM, little free
  memory, cgroups, unknown values, and allocation failures. Model startup and
  repeated requests stay within headroom; record peak RAM, not only CUDA free
  bytes. Check both file-backed loading and pinned allocation fallback.

## Phase 4: setup, packaging and server integration — implemented

- Replace the `[N/A]` workaround with CUDA discovery and explicit shared-memory
  fields. Do not classify every missing VRAM value as Jetson.
- Select source builds on ARM64 until matching ARM release artifacts exist;
  never download or run an x86 engine on ARM. Preserve JetPack's toolkit and
  driver; avoid desktop x86 CUDA repository installation and wheel assumptions.
- Audit `setup.sh`, Python dependencies, CUDA library search paths, engine
  metadata, updates, and optional vision dependencies for ARM64.
- Choose model, context, worker count, cache and RAM complement using the shared
  budget. A file-backed low-RAM configuration will be needed when experts do
  not fit; installer disk requirements still apply.
- Exit check: mocked Jetson installation tests require no downloads; then
  perform a local installation using existing artifacts where possible and
  verify OpenAI/Anthropic API streaming on `127.0.0.1`.

## Phase 5: hardware validation and measured defaults — tested configuration validated

- Start with a short context and file-backed experts, then test increasing
  contexts, repeated requests, cancellation, low-memory fallback, and optional
  vision. Compare token outputs and log probabilities against a reference.
- Record model/quantization, engine revision, JetPack/toolkit versions, power
  mode, storage, context, workers, cache sizes, peak RAM, disk traffic and
  thermal behavior. Measure prompt and decode separately over repeated runs.
- Only after correctness, compare CPU scalar/ggml/NEON paths, mapped expert
  reads versus copies, and optional managed allocations. Keep optimizations
  opt-in until the measurements justify defaults.
- Exit check: publish reproducible commands and results in `bench/results/`,
  document supported configurations and limits in `docs/DETAILS.md`, and run
  desktop setup/build regression checks before proposing the branch upstream.

## Upstream 0.1.41 integration

The port was rebased onto upstream `fb58e0d` after synchronizing the fork.
The full SM87 CUDA build, 102 eligible CTest cases across the initial run
and targeted rerun, 48-layer native Coder checks, seven real API checks,
15 Jetson setup cases and 111 desktop setup cases passed. The x86 CPU/router
cross-build passed. Full Release HIP gfx1100 and SYCL SPIR-V builds subsequently
passed on `maestro1`, an x86-64 Ubuntu 22.04 host with a Celeron N3450;
[commands and logs](../bench/results/2026-10-08-maestro1-builds/README.md).
HIP/SYCL GPU tests, model execution and desktop GPU runtime remain untested.
The [updated report](../bench/results/2026-10-08-jetson-orin/README.md#rebase-onto-upstream-0141)
records fixture exclusions, ARM test-reference corrections and the parametric
1K–256K context comparison against the preserved 0.1.40.3 engine.

All nine tested capacities, 1K through 256K, passed with the six GiB physical
headroom preserved. The 256K single long-prompt run processed 261,936 tokens,
generated 64 tokens and passed recovery; its prefill took 74.7 minutes.
The same-day preserved/rebased comparison includes 18 cells and 174 requests.
Automatic cache sizes differ across capacities and builds; this measures the
complete configuration and does not isolate kernel speed. The 32K starter
recommendation remains appropriate for short-request throughput.
