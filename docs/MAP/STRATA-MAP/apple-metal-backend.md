# Strata map: Apple Metal backend (`metal/`)

Source: Strata commit `296b59d2`. Paths are relative to the Strata repo root. `metal/` holds 20,142 lines.

## 1. Overview

- `metal/` is Strata’s **macOS-only Apple Silicon backend**: a parallel CMake project that never joins the CUDA/HIP/SYCL builds.
- It has two layers: (A) **mac-kern** — runtime face `metal/src/mtl_rt.mm` + MSL kernels in `metal/kernels/` (22 `.metal` files, compiled at run time via `newLibraryWithSource`) plus parity/bench tests; (B) **mac-design** — shared host sources from `src/` compiled **unchanged** against a CUDA-name shim (`metal/include/metal_compat/`, analogous to `include/strata/hip_compat/` on HIP) over a Metal device runtime (`metal/src/gpu/runtime.cpp` / metal-cpp), with decode/prompt executed by the Metal-native `Window` engine (`metal/src/engine/window.cpp`) and optional **SSD-streaming expert arena** (`ExpertStream` in `metal/src/engine/stream.cpp`) when the GGUF does not fit resident RAM. Ops implemented in MSL: quantized unit decode (Q/K/IQ/Q2_0/BF16/F32 — **not NVFP4**), GEMV, MoE (plan/route/gu/down/combine + smap arena), dense GEMM (MPP), GDN decode+prompt, QSA decode+prompt, gated-residual/HC, router fuse, sample argmax, PLE/MTP helpers, glue.
- **See also:** engine-core.md (shared host), gpu-cpu-kernels.md (CUDA/HIP kernels it replaces), moe-experts.md (expert cache vs Metal stream), server.md (`src/core/verify.cpp` → `window_*`), intel-sycl-backend.md (sibling alt backend), build-system.md (`metal/CMakeLists.txt`), tests-bench-docs.md (`metal/tests/`), model-spec-ngram-platform.md (GGUF layout / Qwen3.5 geometry).

## 2. File table

| path | lines | purpose |
|---|---|---|
| metal/CMakeLists.txt | 123 | macOS-only CMake: `mtl_rt`, tests, optional `strata_gpu_metal` + engine + `strata` exe |
| metal/setup_mac.py | 706 | Mac setup/measure; writes cfg env incl. `STRATA_METAL_STREAM` |
| metal/include/strata/gpu/runtime.hpp | 126 | CUDA-shaped Metal runtime API (`strata::gpu::*`) |
| metal/include/strata/gpu/metal.hpp | 44 | Kernel-lib extras: `pipeline()`, `dispatch()`, `Stats` |
| metal/include/metal_compat/cuda_runtime.h | 130 | CUDA name → `strata::gpu` (first on include path) |
| metal/include/metal_compat/immintrin.h | 5 | Stub so host includes that pull x86 intrinsics compile |
| metal/include/metal_compat/cpuid.h | 6 | Stub cpuid |
| metal/include/metal_compat/linux/mman.h | 12 | Stub mman |
| metal/src/mtl_rt.h | 127 | Standalone C++ Metal face for parity/benches (`mtl::Context`) |
| metal/src/mtl_rt.mm | 381 | ObjC++ impl: compile, alloc/wrap, `run`/`run_graph`/`run_graph_seg`/`run_icb` |
| metal/src/gpu/runtime.cpp | 640 | metal-cpp CUDA-compat runtime + graph capture + `host_view`/`purge_device_allocations` |
| metal/src/gguf_lite.h | 37 | Minimal GGUF reader for Metal engine/tools (header) |
| metal/src/gguf_lite.cpp | 117 | Minimal GGUF reader for Metal engine/tools (impl) |
| metal/src/ref_quant.h | 46 | CPU reference dequant for parity (header) |
| metal/src/ref_quant.cpp | 445 | CPU reference dequant for parity (impl) |
| metal/src/engine/window.cpp | 2616 | Decode/prompt Window: layer graph, stream cuts, MTP, prompt GEMM |
| metal/src/engine/prompt_chain.inc | 421 | Prompt-chunk path included by window |
| metal/src/engine/stream.h | 131 | `ExpertStream` SSD-streaming expert arena (decl + design note) |
| metal/src/engine/stream.cpp | 815 | `ExpertStream` arena fill, LRU, prefetch |
| metal/src/engine/arena_budget.h | 177 | Pure-arithmetic arena/footprint caps (testable off-Mac) |
| metal/src/engine/setup_host.cpp | 160 | Host-side path switches/scratch sizes copied from CUDA `.cu` files |
| metal/src/engine/iq_host.cpp | 90 | Host helpers for i-quant paths still referenced by shared code |
| metal/src/engine/mmvq_host.cpp | 66 | Host helpers for MMVQ sizing/flags |
| metal/src/engine/missing.cpp | 80 | `missing()` stub policy + crash/terminate backtraces |
| metal/src/engine/stubs.cpp | 367 | Generated stubs → `missing()` for unported kernel entry points |
| metal/src/engine/stubs_144.cpp | 53 | Extra stubs (NVFP4 CPU / Metal-disabled paths) |
| metal/src/engine/msl/ple.metal | 91 | PLE block MSL (gnorm/gate/conv_add) |
| metal/src/engine/msl/mtp.metal | 68 | MTP rms/xprep/combine |
| metal/src/engine/msl/mtp_chain.metal | 92 | MTP pick + embed on GPU for one-CB draft rounds |
| metal/kernels/quant.metal | 313 | Weight “units” / dequant tables (ggml formats; no NVFP4) |
| metal/kernels/units.metal | 321 | Unit×activation dots + xprep layout |
| metal/kernels/gemv.metal | 264 | Dense/shared/LM-head GEMV for 1..8-token windows |
| metal/kernels/moe.metal | 336 | MoE plan/route/gu/down/combine; arena via FC20 + smap |
| metal/kernels/moe_mm.metal | 1089 | Prompt MoE MM (hist/scan/place + mm_gu/down variants) |
| metal/kernels/dense_mm.metal | 69 | Prompt dense: act_f16, dq_f16, gemm_f16 (MPP) |
| metal/kernels/mma.metal | 216 | Matrix / cooperative helpers used by prompt/QSA |
| metal/kernels/gdn.metal | 170 | GDN decode recurrence (gdn_step / gdn_step8) |
| metal/kernels/gdn_prompt.metal | 485 | GDN prompt pre/scan/post |
| metal/kernels/qsa.metal | 384 | QSA decode (prep/score/topk/attn…) |
| metal/kernels/qsa2.metal | 289 | QSA variants (topk2 etc.) |
| metal/kernels/qsa_prompt.metal | 340 | QSA prompt attn/merge/mm |
| metal/kernels/gr.metal | 179 | Gated-residual / HC kernels |
| metal/kernels/gr2.metal | 341 | Two-dispatch HC read (BF16 or Q8_0) |
| metal/kernels/router.metal | 104 | Fused router GEMV + moe_route in one dispatch |
| metal/kernels/glue.metal | 63 | swiglu_prep, moe_combine_sh |
| metal/kernels/sample.metal | 59 | Greedy argmax_part/final |
| metal/kernels/prompt.metal | 177 | Prompt-chain glue kernels |
| metal/kernels/gridbar.metal | 54 | Grid-barrier microbench |
| metal/kernels/alu_bench.metal | 181 | ALU rate probes |
| metal/kernels/mm_bench.metal | 85 | Matmul bench kernels |
| metal/kernels/rt_bench.metal | 67 | Bandwidth/dispatch-cost probes |
| metal/probe/mtl_probe.cpp | 268 | Device capability probe |
| metal/probe/mtl_doorbell.cpp | 115 | Shared-event doorbell probe |
| metal/probe/mkfile.c | 23 | Tiny probe helper |
| metal/tools/gen_stubs.py | 184 | Regenerates `metal/src/engine/stubs.cpp` from headers |
| metal/tools/metal_repack.py | 243 | Weight repack for Metal packs |
| metal/tools/stream_experts.py | 102 | Builds contiguous `.experts` file for streaming I/O |
| metal/tools/gguf_list.cpp | 25 | List GGUF tensors |
| metal/tools/check_ref.py | 34 | Reference check helper |
| metal/tools/chat_tokens.py | 23 | Token helper |
| metal/tests/ (26 files besides runtime_hello.cpp) | 5,205 total | Parity vs ref_quant / benches; `metal/tests/arena_budget_test.cpp` is host-only |
| metal/tests/runtime_hello.cpp | 162 | Smoke test for strata_gpu_metal + graph |

## 3. Where it does X

### CUDA-name shim (mirror of HIP compat)

See also: build-system.md, engine-core.md
- **What:** Shared `src/` host files keep calling `cudaMalloc` etc.; Metal puts `metal/include/metal_compat` **first** on the include path (same idea as `include/strata/hip_compat` on HIP). Only Phase-1 inventory names exist; missing CUDA symbols fail to compile.
- **Where:** `metal/include/metal_compat/cuda_runtime.h:1-130` (forwards to `strata::gpu::*`); design note `metal/include/strata/gpu/runtime.hpp:1-14`; wiring `metal/CMakeLists.txt:56-61` (`BEFORE PUBLIC …/metal_compat`), host sources list `metal/CMakeLists.txt:90-100`.
- **How:** Types alias (`cudaStream_t` → `strata::gpu::Stream`); each inline wrapper calls the Metal runtime. Graph capture only accepts `cudaStreamCaptureModeThreadLocal` (`metal/include/metal_compat/cuda_runtime.h:114-116`). CUDA/HIP objects never include this directory.

### Metal device runtime (streams, shared buffers, graphs)

See also: engine-core.md, gpu-cpu-kernels.md
- **What:** Implements CUDA semantics on metal-cpp: device pointer = `MTLBuffer::gpuAddress`; all buffers Shared + one `MTLResidencySet`; `caps().unified_memory == true`; `host_view(device_ptr)` returns CPU `contents()`.
- **Where:** API `metal/include/strata/gpu/runtime.hpp:54-58` (`Caps`, `host_view`); impl `metal/src/gpu/runtime.cpp:278-296` (`new_alloc`), `metal/src/gpu/runtime.cpp:317-320` (`host_view`), `metal/src/gpu/runtime.cpp:327-340` (`purge_device_allocations`), `metal/src/gpu/runtime.cpp:534-567` (graph capture/launch), `metal/include/strata/gpu/metal.hpp:35-36` (`dispatch`).
- **How:** Streams open a command buffer + compute encoder; dispatches/blits ordered per stream; capture records `Op`s and re-encodes on `graph_launch`. Large unused host-uploaded weight copies can be purge-emptied (`purge_device_allocations`) so small Macs reclaim arena space — unless `STRATA_METAL_KEEP_HOST_WEIGHTS=1` (`metal/src/gpu/runtime.cpp:187-195`, `metal/src/engine/window.cpp:1132`).

### Standalone kernel runtime (`mtl::Context`)

See also: tests-bench-docs.md, gpu-cpu-kernels.md
- **What:** Thin C++ API used by parity/benches and by the engine Window (not the CUDA-compat path).
- **Where:** `metal/src/mtl_rt.h:72-122` (`Context`); compile `metal/src/mtl_rt.h:87`, pipeline `metal/src/mtl_rt.h:88`, `run_graph_seg` `metal/src/mtl_rt.h:104-105`; impl `metal/src/mtl_rt.mm:68` compile, `metal/src/mtl_rt.mm:249` `run_graph`, `metal/src/mtl_rt.mm:268-307` `run_graph_seg` (shared-event handshake between CB segments and host).
- **How:** MSL sources concatenated (+ optional ggml-common prelude), compiled at runtime. `run_graph_seg` encodes concurrent segments with signal/wait on a shared event so the host can stream experts between layer cuts without ending the command buffer.

### Fatal threadgroup-size check (silent Metal skip)

See also: gpu-cpu-kernels.md, tests-bench-docs.md
- **What:** Metal does **not** error when `maxTotalThreadsPerThreadgroup` < requested TG size; kernels are silently skipped (GDN on M2 Max gave garbage answers).
- **Where:** Documented `metal/src/mtl_rt.h:62-70`; enforced `metal/src/mtl_rt.mm:159-168` `require_fit`; engine calls at graph build `metal/src/engine/window.cpp:156-176` (`push` → `mtl::require_fit`).
- **How:** Every dispatch checks fit; `STRATA_METAL_MAX_THREADS_CAP` lowers reported max for tests (`metal/src/mtl_rt.mm:142`).

### Decode window (replaces CUDA kernel launches)

See also: server.md, engine-core.md, gpu-cpu-kernels.md, model-spec-ngram-platform.md
- **What:** Full Qwen3.5 decode path as one Metal dependency graph of mac-kern dispatches; optional per-layer stream cuts.
- **Where:** Class `Window` `metal/src/engine/window.cpp:115`; `layer` `metal/src/engine/window.cpp:657`; `run` `metal/src/engine/window.cpp:950`; finishes via `run_graph_seg` `metal/src/engine/window.cpp:1098`; C ABI for host `window_init` `metal/src/engine/window.cpp:2474`, `window_run` `metal/src/engine/window.cpp:2491`, `window_commit` `metal/src/engine/window.cpp:2508`.
- **How:** `Window::layer` appends GR → QSA/GDN → router/MoE → residuals for each layer; when streaming, cuts after router so `at_cut` → `ExpertStream::at_layer`. Shared program still owns session/geometry (engine-core.md); Metal never calls the CUDA `.cu` kernels for the hot path (`metal/src/engine/setup_host.cpp:1-9` says path switches do not steer Metal kernels).

### SSD-streaming expert arena (unified-memory expert handling)

See also: moe-experts.md, engine-core.md
- **What:** When experts do not fit, dense weights load into one resident buffer; experts live in an arena of slots filled by `F_NOCACHE` `pread`s; kernels remap via `smap` (function constant 20).
- **Where:** Design `metal/src/engine/stream.h:1-16`; `ExpertStream::wanted` `metal/src/engine/stream.cpp:33-48`; `make_arena` `metal/src/engine/stream.cpp:368`; `at_layer` `metal/src/engine/stream.cpp:614`; budget math `metal/src/engine/arena_budget.h:80-158`; kernel side `metal/kernels/moe.metal:46-50`, `metal/kernels/moe.metal:210-227` (`arena_slot` / `smap`); Window wiring `metal/src/engine/window.cpp:262`, `metal/src/engine/window.cpp:293`, `metal/src/engine/window.cpp:427-430`, `stream_arena` `metal/src/engine/window.cpp:1127`.
- **How:**
  - Auto-on if mapped model + 12 GiB overhead exceeds `resident_fits` (`metal/src/engine/arena_budget.h:137-142`), or `STRATA_METAL_STREAM=1`.
  - After each layer’s router, GPU signals; host fills missing experts (LRU victims), updates `smap`, prefetches next layer from predicted logits (`STRATA_METAL_PREFETCH` / `PREFETCH2`).
  - Contiguous `.experts` file (`metal/tools/stream_experts.py`) optional via `STRATA_METAL_CONTIG`.
  - Arena may be `mlock`’d (`STRATA_METAL_WIRE_ARENA`, default on ≥64 GB) to stop compressor thrash.
- **Relation to `ExpertCache`:** HIP/CUDA have `ExpertCache` with a **unified_memory** branch (`src/core/expert_cache.cpp:34-46`) and CUDA-only `--vram-elastic` segmented VMM (`src/program/generate.cpp:797`, parsed at `src/program/generate.cpp:2070`) — Metal’s stream is a separate Mac-specific design, not a port of ExpertCache.

### Arena / footprint budgeting

See also: moe-experts.md, build-system.md
- **What:** Caps so macOS does not compress/swap the arena (measured panics on M5 Max).
- **Where:** `metal/src/engine/arena_budget.h:36-47` (`ArenaMem`), `metal/src/engine/arena_budget.h:80-90` `footprint_cap`, `metal/src/engine/arena_budget.h:112-118` `arena_budget`, `metal/src/engine/arena_budget.h:146-158` shrink/grow; applied `metal/src/engine/window.cpp:1164-1194`; tested `metal/tests/arena_budget_test.cpp:1-2`.
- **How:** Keep `max(7GB, 15% RAM)` (or `STRATA_METAL_KEEP_FREE_GB`); big Mac (≥64 GB) leaves ≥40% or file-cache floor + 12 GB; stepped fill (`STRATA_METAL_ARENA_STEP_GB`, default 1536 MiB) stops when free would breach keep.

### Quant formats / GEMV / MoE ops

See also: gpu-cpu-kernels.md, model-spec-ngram-platform.md
- **What:** Unit-based decode shared by gemv/moe; formats Q4_0, Q5_0/1, Q8_0, Q3/4/5/6_K, IQ2/3/4_*, Q2_0/Q2_0R, BF16, F32. **No `T_NVFP4` / type 40** in `metal/kernels/quant.metal` (NVFP4 CPU entry points stubbed in `metal/src/engine/stubs_144.cpp:39-51`).
- **Where:** Types `metal/kernels/quant.metal:13-31`; unit dots `metal/kernels/units.metal`; window GEMV `metal/kernels/gemv.metal:1-26`, host dispatch `metal/src/engine/window.cpp:566`; MoE `metal/kernels/moe.metal:104`; prompt MoE `metal/kernels/moe_mm.metal`.
- **How:** Decode tables staged to threadgroup memory (Apple GPUs serialize divergent constant lookups). MoE kernels optionally bind up to 4 arena chunk buffers + `smap`.

### Attention / GDN / GR / sample / PLE / MTP

See also: gpu-cpu-kernels.md, prefill.md, model-spec-ngram-platform.md
- **GDN decode:** `metal/kernels/gdn.metal` (`gdn_step`, 512 threads/head; `gdn_step8` 1024); prompt `metal/kernels/gdn_prompt.metal` (pre/scan/post).
- **QSA:** decode `metal/kernels/qsa.metal` / `metal/kernels/qsa2.metal`; prompt `metal/kernels/qsa_prompt.metal` (flash/mm paths).
- **GR/HC:** `metal/kernels/gr.metal`, `metal/kernels/gr2.metal` (optional `STRATA_HC_Q8`); Window `gr_read` `metal/src/engine/window.cpp:610`.
- **Sample:** greedy only in `metal/kernels/sample.metal` (argmax_part/final).
- **PLE:** MSL `metal/src/engine/msl/ple.metal`; Window also has `ple_cpu` `metal/src/engine/window.cpp:905` for host-side PLE pieces.
- **MTP:** `metal/src/engine/msl/mtp.metal` + `metal/src/engine/msl/mtp_chain.metal`; load/draft `metal/src/engine/window.cpp:1489`, `metal/src/engine/window.cpp:1906`; env `STRATA_METAL_MTP_*`.
- **Prompt path:** `metal/src/engine/prompt_chain.inc` + `metal/kernels/dense_mm.metal` (MPP gemm); toggles `STRATA_METAL_PROMPT*` / `PROMPT_GEMM*`.

### Stubs for unported CUDA entry points

See also: gpu-cpu-kernels.md, engine-core.md
- **What:** Anything still reached from shared host that has no Metal body calls `missing(name)` → exit 86 (or log-once if `STRATA_METAL_STUBS=log`).
- **Where:** `metal/src/engine/missing.cpp:14-28`; generated `metal/src/engine/stubs.cpp:1-3`; regenerate `metal/tools/gen_stubs.py`.
- **How:** Links the engine so the binary builds; a wrong code path dies loudly instead of silent wrong outputs. Control vectors empty (`metal/src/engine/missing.cpp:33-41`).

### Build / setup

See also: build-system.md
- **What:** `cmake -S metal -B build-metal … -DMETAL_CPP_DIR=… [-DSTRATA_GGML_DIR=…]`; needs Apple + metal-cpp for engine; ggml from pinned llama.cpp commit.
- **Where:** `metal/CMakeLists.txt:1-12`, `metal/CMakeLists.txt:50-54` (early return without metal-cpp), `metal/CMakeLists.txt:73-88` ggml, `metal/CMakeLists.txt:122-123` `strata` executable; `metal/setup_mac.py:674` sets `STRATA_METAL_STREAM` from measurement.

## 4. Key types / structs and env / CLI

### Types (file:line)

| name | where | role |
|---|---|---|
| `strata::gpu::Stream` / `Event` / `Graph` / `Caps` | `metal/include/strata/gpu/runtime.hpp:26-57` | CUDA-shaped runtime handles; Caps always unified_memory |
| `strata::gpu::metal::Grid` / `Stats` | `metal/include/strata/gpu/metal.hpp:27-42` | Dispatch grid; dispatch/blit counters |
| `mtl::Buffer` / `Pipeline` / `Dispatch` / `Constants` / `Arg` | `metal/src/mtl_rt.h:17-60` | Standalone runtime objects |
| `mtl::Context` | `metal/src/mtl_rt.h:72` | Compile/run face |
| `strata::metal_engine::ExpertStream` | `metal/src/engine/stream.h:34` | SSD expert arena |
| `ExpertStream::Slot` / `Lay` / `Cls` / `Job` | `metal/src/engine/stream.h:76-79` | LRU slot, layer offsets, size class, I/O job |
| `ArenaMem` / `FootprintPlan` | `metal/src/engine/arena_budget.h:36-47`; `metal/src/engine/arena_budget.h:121-124` | Budget inputs / plan |
| `Window` | `metal/src/engine/window.cpp:115` | Decode+prompt engine |
| `Window::PromptGemm` | `metal/src/engine/window.cpp:1995` | Prompt GEMM state |
| `gguf::File` / `Tensor` | `metal/src/gguf_lite.h:13-21` | Lite GGUF |
| Quant `fmt<T_*>` / `T_Q2_0` etc. | `metal/kernels/quant.metal:13-53` | Per-format unit sizes |

### Env vars controlling this area (runtime; file:line of read)

| var | where | effect |
|---|---|---|
| `STRATA_METAL_STREAM` | `metal/src/engine/stream.cpp:34`; `metal/setup_mac.py:674` | 1 force stream / 0 force resident / unset=auto |
| `STRATA_METAL_CONTIG` | `metal/src/engine/stream.cpp:76` | Use `.experts` contiguous file (default on) |
| `STRATA_METAL_IO_THREADS` | `metal/src/engine/stream.cpp:149` | I/O thread count (default 6) |
| `STRATA_METAL_READ_SPLIT` | `metal/src/engine/stream.cpp:150` | Split expert reads (1..8) |
| `STRATA_METAL_PREFETCH` / `PREFETCH2` / `PREFETCH_MRU` | `metal/src/engine/stream.cpp:157-160`; `metal/src/engine/window.cpp:543` | Next-layer / +2 prefetch top-k; MRU policy |
| `STRATA_METAL_STREAM_FAKE` | `metal/src/engine/stream.cpp:162` | Fake stream (timing) |
| `STRATA_METAL_CLASS_WASTE` | `metal/src/engine/stream.cpp:127` | Slot size-class waste fraction |
| `STRATA_METAL_ARENA_GB` / `ARENA_STEP_GB` / `ARENA_CHUNK_GB` | `metal/src/engine/window.cpp:1165`; `metal/src/engine/stream.cpp:482`; `metal/src/engine/stream.cpp:426` | Arena size / stepped fill / chunk size |
| `STRATA_METAL_KEEP_FREE_GB` / `LEAVE_SHARE` | `metal/src/engine/window.cpp:1164-1166`; `metal/src/engine/arena_budget.h:44-46` | OS keep / leave share override |
| `STRATA_METAL_WIRE_ARENA` | `metal/src/engine/stream.cpp:552` | mlock arena (default on ≥64 GB) |
| `STRATA_METAL_EXPERT_PROFILE` | `metal/src/engine/window.cpp:1177` | Path to `expert-profile.bin` |
| `STRATA_METAL_KEEP_HOST_WEIGHTS` | `metal/src/engine/window.cpp:1132` | Do not purge CUDA-shaped weight uploads |
| `STRATA_METAL_ALLOC_LOG` | `metal/src/gpu/runtime.cpp:272` | Log ≥16 MB allocs |
| `STRATA_METAL_KERNELS` / `STRATA_GGML_COMMON` | `metal/src/mtl_rt.mm:49-52` | Override kernel dir / ggml-common path |
| `STRATA_METAL_MATH` | `metal/src/mtl_rt.mm:83` | `fast` enables fast math |
| `STRATA_METAL_MAX_THREADS_CAP` | `metal/src/mtl_rt.mm:142` | Test hook for require_fit |
| `STRATA_METAL_STUBS` | `metal/src/engine/missing.cpp:16` | `log` = list stubs, don’t exit |
| `STRATA_METAL_CTAB` | `metal/src/engine/window.cpp:69` | Quant constant-table path |
| `STRATA_METAL_FUSE` / `ROUTE` / `PARTIAL` / `LAZY_FOLD` | `metal/src/engine/window.cpp:241-247`; `metal/src/engine/window.cpp:1449`; `metal/src/engine/window.cpp:235` | Fusion / fused router / partial commit / lazy fold |
| `STRATA_METAL_GEMV_NR` | `metal/src/engine/window.cpp:577` | GEMV tile NR (1 or 2) |
| `STRATA_METAL_SKIP` / `KPROF` / `PROFILE` / `TRACE` / `LOGPOS` | `metal/src/engine/window.cpp:163`; `metal/src/engine/window.cpp:1008`; `metal/src/engine/window.cpp:1304`; `metal/src/engine/window.cpp:1336`; `metal/src/engine/window.cpp:1224` | Timing/debug |
| `STRATA_METAL_PROMPT*` / `PROMPT_GEMM*` / `PROMPT_CHUNK` / `PROMPT_EXPERTS` / `PROMPT_QSA` | `metal/src/engine/window.cpp:2009-2043`; `metal/src/engine/window.cpp:2558-2563`, `metal/src/engine/prompt_chain.inc:8-9` | Prompt path selection |
| `STRATA_METAL_MTP_*` / `STRATA_METAL_DRAFT_VOCAB` | `metal/src/engine/window.cpp:1508`; `metal/src/engine/window.cpp:1582`; `metal/src/engine/window.cpp:2529`; `metal/src/engine/window.cpp:1557` | MTP Q8/chain/GGUF/draft vocab |
| `STRATA_METAL_IGNORE_REPACK` | `metal/src/engine/window.cpp:420` | Allow start while metal_repack still running |
| `STRATA_HC_Q8` | `metal/src/engine/window.cpp:496` | Q8_0 HC matrices |
| `COOP_K4` | `metal/src/engine/prompt_chain.inc:27`; `metal/tests/coop_probe.h:11` | Cooperative K4 layout |

CMake options (not runtime): `STRATA_METAL_ENGINE`, `METAL_CPP_DIR`, `STRATA_GGML_DIR` (`metal/CMakeLists.txt:50-73`). The `strata` binary's CLI is the shared one from `src/program/generate.cpp` (see server.md); Metal-specific behavior is almost entirely via the env vars above.

## 5. Gotchas and constants

1. **Silent skipped kernels** — Metal will skip a dispatch with oversized threadgroups with **no error**; answers stay stale. Every dispatch goes through `require_fit` (`metal/src/mtl_rt.h:62-65`). GDN needs 512 (or 1024 for `gdn_step8`).
2. **No NVFP4 on Metal** — `metal/kernels/quant.metal` has no type 40; `metal/src/engine/stubs_144.cpp:39-51` stubs CPU NVFP4 and notes NVFP4 prompt paths are CUDA/HIP only (`metal/src/engine/stubs_144.cpp:4-5`).
3. **MSL is run-time compiled** — CLT has no offline `metal` compiler (`metal/src/mtl_rt.h:4-5`, `metal/include/strata/gpu/metal.hpp:23-24`). First launch pays compile cost; caches by (source, function[, constants]).
4. **Whole-GGUF NoCopy is forbidden on small Macs** — 37.6 GB wrap cannot be made resident in 32 GB (`metal/src/engine/stream.h:4-5`); streaming path loads dense only + expert arena.
5. **Footprint constants (hardcoded)** — `kLeaveShare=0.40`, `kResidentLeaveShare=0.30`, `kBigMac=64e9`, `kFileCacheHeadroom=12e9`, keep `max(7GB, 15%)`, PLE reserve 0.5 GB if not fully mapped (`metal/src/engine/arena_budget.h:49-81`, `metal/src/engine/arena_budget.h:109`). Derived from M4/M5 Max panic/soak measurements documented in-header.
6. **Arena wired with mlock** — default on ≥64 GB Macs; idle unwired arena was pageable and froze a Mac under file-cache pressure (`metal/src/engine/arena_budget.h:64-68`, `metal/src/engine/stream.cpp:546-553`).
7. **Most CUDA kernel entry points are stubs** — hot path is Window+mac-kern only; calling stubbed `strata::kernels::*` exits 86. Stubs are regenerated after porting (`metal/tools/gen_stubs.py`).
8. **Repack concurrency** — starting while `metal_repack` holds the shard can freeze a 128 GB Mac; engine refuses unless `STRATA_METAL_IGNORE_REPACK=1` (`metal/src/engine/window.cpp:420-422`).
9. **Version string** — Metal host defines `STRATA_VERSION="0.1.44.1-metal"` (`metal/CMakeLists.txt:110`).
10. **HIP parallel, not a fork of HIP sources** — Metal reimplements ops in MSL; HIP continues to compile the `.cu` files in `src/kernels/cuda/`.

## See also

| Area | Relation |
|---|---|
| engine-core.md | Shared host (`src/core/`) linked as `strata_metal_host`; geometry/session/weights; RemoteExpertOpt stubbed |
| gpu-cpu-kernels.md | CUDA/HIP kernels replaced on hot path by `metal/kernels` + Window; stubs cover the rest |
| moe-experts.md | Metal `ExpertStream` vs HIP/CUDA `ExpertCache` (unified_memory branch + CUDA VMM elastic) |
| prefill.md | Prompt GEMM/MoE MM in Metal; CUDA prefill MMQ not used |
| server.md | `src/program/generate.cpp` linked as `strata`; `src/core/verify.cpp` calls `window_*` |
| intel-sycl-backend.md | Sibling alt backend under `sycl/` (same “own CMake project” pattern) |
| model-spec-ngram-platform.md | GGUF tensors / Qwen3.5 shapes; expert-profile.bin; MTP GGUF |
| build-system.md | `metal/CMakeLists.txt`, metal-cpp, ggml pin |
| tests-bench-docs.md | `metal/tests/` parity and benches |
