# Strata map: Intel SYCL backend (`sycl/`)

Source: Strata commit `296b59d2`. Paths are relative to the Strata repo root.

## 1. Overview

- `sycl/` is a **self-contained Intel Arc port** of Strata’s CUDA engine: CUDA `.cu`/CUDA-touching `.cpp` are migrated with SYCLomatic (`dpct`) into `.dp.cpp` files under `sycl/src/` (and some plain `.cpp`), then built with `icpx -fsycl` against bundled `sycl/include/dpct/` helpers and a thin Strata-owned API surface (`sycl/include/strata/sycl_queue.hpp`, `sycl/include/strata/sycl_doorbell.hpp`, `sycl/include/strata/sycl_math.hpp`, `sycl/include/strata/copy_queue.hpp`).
- Upstream `src/` / `include/` are **not** modified; `sycl/CMakeLists.txt` resolves each TU to the migrated copy: a `.cu` must have one (`.dp.cpp`, else FATAL_ERROR), other files use the migrated copy when present, else fall back to the original under `../` (`sycl/CMakeLists.txt:88-107`).
- Serve/setup wrap upstream via `sycl/setup_intel.py` + `sycl/serve/strata-sycl.sh` + `sycl/serve/server_intel.py` (Docker + oneAPI runtime).
- Ops implemented are essentially the same as CUDA/HIP (MMVQ, IQ experts, MoE router, GDN, QSA, sampler, verify, prefill GEMM via oneMKL) plus **port-only** XMX (`joint_matrix`) paths (`sycl/src/kernels/cuda/qsa_prompt_attn_xmx.dp.cpp`, `xmx_gemm_iq` in `sycl/src/kernels/cuda/iq_kernels.dp.cpp`) and Intel-specific host↔device doorbell / arena / copy-engine workarounds.
- **NVFP4 experts: GPU paths refuse them** (CUDA-only exits, sycl/src/kernels/cuda/iq_kernels.dp.cpp:4522-4523, sycl/src/kernels/cuda/iq_kernels.dp.cpp:4623-4624, sycl/src/prefill/kernels.dp.cpp:2966); the CPU nvfp4_avx512/nvfp4_avx2 expert rows are built (sycl/CMakeLists.txt:183-186).
- Relative to HIP: HIP recompiles the same root `.cu` through `hip_compat/`; SYCL is a parallel tree of migrated sources + Intel-only infrastructure.

Line counts: strata-owned under `sycl/` excluding `include/dpct` and `third_party` 111,881 lines; `sycl/include/dpct` 37,649; largest TUs: `sycl/src/program/generate.cpp` 12996, `sycl/src/kernels/cuda/iq_kernels.dp.cpp` 5788, `sycl/src/prefill/prefill.cpp` 4715, `sycl/src/core/verify.cpp` 3964, `sycl/src/kernels/cuda/native_mmvq.dp.cpp` 3801, `sycl/src/core/expert_source.cpp` 3647.

## 2. File table

Paths relative to the repo root. DPCT headers summarized as one row (vendor migration helpers, not Strata logic).

### Top-level / build / setup

| path | lines | purpose |
|---|---|---|
| sycl/CMakeLists.txt | 396 | SYCL CMake: AOT (`STRATA_SYCL_AOT`), spin max, resolve migrated vs upstream, libs `strata_*`, parity tests |
| sycl/setup_intel.py | 366 | Wraps upstream `setup.py` for `--backend sycl`: Arc detect, Docker exe, config `env`, layer-split knobs |
| sycl/.gitignore | 4 | Ignores dpct migration output (`dpct_report.*`, `**/*.yaml`, `migrate.log`) (`sycl/.gitignore:1-4`) |

### Port-owned headers (`sycl/include/strata/`)

| path | lines | purpose |
|---|---|---|
| sycl/include/strata/sycl_queue.hpp | 290 | `q_of`, `big_fill_zero`, `host_malloc_polled`, `intel_gpu_driver`/`IntelGpuGen`, arena alias check/alloc |
| sycl/include/strata/sycl_doorbell.hpp | 154 | Uncached host↔device loads/stores (`sys_load`/`sys_store`), `spin_max`, `a770_fast` |
| sycl/include/strata/sycl_math.hpp | 27 | `dp4a` wrapper; sampler accumulator type (`samp_acc_t`) |
| sycl/include/strata/copy_queue.hpp | 27 | Declares Level Zero copy-engine queue API |
| sycl/include/strata/prefill/gdn_xe.hpp | 104 | Intel GDN recurrence variant (16-lane sub-group layout) |
| sycl/include/strata/core/ (12 headers) | 2812 | Port copies/overlays of session, verify, expert_source, gguf_expert_source, mtp, graph, dma_batch, … |
| sycl/include/strata/kernels/ (14 headers) | 909 | Thin decls incl. SYCL-only `sycl/include/strata/kernels/xmx_gemm.hpp`; `sycl/include/strata/kernels/qsa_prompt_attn_xmx.hpp`, `sycl/include/strata/kernels/native_mmvq.hpp` extras |
| sycl/include/dpct/ (46 files) | 37649 | SYCLomatic helper headers (device, memory, blas, dnnl, …) — not authored for Strata semantics |

### Probes (`sycl/probe/`)

| path | lines | purpose |
|---|---|---|
| sycl/probe/alias_scan.cpp | 38 | Find xe page-table aliasing across device alloc sizes |
| sycl/probe/arena_rw.cpp | 63 | Reproduce expert-cache fill/readback corruption shape |
| sycl/probe/doorbell_store.cpp | 43 | Host visibility of device stores mid-kernel |
| sycl/probe/fp64_cost.cpp | 105 | Cost of emulated FP64 vs float in small norms |
| sycl/probe/gemm_launch.cpp | 60 | oneMKL FP16 GEMM launch cost by rows |
| sycl/probe/gemm_vary.cpp | 47 | Per-M kernel-gen cost / alignment |
| sycl/probe/launch_cost.cpp | 45 | Empty-kernel submission latency |
| sycl/probe/payload_pairs.cpp | 61 | Paired mapped payload transfer contracts |
| sycl/probe/router_fast.cpp | 71 | A770 fast router bit-exact vs baseline |

### Serve

| path | lines | purpose |
|---|---|---|
| sycl/serve/strata-sycl.sh | 42 | Docker wrapper: mount `/work`, forward STRATA_/ONEAPI_/…, default VERIFY_NO_HOST |
| sycl/serve/server_intel.py | 314 | Patches upstream server: Xe telemetry, model switcher web, logprobs, narrator |
| sycl/serve/xe_telemetry.py | 117 | Arc (xe) GPU reader for the Monitor tab: temperature, power, PCIe link from sysfs; load and VRAM from the root sampler's /run/gpustat.json (`sycl/serve/xe_telemetry.py:1-6`) |
| sycl/serve/web/switcher.js, sycl/serve/web/switcher.css | 60, 7 | UI for model switcher |

### Tools

| path | lines | purpose |
|---|---|---|
| sycl/tools/migrate.sh | 49 | Run `dpct` migration into out tree |
| sycl/tools/normalize.sh | 26 | Shape dpct out for 3-way merge; run fixups |
| sycl/tools/fixups.py | 342 | Idempotent post-dpct source rewrites |
| sycl/tools/merge_upstream.py | 37 | 3-way merge migrated files across upstream revs |
| sycl/tools/build.sh | 12 | Configure+build inside strata-sycl-dev |
| sycl/tools/Dockerfile, sycl/tools/Dockerfile.unitrace | 9, 8 | oneAPI 2026.1 dev image; the unitrace image adds Intel pti-gpu unitrace (per-kernel Level Zero profiling) |
| sycl/tools/get-cuda-headers.sh | 28 | Pip-fetch CUDA headers for dpct parse |
| sycl/tools/gpustat.py, sycl/tools/gpustat.service | 182, 10 | Root systemd telemetry sampler writing /run/gpustat.json for the Monitor tab (VRAM from per-client fdinfo, root-only; busy, power, temps, PCIe link) (`sycl/tools/gpustat.py:2-12`, `sycl/tools/gpustat.service:2`) |

### Core sources (`sycl/src/core/`) — migrated / port

| path | lines | purpose |
|---|---|---|
| sycl/src/core/expert_source.cpp | 3647 | Expert pool adapter (shared contract; large) |
| sycl/src/core/verify.cpp | 3964 | Speculative verify plan (device plan / NO_HOST sensitive) |
| sycl/src/core/mtp.cpp | 1969 | Multi-token prediction |
| sycl/src/core/layer.cpp | 1741 | Layer forward / moe_combine |
| sycl/src/core/session.cpp | 1449 | Session graphs / decode loop glue |
| sycl/src/core/expert_cache.cpp | 1059 | VRAM expert cache |
| sycl/src/core/pinned.dp.cpp | 901 | Pinned / USM host buffers (migrated) |
| sycl/src/core/weights.cpp | 534 | Weight load |
| sycl/src/core/remote_experts.cpp | 477 | Migrated static, profile-filled expert tier on another device; results return through the pinned CPU rows (`sycl/include/strata/core/remote_experts.hpp:15-16`); the port calls its remote-expert helpers single-device stubs (`sycl/src/core/remote_expert_opt_stub.cpp:1-2`) |
| sycl/src/core/device.dp.cpp | 485 | Device info / DeviceArena (dpct; still carries HIP/CUDA `#if` arms from upstream) |
| sycl/src/core/graph.cpp | 249 | CUDA-graph → SYCL graph path |
| sycl/src/core/gguf_expert_source.cpp | 239 | **Port**: stream experts from GGUF + host mirror |
| sycl/src/core/copy_queue.cpp | 142 | **Port**: Level Zero copy-engine queue |
| sycl/src/core/native_dense.cpp, sycl/src/core/native_head.cpp, sycl/src/core/peer_experts.cpp, sycl/src/core/overlap_main.cpp, sycl/src/core/concurrent_main.cpp, sycl/src/core/conversation_snapshot.cpp, sycl/src/core/conversation_snapshot_test.cpp, sycl/src/core/conversation_state.cpp | 421, 293, 41, 239, 272, 324, 432, 395 | Same roles as engine-core.md; peer/remote opt stubbed; overlap_main.cpp and concurrent_main.cpp (measurement programs, sycl/src/core/overlap_main.cpp:1, sycl/src/core/concurrent_main.cpp:1) are not built by sycl/CMakeLists.txt; conversation_snapshot_test is a ctest (sycl/CMakeLists.txt:346-350) |
| sycl/src/core/remote_expert_opt_stub.cpp | 38 | Stub: no second-GPU remote-opt tier |
| sycl/src/core/vmm.cpp | 27 | Stub: no Level Zero driver VMM |

### Kernels (`.dp.cpp` files in `sycl/src/kernels/cuda/` + parity)

- Built kernel stems (CMake `_kernel_cu`, `sycl/CMakeLists.txt:213-217`): dequant_s2, s2_gemv, s_gemv, s2_gemv_quads, s2_gemv_fast, router_top10, quantize_act, rope, sampler, s2_gemv_q8, s2_expert_grouped, shared_expert, gr, gdn, qsa, elementwise, bf16_gemv, native_mmvq, native_bf16, native_gr_norm, native_moe, native_gdn, native_gr_postops, native_router, native_gdn_preprocess, native_qsa, native_rope, native_ple_postops, native_flash_attn, native_qsa_indexer, native_qsa_score, ple, kv_q8, kv_stream, kv_q4, fused_gr, fused_gdn, cvec, qsa_decode_attn, qsa_prompt_attn, dequant_bf16, qsa_select, verify_kernels, iq_kernels — plus **port-only** `sycl/src/kernels/cuda/qsa_prompt_attn_xmx.dp.cpp` (388).
- Largest: iq_kernels 5788, native_mmvq 3801, verify_kernels 2593, fused_gr 2396, qsa_select 2096, qsa_prompt_attn 2056, s2_expert_grouped 1879, sampler 1777.
- Many parity and bench `.cpp` files under `sycl/src/kernels/` (`sycl/CMakeLists.txt:253-264`, `sycl/CMakeLists.txt:382-384`).
- **Not in SYCL cuda list vs root:** `src/kernels/cuda/route_prior.cu`, `src/kernels/cuda/q8_1_il.cuh`.
- **Only in SYCL:** `sycl/src/kernels/cuda/qsa_prompt_attn_xmx.dp.cpp`.

### Prefill / program / other

| path | lines | purpose |
|---|---|---|
| sycl/src/prefill/prefill.cpp | 4715 | Prefill orchestration |
| sycl/src/prefill/kernels.dp.cpp | 3121 | Prefill kernels incl. GDN variants |
| sycl/src/prefill/moe_fused.dp.cpp, sycl/src/prefill/moe_fused_iq.dp.cpp, sycl/src/prefill/moe_fused_stub.cpp | 1162, 1469, 37 | dpct copies of upstream's fused int8 prompt MoE kernels (#136), not built; moe_fused_stub.cpp is built in their place (sycl/src/prefill/moe_fused.dp.cpp:1-2, sycl/src/prefill/moe_fused_stub.cpp:1-3, sycl/CMakeLists.txt:243) |
| sycl/src/prefill/gemm.dp.cpp | 1015 | GEMM (oneMKL via dpct blas) |
| sycl/src/prefill/gemm_bf16_parity.dp.cpp | 221 | Migrated Gemm::bf16 parity test vs cuBLAS's BF16 product (sycl/src/prefill/gemm_bf16_parity.dp.cpp:1); not built by sycl/CMakeLists.txt |
| sycl/src/prefill/moe_mmq.dp.cpp, sycl/src/prefill/xmx_moe.dp.cpp, sycl/src/prefill/ggml_cuda_host.dp.cpp | 311, 194, 104 | dpct copies of the MMQ path (llama.cpp ggml-cuda) and its ggml host glue are in the tree (`sycl/src/prefill/moe_mmq.dp.cpp:1-4`, `sycl/src/prefill/ggml_cuda_host.dp.cpp:1-5`) but are skipped by the current migration (`sycl/tools/migrate.sh:26-28`) and not built (`sycl/CMakeLists.txt:241-244`); xmx_moe is built: A770 grouped FP16 XMX expert GEMM, `STRATA_PF_XMX=1` (`sycl/CMakeLists.txt:244`, `sycl/src/prefill/xmx_moe.dp.cpp:1-3`) |
| sycl/src/program/generate.cpp | 12996 | CLI + engine main (port flags: `--stream-experts`, xe checks) |
| sycl/src/ngram/ple_reader.cpp | 570 | PLE SSD row reader: upstream's file plus a per-gather timing trace, `STRATA_PLE_TRACE=1` (`sycl/src/ngram/ple_reader.cpp:1-2`) |
| sycl/src/platform/direct_file.cpp | 369 | Direct I/O helper: upstream's file with a deeper default read queue on Linux (`sycl/src/platform/direct_file.cpp:1`) |
| sycl/third_party/ggml/ggml-common.h | 2193 | Pinned ggml types for kernels |
| sycl/third_party/ggml/LICENSE | 21 | MIT license of the vendored ggml header |

## 3. Where it does X

### Stream / queue null-safety — `sycl/include/strata/sycl_queue.hpp:12` `q_of`

Every launcher still takes `void* stream` (CUDA null = default). dpct casts to `dpct::queue_ptr` and would crash on null; `q_of` maps null → `dpct::get_in_order_queue()`. **See also:** gpu-cpu-kernels.md (all `.dp.cpp` launchers), engine-core.md (session/verify streams).

### Large device zero-fill — `sycl/include/strata/sycl_queue.hpp:18-23` `big_fill_zero`

Chunks fills (default 256 MiB) as compute kernels + wait; plain `queue.memset` of many GiB on Arc Pro B70 (xe) hits blitter timeout / GT reset. Env `STRATA_CHUNKED_FILL=0` forces memset. **See also:** engine-core.md (DeviceArena / cache init).

### Polled host USM — `sycl/include/strata/sycl_queue.hpp:63` `host_malloc_polled` / `sycl/include/strata/sycl_queue.hpp:79` `host_free_polled`

On Level Zero, prefers `zeMemAllocHost(... BIAS_UNCACHED)` so device kernels see host stores (Alchemist i915 ordinary host USM was invisible, `sycl/include/strata/sycl_queue.hpp:42-47`). `STRATA_HOST_UNCACHED=0` disables. **See also:** engine-core.md (doorbell flags, CPU expert result rows), moe-experts.md.

### Driver / GPU gen — `sycl/include/strata/sycl_queue.hpp:101` `intel_gpu_driver`, `sycl/include/strata/sycl_queue.hpp:123` `IntelGpuGen`, `sycl/include/strata/sycl_queue.hpp:124` `intel_gpu_gen`

Reads sysfs for `xe` vs `i915`; classifies Alchemist vs Battlemage from the device's own info (`sycl/include/strata/sycl_queue.hpp:120-122`). Used by spin bounds, doorbell plain payload default, copy-engine gating, generate xe guards. **See also:** server.md, engine-core.md.

### Arena page aliasing — `sycl/include/strata/sycl_queue.hpp:168` `arena_alias_check`, `sycl/include/strata/sycl_queue.hpp:259` `malloc_device_guarded` alloc-with-retry

xe page tables can alias 2 MiB pages (write to one shows at another). Alloc path retries with spacers; `STRATA_ARENA_ALIAS_CHECK=0` skips. Probes: `sycl/probe/alias_scan.cpp:1-2`, `sycl/probe/arena_rw.cpp:1-3`. **See also:** engine-core.md (expert cache arena), gpu-cpu-kernels.md (fill kernels).

### Host↔device doorbell — `sycl/include/strata/sycl_doorbell.hpp:56`, `sycl/include/strata/sycl_doorbell.hpp:77` `sys_load`/`sys_store` (default uncached annotated_ptr)

- CUDA `volatile` does not bypass L3 on Intel; system-scope atomics still cached on some Arc.
- Default path uses Intel experimental uncached L1+L3 read/write hints + acquire/release fences.
- `-DSTRATA_DOORBELL_ATOMIC_LOAD/STORE` restores atomics.
- `spin_max` `sycl/include/strata/sycl_doorbell.hpp:134-149`: 20k reads on xe and on a Battlemage card with no sysfs driver (2M there in a run that drafts without MTP), 2M elsewhere; `STRATA_SPIN_MAX` / CMake `STRATA_SYCL_SPIN_MAX` override.
- `doorbell_plain_payload` `sycl/include/strata/sycl_doorbell.hpp:106-110` defaults plain on i915.
- `a770_fast` `sycl/include/strata/sycl_doorbell.hpp:21-23` opt-in paired 64-bit payload path (`STRATA_SYCL_A770_FAST=1`).
- **See also:** engine-core.md (verify/decode handshake), gpu-cpu-kernels.md (`elementwise` publish), moe-experts.md (CPU result rows).
- Without working doorbell, decode waits whole graph (`STRATA_VERIFY_NO_HOST`) and experts must be device-resident.

### Copy (blitter) engine queue — `sycl/include/strata/copy_queue.hpp:22`, `sycl/src/core/copy_queue.cpp:73` `make_copy_engine_queue`

Level Zero v2 puts H2D copies on compute engine; dedicated COPY queue group wraps via `sycl::make_queue<ext_oneapi_level_zero>`. Default on for Battlemage; `STRATA_COPY_ENGINE=0` off, `=1` force any L0 device. On an A750 (i915) the prompt path once hung with it on, GPU not hung (`sycl/src/core/copy_queue.cpp:88-90`, `sycl/include/strata/copy_queue.hpp:3-15`, `sycl/src/core/copy_queue.cpp:73-98`). **See also:** prefill.md (expert streaming during prompt), engine-core.md (stager).

### Stream experts from GGUF (no host arena) — `sycl/include/strata/core/gguf_expert_source.hpp:27` `GgufExpertSource`, `sycl/src/core/gguf_expert_source.cpp`, `sycl/src/program/generate.cpp:1976` `--stream-experts`, `sycl/src/program/generate.cpp:4274`

Port-specific `ExpertSource`: gather blobs from shard on demand; `mirror()` `sycl/src/core/gguf_expert_source.cpp:118` (declared at `sycl/include/strata/core/gguf_expert_source.hpp:51`) builds USM-host mirror for overflows. On xe, pageable host expert DMA faults — `sycl/src/program/generate.cpp:2610-2622` defaults `--pcie-frac` to 0 and warns (never refuses) without `--stream-experts`. **See also:** moe-experts.md (CPU pool misses), engine-core.md (expert_cache, verify plan), server.md (`setup_intel` sets stream-experts / profiles).

### Device verify / no host handshake — env `STRATA_VERIFY_NO_HOST`, `STRATA_VERIFY_DEVICE_PLAN` (defaults in `sycl/serve/strata-sycl.sh:25`)

When host cannot see mid-kernel rings (xe), verify runs as device plan with no host expert answers; missing experts must be VRAM or mirrored (`sycl/src/program/generate.cpp:4961-4966`). Layer-split short VRAM → setup may force `STRATA_VERIFY_NO_HOST=0` so CPU computes overflow (`sycl/setup_intel.py:316-317`). **See also:** engine-core.md (verify.cpp), moe-experts.md, server.md.

### XMX / joint_matrix GEMM & prompt attn — `sycl/include/strata/kernels/xmx_gemm.hpp:5-9` `xmx_gemm_iq`; `sycl/include/strata/kernels/qsa_prompt_attn_xmx.hpp:9`; `sycl/src/kernels/cuda/qsa_prompt_attn_xmx.dp.cpp`; CMake `sycl/CMakeLists.txt:40-43` `STRATA_XMX_SG16` / `STRATA_NO_DG2_XMX`

Port-only kernels: quantized weight × FP16 act via `joint_matrix` (types IQ3_XXS/IQ4_NL/IQ3_S/IQ2_S/Q2_0). Fallback = dequant + oneMKL (`sycl/include/strata/kernels/xmx_gemm.hpp:7-8`). The XMX prompt attention is off by default (slower than its fallback); `STRATA_PROMPT_ATTN_XMX=1` enables it (`sycl/include/strata/kernels/qsa_prompt_attn_xmx.hpp:8`). AOT `bmg*`/`lnl*` → SG16 8x16x16; `pvc` disables DG2 XMX. **See also:** gpu-cpu-kernels.md (iq_kernels), prefill.md (prompt path). 

### GDN on Xe — `sycl/include/strata/prefill/gdn_xe.hpp:18` `gdn_xe_sum4`, `sycl/include/strata/prefill/gdn_xe.hpp:26` `gdn_xe_body`, `sycl/include/strata/prefill/gdn_xe.hpp:92` `gdn_xe_launch`

Prefill GDN variant 6: 16-lane sub-group packs four columns × four row groups; shuffles replace SLM; bit-identical to variant 0 on A770 (`sycl/include/strata/prefill/gdn_xe.hpp:1-5`, `sycl/include/strata/prefill/gdn_xe.hpp:12`). Constants `GDN_XE_S=128`, `HK=16`, `HV=48`, `C=10240`. **See also:** prefill.md (`sycl/src/prefill/kernels.dp.cpp`).

### MMVQ / expert kernels — same stems as gpu-cpu-kernels.md, as `.dp.cpp` files in `sycl/src/kernels/cuda/`

`sycl/include/strata/kernels/native_mmvq.hpp` adds SYCL-only knobs e.g. `native_mmvq_set_q6k_wide` `sycl/include/strata/kernels/native_mmvq.hpp:33`. Sub-group forced to **32** in CMake (Xe2 default 16 would race warp-sync code; `sycl/CMakeLists.txt:76-78`). FP32 divide/sqrt correctly rounded for quant byte-exact vs ggml (`sycl/CMakeLists.txt:53-57`). **See also:** gpu-cpu-kernels.md map for op semantics; build-system.md for flags.

### Prefill GEMM — `sycl/src/prefill/gemm.dp.cpp` via oneMKL (`dpct::blas`); probes `sycl/probe/gemm_launch.cpp:1-3`, `sycl/probe/gemm_vary.cpp:1-2`

Two probes test whether the prompt path (one gate/up and one down GEMM per routed expert, ~25k experts per 4K prompt) is bound by launches rather than by the XMX units (`sycl/probe/gemm_launch.cpp:1-3`, `sycl/probe/launch_cost.cpp:1-3`). Fused int8 MoE (`src/prefill/moe_fused.cu`, `src/prefill/moe_fused_iq.cu`) **not built** — `sycl/src/prefill/moe_fused_stub.cpp:1-3`, `sycl/CMakeLists.txt:243`. **See also:** prefill.md, build-system.md.

### Stubs / missing tiers — `sycl/src/core/vmm.cpp`, `sycl/src/core/remote_expert_opt_stub.cpp`, `sycl/CMakeLists.txt:232-234`

No driver VMM; no remote-expert-opt / second-GPU peer tier yet. **See also:** engine-core.md (HIP/CUDA have real src/core/remote_expert_opt.cu).

### Migration pipeline — `sycl/tools/migrate.sh` → `sycl/tools/normalize.sh` → `sycl/tools/fixups.py` → `sycl/tools/merge_upstream.py`

dpct with experimental graph/matrix/…; fixups repair queue casts, null stream, cudaGraphUpload, `__fadd_rn`, `cudaStreamQuery`, GDN cp.async off, etc. **See also:** build-system.md; docs `docs/INTEL.md:168-184`; dpct flags `sycl/tools/migrate.sh:39-47`; fixups `sycl/tools/fixups.py:56-59` (cudaStreamQuery), `sycl/tools/fixups.py:110-114` (GDN cp.async off), `sycl/tools/fixups.py:61-67` (queue casts).

### Serve integration — `sycl/setup_intel.py` `to_sycl` / `install`; `sycl/serve/strata-sycl.sh`; `sycl/serve/server_intel.py`

Config `"backend": "sycl"`, `"exe": sycl/serve/strata-sycl.sh` (`sycl/setup_intel.py:37`, `sycl/setup_intel.py:206`). Forwards STRATA_/ONEAPI_/UR_/IGC_/SYCL_/ZES_ plus NEOReadDebugKeys and OverrideDefaultFP64Settings (not STRATA_SYCL_*), `sycl/serve/strata-sycl.sh:28-29`. FP64 emu env on Alchemist: `IGC_EnableDPEmulation`, `OverrideDefaultFP64Settings`, `NEOReadDebugKeys` (`sycl/setup_intel.py:213-216`). **See also:** server.md, model-spec-ngram-platform.md (packs/profiles `expert-profile-sycl.bin`).

### HIP mirror / differ (comparison, not this tree’s build)

| Aspect | HIP (root) | SYCL (`sycl/`) |
|---|---|---|
| Sources | Same `.cu` files in `src/kernels/cuda/` via HIP | Migrated copies under `sycl/src/` |
| Compat layer | `include/strata/hip_compat/` | `sycl/include/dpct/` + `sycl/include/strata/sycl_queue.hpp`, `sycl/include/strata/sycl_doorbell.hpp`, `sycl/include/strata/sycl_math.hpp` |
| Matrix units | AMD WMMA in `src/kernels/cuda/qsa_prompt_attn.cu`, `src/kernels/cuda/qsa_select.cu` | XMX `joint_matrix` port-only files |
| Expert host model | Arena / mmap / CPU pool | `--stream-experts` + GgufExpertSource + optional mirror |
| Doorbell | CUDA/HIP volatile | Uncached annotated_ptr loads/stores + system-scope fences (atomics only with `-DSTRATA_DOORBELL_ATOMIC_LOAD/STORE`, `sycl/include/strata/sycl_doorbell.hpp:51-82`) |
| NVFP4 | Implemented (iq_kernels / cpu nvfp4_*) | GPU paths refuse it (CUDA-only exits, sycl/src/kernels/cuda/iq_kernels.dp.cpp:4522-4523, sycl/src/kernels/cuda/iq_kernels.dp.cpp:4623-4624, sycl/src/prefill/kernels.dp.cpp:2966); the CPU nvfp4_avx512/nvfp4_avx2 expert rows are built (sycl/CMakeLists.txt:183-186) |



## 4. Key types / structs and env / CLI / CMake

### Types (port-owned)

- `strata::IntelGpuGen` — `sycl/include/strata/sycl_queue.hpp:123` (`Unknown`, `Alchemist`, `Battlemage`)
- `strata::sys_atomic_u32` — `sycl/include/strata/sycl_doorbell.hpp:26`
- `strata::samp_acc_t` — `sycl/include/strata/sycl_math.hpp:22-25` (double if `STRATA_SYCL_SAMPLER_FP64`, else float)
- `strata::core::GgufExpertSource` — `sycl/include/strata/core/gguf_expert_source.hpp:27`
- `strata::core::Verifier`, `SessionState`, `SessionGraphs`, `TokenGraph`, … — port overlays of engine-core.md headers under `sycl/include/strata/core/`
- `strata::kernels::SamplerParams`, `RopeScaling`, … — kernel headers
- Prefill: `GDN_XE_*` constexprs — `sycl/include/strata/prefill/gdn_xe.hpp:12-14`

### Env vars (SYCL-critical; file:line of reader)

| var | where | effect |
|---|---|---|
| STRATA_CHUNKED_FILL | sycl/include/strata/sycl_queue.hpp:22 | `0` → plain memset |
| STRATA_HOST_UNCACHED | sycl/include/strata/sycl_queue.hpp:65 | `0` → no ze uncached host |
| STRATA_ARENA_ALIAS_CHECK | sycl/include/strata/sycl_queue.hpp:209 | `0` skips alias retry |
| STRATA_SYCL_A770_FAST | sycl/include/strata/sycl_doorbell.hpp:23 | opt-in fast router / paired payloads |
| STRATA_DOORBELL_PLAIN | sycl/include/strata/sycl_doorbell.hpp:108 | `1` plain / `0` uncached-hinted payload (overrides the i915 → plain default) |
| STRATA_SPIN_MAX | sycl/include/strata/sycl_doorbell.hpp:142 | override spin bound (reads) |
| STRATA_COPY_ENGINE | sycl/src/core/copy_queue.cpp:74, sycl/src/core/copy_queue.cpp:92 | `0` off / `1` force copy queue |
| STRATA_PROMPT_ATTN_XMX | sycl/src/kernels/cuda/qsa_prompt_attn_xmx.dp.cpp:369-370 | `1` = XMX prompt attention, 64-cell chunks; `32` = 32-cell chunks; unset/`0` = fallback (off by default) |
| STRATA_VERIFY_NO_HOST | sycl/src/program/generate.cpp:4961; sycl/serve/strata-sycl.sh:25 | device verify, no host expert answers |
| STRATA_VERIFY_DEVICE_PLAN | sycl/serve/strata-sycl.sh:25 | device-built verify plan |
| STRATA_STAGER_THREADS | sycl/serve/strata-sycl.sh:25 | default 12 |
| STRATA_STAGE_TRIM | sycl/setup_intel.py:311 | layer-split staging |
| STRATA_SYCL_ROOT/BIN/IMAGE/NAME | sycl/serve/strata-sycl.sh:6-9 | Docker mount/binary/image/name |
| ONEAPI_DEVICE_SELECTOR | sycl/setup_intel.py:307; sycl/tools/Dockerfile:8 | device pick / multi-card |
| SYCL_CACHE_PERSISTENT | sycl/tools/Dockerfile:8; docs/INTEL.md:115 | `0` in the image (persistent JIT cache segfaults on Xe2) |
| IGC_EnableDPEmulation / OverrideDefaultFP64Settings / NEOReadDebugKeys | sycl/setup_intel.py:216 | Alchemist FP64 emu |
| STRATA_DOORBELL_CHECK | sycl/CMakeLists.txt:381 (test env) | payload publish check |

Compile-time: `STRATA_SYCL_AOT`, `STRATA_SYCL_SPIN_MAX`, `STRATA_XMX_SG16`, `STRATA_NO_DG2_XMX`, `STRATA_DOORBELL_ATOMIC_LOAD/STORE`, `STRATA_SYCL_SAMPLER_FP64`, `STRATA_SYCL_JIT_STATELESS`, `STRATA_NATIVE_EXPERTS`, `STRATA_Q6K_EXPERTS`, `STRATA_SYCL_PARITY` — `sycl/CMakeLists.txt:31-65`, `sycl/CMakeLists.txt:151-255`; `STRATA_DOORBELL_ATOMIC_LOAD/STORE` at `sycl/include/strata/sycl_doorbell.hpp:51`, `sycl/include/strata/sycl_doorbell.hpp:72`; `STRATA_SYCL_SAMPLER_FP64` at `sycl/include/strata/sycl_math.hpp:22`.

### CLI (generate / setup)

- `--stream-experts` — `sycl/src/program/generate.cpp:1976` / option field `sycl/src/program/generate.cpp:513` (SYCL port primary expert residency model)
- `--mmap-experts`, `--expert-profile`, `--layer-split`, `--ple-io`, `--kv-resident` — shared with upstream; setup_intel retargets profiles to `expert-profile-sycl.bin` on non-i915 cards (`sycl/setup_intel.py:167-176`) and VRAM/RAM rules per `docs/INTEL.md:681-691`
- Setup entry: `./setup.sh --backend sycl` (not bare `python3 sycl/setup_intel.py` on Ubuntu externally-managed Python; `docs/INTEL.md:51-53`)

## 5. Gotchas and constants

1. **Separate tree, not HIP-style in-place compile.** Touching root `.cu` does nothing for Intel until re-migrate + fixups + merge. Upstream merges stay clean by design (`docs/INTEL.md:8`).
2. **Sub-group size forced to 32** everywhere (`sycl/CMakeLists.txt:76-78`): Xe2 default 16 breaks warp-synchronous kernels.
3. **NVFP4 GPU paths are CUDA-only here; the CPU nvfp4 expert rows are built (sycl/CMakeLists.txt:183-186).** `sycl/src/kernels/cuda/iq_kernels.dp.cpp:4522-4523`, `sycl/src/kernels/cuda/iq_kernels.dp.cpp:4623-4624` and `sycl/src/prefill/kernels.dp.cpp:2966` (`moe_combine` per-row scales exit). For NVFP4 see gpu-cpu-kernels.md.
4. **Doorbell / NO_HOST coupling.** Broken visibility forced `STRATA_VERIFY_NO_HOST` + full VRAM residency; uncached hints restored mid-window rings (measured 482 ms vs 44 ms, `sycl/include/strata/sycl_doorbell.hpp:66-70`).
5. **xe vs i915 behavior differs:** stream-experts set by setup on any driver other than i915 (`sycl/setup_intel.py:198-199`; generate only recommends it, `sycl/src/program/generate.cpp:2613-2614`); i915 loads experts into an ordinary RAM arena and the CPU computes the ones the card does not hold, with no mirror built, + `STRATA_VERIFY_NO_HOST=0` (A750 path, `sycl/setup_intel.py:193-194`, `sycl/setup_intel.py:210-212`). Copy-engine default Battlemage-only.
6. **Arena aliasing on xe** is real (2 MiB wrong pages, `sycl/include/strata/sycl_queue.hpp:163-164`); probes exist; `STRATA_ARENA_ALIAS_CHECK` is on by default (`sycl/include/strata/sycl_queue.hpp:209`).
7. **big_fill_zero chunk default 256 MiB** (`sycl/include/strata/sycl_queue.hpp:18-21`); large memset unsafe on B70.
8. **spin_max defaults:** 20000 (xe; Battlemage with no sysfs driver unless drafting without MTP) vs 2000000 (else) — `sycl/include/strata/sycl_doorbell.hpp:146-149`.
9. **GDN_XE constants** hardcoded (`S=128`, `HK=16`, `HV=48`, `C=10240`, `sycl/include/strata/prefill/gdn_xe.hpp:12`) — geometry-specific.
10. **Fused int8 prefill MoE not in SYCL build** (`sycl/src/prefill/moe_fused_stub.cpp:1-3`); ggml MMQ path also skipped at migrate (`sycl/tools/migrate.sh:28` skip list).
11. **Images / peer GPU tier / VMM:** not yet / stubs (`docs/INTEL.md:746-747`, `docs/INTEL.md:639-640`; `sycl/src/core/remote_expert_opt_stub.cpp:1-3`; `sycl/src/core/vmm.cpp:1-6`).
12. **fp-model=precise + correctly-rounded divide/sqrt** required for quant parity (`sycl/CMakeLists.txt:53-57`, `sycl/CMakeLists.txt:74-75`, `sycl/CMakeLists.txt:83`); missing AOT flag historically caused 303k quantize_act mismatches.
13. **Docker required** for oneAPI (`sycl/serve/strata-sycl.sh:2-3`); the image sets `SYCL_CACHE_PERSISTENT=0` (`sycl/tools/Dockerfile:8`; persistent JIT cache segfaults on Xe2, `docs/INTEL.md:115`).
14. **`sycl/src/program/generate.cpp` / `sycl/src/core/device.dp.cpp` still contain HIP `#if` arms** (`sycl/src/core/device.dp.cpp:45-47`, `sycl/src/program/generate.cpp:202`) from shared upstream text.
15. **Parity / probes are the regression surface** for Intel-only memory contracts; root HIP tests do not cover doorbell/uncached/alias.

