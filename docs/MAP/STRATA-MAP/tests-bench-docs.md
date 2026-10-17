# Strata map: tests, bench and docs

Source: Strata commit `296b59d2`. Paths are relative to the Strata repo root.

## 1. Overview

- Strata’s public claim (README.md:5; README.md:12; README.md:58; README.md:218-220; docs/HOW_IT_WORKS.md:27) is that Qwen3.8-Flash-Next (125B-class MoE, 24,576 experts, 10 active per token) runs on a gaming PC with ≥12 GB VRAM by pinning all experts in system RAM, caching the most-used experts in VRAM, and computing the rest on the CPU at the same time. This map covers the **tests/** tree (51 files: core/cuda/hip/platform/prefill/program + fixture data), **docs/** (design + install + AMD/NVIDIA/Intel/Mac notes + paper/media/monitoring), root **README.md** / **AGENTS.md**, and **bench/** including **bench/results** (132 dated folders, 2,086 files — summarized, not enumerated).
- Official speed tables are NVIDIA RTX 5070–centric (docs/DETAILS.md:17; bench/results/2026-09-29-speed-0126/README.md:3); AMD RDNA4 (gfx1200/gfx1201: RX 9060 XT / 9070 / 9070 XT / R9700) is documented in docs/AMD_HIP.md:1-6; docs/AMD_HIP.md:270-296 with 4K/16K validation on a contributor's (doplxyz) test machine and richer community long-context runs under bench/results.
- Expert-cache “hit_rate” is defined as VRAM share of looked-up experts (docs/DETAILS.md:309), not a simple miss counter — high hit_rate with an oversized cache can still mean slow decode when VRAM is exhausted (docs/DETAILS.md:639-640).

## 2. File tables

### 2a. Root entrypoints

| path | lines | purpose |
| --- | ---: | --- |
| README.md | 243 | English product README: speed tables (RTX 5070 + RX 9070 XT), hardware list incl. R9700, install, models, how-it-works |
| README.zh-CN.md, README.ja.md, README.de.md, README.fr.md, README.es.md, README.pt-BR.md | 217 / 229 / 239 / 238 / 233 / 234 | Translations of README; English README is the mapped source |
| AGENTS.md | 36 | Contributor/agent brief: AI_SETUP, DETAILS, AMD_HIP, community-bench rules, build all backends |
| bench.sh | 10 | Linux wrapper → `.venv/bin/python tools/strata_bench.py` |
| BENCH.bat | 16 | Windows wrapper → same `tools/strata_bench.py` |

### 2b. tests/

| path | lines | purpose |
| --- | ---: | --- |
| tests/core/batch_rows_test.cpp | 78 | Batch window row layout / MTP groups / layer-split hand-off capacity (CPU header-only) |
| tests/core/dequant_q5_1_test.cpp | 40 | PLE Q5_1 dequant vs ggml on synthetic blocks |
| tests/core/dma_batch_parity.cu | 107 | Byte parity + event ordering for pinned-host expert DMA uploads (CUDA) |
| tests/core/exchange_storage_test.cpp | 133 | ExchangeStorage resident/spare exchanges: byte, alias (no stale CUDA address) and ownership checks |
| tests/core/expert_cache_per_layer_test.cpp | 44 | Per-layer admission into disjoint slot ranges (needs CUDA; exit 77 if none) |
| tests/core/expert_cache_segmented_test.cpp | 101 | Segmented/elastic expert cache shrink/grow keeps slot addresses |
| tests/core/expert_layout_test.cpp | 366 | native_experts.txt v2/v3/v4 (fixtures in tests/data/native_experts) + arena fill from GGUF shards (no GPU) |
| tests/core/expert_profile_save_test.cpp | 80 | Profile save order + byte-identical write_expert_profile |
| tests/core/expert_profile_version_test.cpp | 73 | Refuse non-v1 expert-profile headers before geometry check |
| tests/core/file_expert_source_test.cpp | 864 | FileExpertSource / foresight swap / CPU expert kernels (large CUDA integration) |
| tests/core/gguf_fixture.hpp | 132 | Synthetic GGUF v3 builder for loader tests (no gguf-py/GPU) |
| tests/core/gguf_geometry_test.cpp | 271 | read_geometry vs compiled kernel shape from GGUF metadata |
| tests/core/gguf_split_test.cpp | 220 | Split-shard paths + GgufModel joint view on synthetic GGUFs |
| tests/core/host_graph_launch_test.cu | 82 | Host graph launch backpressure / early-failure release under sanitizer |
| tests/core/memory_guard_test.cpp | 354 | available_host_bytes: MemAvailable / cgroup / operator cap (CPU fixtures) |
| tests/core/model_family_test.cpp | 292 | Model families (qwen4exp / qwen35moe) on synthetic GGUFs: architecture guard, geometry_from_gguf, layout tables |
| tests/core/native_dense_ple_key_test.cpp | 169 | BF16 PLE key row for native packs (OrcaRouter IQ3_XXS compat) |
| tests/core/split_lend_complement_test.cpp | 219 | Resident RAM + layer-split: lend-region complement vs per-stage cache |
| tests/core/vmm_test.cpp | 71 | Elastic KV VMM map/unmap/remap byte preservation |
| tests/cuda/prefill_fused_iq_test.cpp | 700 | Fused int8 IQ prefill experts (STRATA_PF_FUSED) vs reference |
| tests/cuda/prefill_fused_moe_test.cpp | 420 | Fused int8 Q2_0 prefill MoE experts vs reference |
| tests/cuda/prefill_mmq_kquant_test.cpp | 172 | MMQ products for Unsloth UD-Q4_K / Q5_K / Q6_K expert formats |
| tests/hip/expert_cache_staging.cpp | 34 | HIP open_sized growth + reuse with pageable payloads |
| tests/hip/gdn_rec_head.cpp | 118 | Prompt GDN recurrence variant 1 (four lanes per column + norm) vs the column-split variant 0 (the default before it), bitwise |
| tests/hip/gemm_f16_io_parity.cpp | 112 | FP16-in/out prompt GEMMs (STRATA_HIP_PROMPT_F16) vs CPU |
| tests/hip/handoff.cpp | 122 | Async CPU/GPU handoff + graph replay; gfx1201 needs volatile ring store |
| tests/hip/intrinsics.cpp | 236 | Numerical parity of HIP compat-layer CUDA intrinsics |
| tests/hip/mapped_alias.cpp | 108 | hipHostGetDevicePointer alias usability (gfx1201/Windows context) |
| tests/hip/native_qsa_score.cpp | 95 | Device parity for optional native QSA scorer (HIP scalar) |
| tests/hip/ple_iq4.cpp | 202 | Real PLE IQ3_XXS/IQ4_XS: ggml CPU dequant vs HIP MMVQ |
| tests/hip/prefill_gemm.cpp | 93 | BF16/F16 prefill GEMM path vs CPU reference |
| tests/hip/prefill_hcd_exact_parity.cpp | 102 | STRATA_HCD_EXACT BF16 HCD down-proj vs hipBLASLt bit-for-bit |
| tests/hip/prefill_hipblaslt_gemm.cpp | 197 | Opt-in hipBLASLt tuning-table smoke (SKIP without STRATA_HIPBLASLT_TUNING) |
| tests/hip/prefill_mmq_parity.cpp | 324 | Standalone HIP MMQ prefill diagnostic harness |
| tests/hip/prefill_native_batch.cpp | 293 | Native QSA append + embedding gather parity (ported from q8atnight/Strata PR #108) |
| tests/hip/prefill_wmma_gemm_parity.cpp | 449 | RDNA3 WMMA GEMM numerical parity |
| tests/hip/q2_zero.cpp | 50 | Q2_0 signed-zero dequant (HIP folding bug catch) |
| tests/hip/router_fast.cpp | 101 | HIP fast router_top10 vs portable kernel, bitwise 65,536 rows |
| tests/platform/aux_cpus_test.cpp | 217 | --aux-cpus CPU-set derivation + live Linux thread placement |
| tests/platform/direct_file_async.cpp | 60 | DirectFile async I/O pool (STRATA_IO_THREADS) |
| tests/platform/host_limits_test.cpp | 34 | Windows HIP auto-chunk cap + commit-limit verdict |
| tests/prefill/share_rules_test.cpp | 56 | Default CPU share disarmed when every expert resident |
| tests/prefill/tuning_header_arch_family_test.cpp | 106 | hipBLASLt table header gate: RDNA4 (gfx1200↔gfx1201) and RDNA3 (gfx1100/1101/1102) siblings accept each other's tables (shared solution IDs); other pairs, incl. prefix-only gfx1010/gfx1030, stay rejected |
| tests/program/batch_read_test.cpp | 32 | batch_read_piece chunking for --batch prompts (synthetic) |
| tests/program/split_rules_test.cpp | 96 | Auto layer-split stage-size rule on synthetic 48-layer model |
| tests/data/native_experts/ (6 .txt files) | 49 each | Fixture expert ID lists: coder-iq1_m, iq2_xs, iq3_s, iq3_xxs, q2_0, swift-iq2_xs |

### 2c. docs/ (markdown + small assets)

| path | lines | purpose |
| --- | ---: | --- |
| docs/DETAILS.md | 2047 | Canonical measured numbers, API, settings, how-it-works (source of truth for speed) |
| docs/AMD_HIP.md | 684 | HIP build/validation; RDNA4 gfx1200/1201 section with R9700/9070 XT tables |
| docs/INTEL.md | 865 | Intel / SYCL deep notes |
| docs/A770_SPEED_SPEC.md | 761 | Arc A770 speed specification |
| docs/MULTI_GPU.md | 353 | Layer split, helpers, gfx12 pipeline-windows defaults |
| docs/BATCHING.md | 322 | Parallel slots, batch-groups, batch-mtp (incl. 2x/4x R9700) |
| docs/UNSLOTH_Q4.md | 316 | UD-Q4_K_XL workflow + measured slow path on 64 GB |
| docs/INSTALL.md | 300 | Full requirements; AMD gfx codes (gfx1200/1201 named) |
| docs/COMMUNITY_BENCHMARKS.md | 263 | How to run/share benches; index of community reports |
| docs/MAC.md | 212 | Apple Silicon Metal engine |
| docs/MCP_SERVER.md | 205 | MCP tools for install/start/stop |
| docs/AI_SETUP.md | 200 | Non-interactive setup script for coding agents |
| docs/NVFP4.md | 197 | NVFP4 pack path, RTX 5090/5070/3060 measurements |
| docs/TROUBLESHOOTING.md | 195 | Common failures (incl. gfx1201 long-prompt notes) |
| docs/QUANTS.md | 178 | Quant quality yardsticks; many R9700 KL/decode rows |
| docs/INTEL_ARC.md | 175 | Arc build-from-source |
| docs/STRIX_HALO.md | 175 | Ryzen AI Max gfx1151 |
| docs/SECOND_GPU.md | 173 | Helper / second GPU modes |
| docs/PREFILL_HEAD_ALIAS.md | 170 | Prefill head alias design |
| docs/MODELS.md | 157 | Model sizes, RAM fits, NVIDIA+AMD speed tables |
| docs/AMD_HIP_PERFORMANCE.md | 145 | Opt-in gfx1100 MMQ/hipBLASLt performance path (excl. Windows/gfx12 claims) |
| docs/UNSLOTH_Q6.md | 143 | UD-Q6 path notes |
| docs/EXCHANGE_ROTATION.md | 139 | Exchange rotation |
| docs/VRAM_ELASTIC.md | 131 | Elastic VRAM / segmented cache |
| docs/BATCHED_DMA.md | 114 | Batched DMA design |
| docs/ORCA.md | 96 | OrcaRouter uncensored notes |
| docs/NVIDIA_V100.md | 91 | Volta / SM70 |
| docs/IMAGES.md | 88 | Vision / images |
| docs/benchmarks/windows-zen2-gfx1030-workers.md | 86 | Zen2 + gfx1030 worker notes |
| docs/HOW_IT_WORKS.md | 77 | Plain-language architecture |
| docs/LOGIT_BIAS.md | 76 | Logit bias |
| docs/LLAMA_SWAP.md | 66 | llama-swap interop |
| docs/MESSAGE_BOUNDARY_CACHE.md | 66 | Message-boundary cache |
| docs/RESEARCH_RUNS.md | 60 | Research run notes |
| docs/DISJOINT_EXPERT_CACHE.md | 59 | Avoid duplicate helper/primary expert slots |
| docs/KV_PREFETCH.md | 57 | KV prefetch |
| docs/ORCA_Q4_K_S.md | 36 | Orca Q4_K_S |
| docs/PROMPT_CACHE_TAIL.md | 33 | Prompt-cache tail |
| docs/MMVQ_IL_TABLE.md | 32 | MMVQ IL table |
| docs/TEST_REQUESTS.md | 19 | Open hardware wanted for community tests |
| docs/OLDER_GPUS.md | 224 | P40/V100/GTX10/Radeon VII/MI50/etc. |
| docs/benchmarks/2026-09-29-gfx1100.json | 299 | RX 7900 XTX (gfx1100) control/optimized run dump |
| docs/benchmarks/2026-10-01-gfx1100-0.1.31.json | 298 | 0.1.31 vs 0.1.30 gfx1100 prefill/decode trials |
| docs/monitoring/prometheus.yml | 12 | Prometheus scrape sample |
| docs/monitoring/servicemonitor.yaml | 21 | K8s ServiceMonitor |
| docs/monitoring/grafana-strata.json | 1033 | Grafana dashboard (hit_rate panels etc.) |
| docs/media/ (5 files) | (bin/svg) | how-it-works / guess-and-check SVGs, pagoda assets; docs/media/make_figures.py |
| docs/paper/Strata-Paper.pdf | (PDF) | Paper; tiers.svg companion |

### 2d. bench/ (non-results)

| path | lines | purpose |
| --- | ---: | --- |
| bench/bench_vs_llama.py | 281 | Compare Strata vs llama.cpp throughput |
| bench/gfx906_qsa_query_swizzle_probe.cpp | 284 | gfx906 QSA query-swizzle probe |
| bench/test_gfx906_qsa_query_swizzle_guard.py | 130 | Guard/test for that probe |
| bench/micro/cuda_callback_affinity.cpp | 67 | CUDA callback affinity microbench |
| bench/profile_corpus.json | 47 | Profile corpus for routing/bench |
| bench/results/COMMUNITY.md | 123 | Tabular index of community result folders |
| bench/results/exchange-rotation-ab.json / bench/results/exchange-rotation-token-ids.json | 288 / 1 | Exchange-rotation A/B artifacts |

### 2e. bench/results — grouped summary (132 folders, 2,086 files)

Groups:

| Group | Example folders | What they say |
| --- | --- | --- |
| **Maintainer NVIDIA speed ladder** | `2026-09-24-final`, `2026-09-28-speed-0114`, `2026-09-29-speed-0122`, `2026-09-29-speed-0126` | RTX 5070 official matrices; 0.1.26 is what DETAILS quotes for NVIDIA (README: Q2_0 row 0.1.36, other rows 0.1.26) |
| **Prefill / KV / cache research** | `2026-09-28-prefill-speed`, `2026-09-27-kv-q4`, `2026-09-27-cache-parity`, `2026-09-27-esp`, `2026-09-28-rope-scaling` | Prefill 572→1290 tok/s (Q2_0 32K); KV-q4 tradeoffs; esp = the experimental-speed-projection control vector, a refusal-direction projection (bench/results/2026-09-27-esp/README.md:1-5); cache-parity = expert cache on vs off, teacher-forced logits (bench/results/2026-09-27-cache-parity/README.md:1-9); rope/needle |
| **Multi-GPU / layer-split** | `2026-09-29-layer-split`, `2026-09-29-layer-split-limits`, `2026-09-29-rtx3090-epyc-milan`, `2026-10-01-community-2x-rtx-pro-4500`, `2026-10-06-community-4x-tesla-p100` | Split planner, K-split sweeps, RTX 5080 + 3090 splits, single RTX 3090 / EPYC baseline, 2x RTX PRO 4500, 4x P100 (single RTX PRO 4500 and 2x P100 are in the NVIDIA community row) |
| **RDNA4 / gfx12** | `2026-10-02-community-rx9070xt-windows`, `2026-10-03-community-r9700-windows`, `2026-10-08-community-r9700-linux`, `2026-10-10-community-rx-9070`, `2026-10-10-community-rx-9060-xt` | R9700 Win IQ2_XS 855.8–1,231.7 prefill / 87.5–99.8 decode; R9700 Linux IQ3_S 632.9–768.1 prefill / 87.6–101.9 decode; RX 9070 XT Windows IQ3_S about 600 prefill / 45-46 decode (bench/results/COMMUNITY.md:12; bench/results/COMMUNITY.md:30); RX 9070 and RX 9060 XT community |
| **RDNA2/3 / gfx1030/1100** | `2026-10-03-community-rx7900xtx-rocm102-nightly`, `2026-10-04-community-rx-6800-windows`, `2026-10-05-community-2x-rx-6900xt`, `2026-10-04-community-rx-7900-gre`, `2026-10-04-rdna2-fp16-prompt`, `2026-10-01-0.1.31-release`; docs/benchmarks gfx1100 JSON | 7900 XTX / 6800 / 6900 / GRE community; 0.1.31 vs 0.1.30 RX 7900 XTX release run; RX 6900 XT FP16 prompt-GEMM study |
| **NVIDIA community** | `2026-09-30-community-rtx-5090`, `2026-10-04-community-rtx-4090-laptop`, `2026-10-04-community-rtx-5070ti-windows`, `2026-10-01-community-rtx-pro-4500`, `2026-10-03-community-v100`, `2026-10-05-community-2x-p40`, `2026-10-06-community-2x-p100` | RTX 5090, RTX 4090 Laptop, RTX 5070 Ti (Windows), RTX PRO 4500, V100, 2x P40 and 2x P100 community runs; see bench/results/COMMUNITY.md |
| **SYCL / Arc / Strix** | `2026-10-04-community-2x-arc-pro-b60`, `2026-10-05-community-arc-b65`, `2026-10-07-community-gfx1150`, `2026-10-05-community-gfx1151`, `2026-10-09-community-strix-halo-laptop-70w` | Intel Arc + Strix Point (gfx1150) / Strix Halo (gfx1151) |
| **Kernel micro** | `2026-10-08-gfx906-qsa-query-swizzle`, `2026-10-11-rtx-5090-callback-affinity`, `2026-10-07-v100-decode-kernels`, `2026-10-12-p40-sm61-kernels` | Targeted kernel A/Bs: gfx906 QSA query swizzle, RTX 5090 callback affinity, V100 decode kernels, P40 sm_61 kernels |
| **Vision / coder / rope extras** | `2026-09-24-vision`, `2026-09-28-coder`, `2026-10-09-rope-yarn-long-context` | Vision on vs off prompt/output speed and expert slots (bench/results/2026-09-24-vision/vision.md:1-3); the GSQ-RCO Coder (256 of 512 experts) on the RTX 5070 (bench/results/2026-09-28-coder/README.md:1-3); existing RoPE/YaRN recall at 512K and 1M (bench/results/2026-10-09-rope-yarn-long-context/README.md:1-5) |

## 3. ‘Where it does X’ index

Area tags: engine-core.md · gpu-cpu-kernels.md · prefill.md · moe-experts.md · server.md · intel-sycl-backend.md · apple-metal-backend.md · model-spec-ngram-platform.md · build-system.md

| Behavior | Evidence | Area |
| --- | --- | --- |
| Product speed claim (NVIDIA decode/prefill table) | README.md:27-33; docs/DETAILS.md:15-66; docs/MODELS.md:26-34 | server.md / model-spec-ngram-platform.md |
| Product speed claim (AMD RX 9070 XT) | README.md:37-41; docs/MODELS.md:36-43 | server.md / model-spec-ngram-platform.md |
| RDNA4 supported GPUs listed | README.md:58; docs/INSTALL.md:16; docs/INSTALL.md:70-71 (gfx1200/1201); docs/AMD_HIP.md:1-6; docs/AMD_HIP.md:270-272 | build-system.md |
| R9700/9070 XT 4K/16K validation (doplxyz's test machine) | docs/AMD_HIP.md:274-296 (R9700 982/45.5 @4K, 1402/48.3 @16K Coder) | gpu-cpu-kernels.md / prefill.md / moe-experts.md |
| gfx12 fused int8 prefill (STRATA_PF_FUSED) | docs/DETAILS.md:2028; docs/AMD_HIP.md:357-369; tests/cuda/prefill_fused_iq_test.cpp:1-2; tests/cuda/prefill_fused_moe_test.cpp:1-2; tests/hip/prefill_mmq_parity.cpp:1-4 (MMQ path) | prefill.md / gpu-cpu-kernels.md |
| gfx12 WMMA prompt attention (opt-in, not default) | docs/AMD_HIP.md:337-351; R9700 4K 1784→2427 tok/s | prefill.md / gpu-cpu-kernels.md |
| hipBLASLt table family gfx1200↔gfx1201 | tests/prefill/tuning_header_arch_family_test.cpp:1-20 | build-system.md / prefill.md |
| Expert cache hit_rate definition | docs/DETAILS.md:309-312; Grafana docs/monitoring/grafana-strata.json:775-776 | moe-experts.md / server.md |
| Oversized cache → high hit_rate but slow decode | docs/DETAILS.md:631-639 (91.4% hit but 13.7 tok/s vs 94 tok/s with free VRAM) | moe-experts.md |
| Disjoint helper/primary expert slots | docs/DISJOINT_EXPERT_CACHE.md:1-9 (~1900–2000 dup slots observed) | moe-experts.md |
| Layer split / pipeline on gfx12 | docs/MULTI_GPU.md:73-79; docs/MULTI_GPU.md:197-198; docs/MULTI_GPU.md:248-258; docs/AMD_HIP.md:311-329 | server.md / build-system.md |
| Batch groups on multi-R9700 | docs/BATCHING.md:89 (4x R9700 166 vs 86 total tok/s) | server.md |
| NVFP4 decode/prefill claims | docs/NVFP4.md:45-61; docs/NVFP4.md:99; docs/NVFP4.md:147-156; docs/NVFP4.md:189-190 (RTX 5090/5070; HIP decode OK, MMQ CUDA-only) | model-spec-ngram-platform.md / gpu-cpu-kernels.md / prefill.md |
| How MoE fits on PC | docs/HOW_IT_WORKS.md:12-30; README.md:210-227 | engine-core.md / moe-experts.md |
| Prefill chunk auto / MMQ history | bench/results/2026-09-28-prefill-speed/README.md:12-15; docs/DETAILS.md:332-338 | prefill.md |
| HIP handoff / mapped alias (gfx1201) | tests/hip/handoff.cpp:1; tests/hip/mapped_alias.cpp:1-4; docs/AMD_HIP.md:281-284 | gpu-cpu-kernels.md / build-system.md |
| Fast HIP router | tests/hip/router_fast.cpp:1-2; docs/AMD_HIP.md:331-335 | gpu-cpu-kernels.md |
| Community R9700 Windows long ctx | bench/results/2026-10-03-community-r9700-windows/README.md:1; bench/results/2026-10-03-community-r9700-windows/README.md:8-13 | server.md |
| Community R9700 Linux IQ3_S | bench/results/2026-10-08-community-r9700-linux/README.md:3-4; bench/results/2026-10-08-community-r9700-linux/README.md:24; bench/results/2026-10-08-community-r9700-linux/README.md:83-88 | server.md |
| Metal Mac path | docs/MAC.md:1-4; README.md:69; README.md:94-98 | apple-metal-backend.md |
| SYCL / Arc | docs/INTEL.md:1-4; docs/INTEL_ARC.md:1-3; docs/A770_SPEED_SPEC.md:1 | intel-sycl-backend.md |
| Installer / agent setup | docs/AI_SETUP.md:1-4; AGENTS.md:8-13 | build-system.md / server.md |
| Community bench harness | bench.sh:7; BENCH.bat:13; docs/COMMUNITY_BENCHMARKS.md:81-85 | server.md |
| Open test-hardware wishlist | docs/TEST_REQUESTS.md:1-4 | build-system.md |

## 4. Published benchmark numbers (most relevant)

All numbers copied from cited files; different engines/machines.

### Official docs and contributor validation

| Metric | Value | Source |
| --- | --- | --- |
| RTX 5070 Q2_0 decode / 32K prefill | 94 / 2,650 tok/s (README short table); DETAILS matrix 93.0 @4K decode, 2,171 @32K prefill | README.md:24-29; README.md:46; docs/DETAILS.md:17; docs/DETAILS.md:33-35; docs/DETAILS.md:52-54; bench/results/2026-09-29-speed-0126/README.md:10-12; bench/results/2026-09-29-speed-0126/README.md:20-22 |
| RTX 5070 IQ3_S | 53 decode / 1,620 prefill (README); matrix 53.3 @4K / 1,624 @32K | README.md:24-32; README.md:46; docs/DETAILS.md:17; docs/DETAILS.md:33-38; docs/DETAILS.md:52-57 |
| RTX 5070 Coder | 55 / 2,180 (README); 55.1 @4K / 2,177 @32K | README.md:24-33; README.md:46; docs/DETAILS.md:17; docs/DETAILS.md:33-39; docs/DETAILS.md:52-58 |
| RX 9070 XT Q2_0 / IQ2_XS / Coder (47 GB RAM box) | 60/1160; 52/1110; 44/1420 tok/s | README.md:24; README.md:37-41; docs/MODELS.md:36-43 |
| R9700 Coder IQ1_M (doplxyz test machine, ROCm 7.14) | 4K: 982 prefill / 45.5 decode; 16K: 1,402 / 48.3; cache 12,288 slots | docs/AMD_HIP.md:274-294 |
| RX 9070 XT Coder (same validation) | 4K: 782 / 30.8; 16K: 1,235 / 35.4; cache 4,931 slots | docs/AMD_HIP.md:274-295 |
| Prefill path 0.1.12→0.1.13 (RTX 5070, 32K) | Q2_0 572→1,290; IQ3_S 383→1,208 | bench/results/2026-09-28-prefill-speed/README.md:3; bench/results/2026-09-28-prefill-speed/README.md:22-25; docs/DETAILS.md:336-337 |
| NVFP4 vs IQ2_XS (RTX 5090 author fork) | decode 114–116 vs 132–139; 32K prefill 3,900 vs 5,600; ~8350 vs ~17300 cache slots | docs/NVFP4.md:44-52; docs/NVFP4.md:68 |
| Expert-cache size vs decode (Windows example: RTX 4080 SUPER 32 GB, IQ3_S, engine 0.1.39) | 11,631 slots / 0 MiB free → 13.7 tok/s @ 91.4% hit; 9,148 slots / 4 GiB free → 94.1 tok/s | docs/DETAILS.md:617; docs/DETAILS.md:627-639 |

### Community RDNA4 / gfx12 (bench/results + bench/results/COMMUNITY.md)

| Setup | Prefill tok/s | Decode tok/s | Source |
| --- | --- | --- | --- |
| R9700 Win, IQ2_XS, 0.1.38 | 856 (4K) / 1,182 (32K) / 1,232 (131K) / 1,130 (248K) | 99.8 / 97.8 / 89.9 / 87.5 | bench/results/2026-10-03-community-r9700-windows/README.md:3-13; bench/results/COMMUNITY.md:12; bench/results/COMMUNITY.md:33 |
| R9700 Linux, IQ3_S, exp. 0.1.40.3 | 633 / 768 / 751 / 718 (1K–131K) | 101.9 / 96.4 / 95.6 / 87.6 | bench/results/2026-10-08-community-r9700-linux/README.md:3-7; bench/results/2026-10-08-community-r9700-linux/README.md:24; bench/results/2026-10-08-community-r9700-linux/README.md:13; bench/results/2026-10-08-community-r9700-linux/README.md:83-88 |
| RX 9070, IQ3_XXS, 0.1.41 Win | 586 / 804 / 833 (4.2K/33.5K/130K) | 58.7 / 57.9 / 57.9 | bench/results/COMMUNITY.md:12; bench/results/COMMUNITY.md:117; bench/results/2026-10-10-community-rx-9070/README.md:106-110 |
| RX 9060 XT (gfx1200), IQ2_XS | 879 / 807 (8K / 28K, prefill 8,192) | 49.6 (coding, weighted) | bench/results/COMMUNITY.md:12; bench/results/COMMUNITY.md:120 |
| RX 9070 XT Win IQ3_S (earlier) | ~600 prefill | 45–46 decode | bench/results/COMMUNITY.md:12; bench/results/COMMUNITY.md:30 |
| R9700 WMMA opt-in | 4K 1,784→2,427; 16K 1,797→2,700 | decode unchanged / slightly down | docs/AMD_HIP.md:337-350 |
| R9700 STRATA_PF_FUSED (Coder) | 4K time −16% (1612→1353 ms; +19.1% speed as reported) etc. | not reported | docs/AMD_HIP.md:361-364 |

### Expert cache / hit rates

- **Definition:** `hit_rate` = VRAM share of experts looked up while answering; PCIe-copied experts (`--pcie-frac`) are excluded from the miss side, so raising PCIe share **raises** reported hit_rate even if decode slows (docs/DETAILS.md:309-312).
- **Example contradiction of intuition:** higher hit_rate with oversized cache coincided with **much slower** decode (91.4 / 94.7% hit at 13.7 tok/s and 0 MiB free vs 90.4 / 93.9% at 102 tok/s with `auto`) because free VRAM went to zero (docs/DETAILS.md:631-640).
- **R9700 Windows log excerpt:** auto cache 17,739→18,575 slots, “pre-filled 18575 of 18575 slots from the profile” (bench/results/2026-10-03-community-r9700-windows/README.md:1; bench/results/2026-10-03-community-r9700-windows/README.md:81-83) — full profile fill, not a runtime hit% table.
- **R9700 Linux product config:** auto **13,021 slots / 24.71 GiB** on IQ3_S (bench/results/2026-10-08-community-r9700-linux/README.md:3; bench/results/2026-10-08-community-r9700-linux/README.md:24; bench/results/2026-10-08-community-r9700-linux/README.md:57-61).
- **RDNA4 end-to-end (doplxyz's test machine):** R9700 **12,288 slots** (all Coder experts) vs 9070 XT **4,931** (docs/AMD_HIP.md:274-295; docs/AMD_HIP.md:316).
- **Tail-skip note:** Coder IQ1_M “94.8% of the experts cached, 99.9% decode hits” in one DETAILS route-tail discussion (docs/DETAILS.md:1660).

## 5. Gotchas and constants

1. **RDNA4 long context: docs vs community.** docs/AMD_HIP.md:372 marks “long contexts beyond 16K” and “answer-quality benchmarks” as **not validated** for RDNA4; yet community R9700 Windows reports to **248K** (bench/results/2026-10-03-community-r9700-windows/README.md:1; bench/results/2026-10-03-community-r9700-windows/README.md:13) and Linux to **131K** (bench/results/2026-10-08-community-r9700-linux/README.md:3; bench/results/2026-10-08-community-r9700-linux/README.md:24; bench/results/2026-10-08-community-r9700-linux/README.md:88).
2. **README AMD table vs AMD_HIP validation table.** README’s RX 9070 XT numbers (Q2_0 60/1160) are a different pack/setup than AMD_HIP’s Coder IQ1_M 4K/16K table (782/30.8 @4K). Same card family, not the same row (README.md:24; README.md:37-39; docs/AMD_HIP.md:287; docs/AMD_HIP.md:292-295).
3. **hit_rate is not “goodness”.** docs/DETAILS.md:639-640 explicitly warns high hit_rate with no free VRAM destroys decode.
4. **STRATA_PF_FUSED / WMMA are opt-in on gfx12 and change bits.** docs/AMD_HIP.md:337-369: WMMA changes greedy text from ~token 50; fused IQ3_XXS KL not always ≤ MMQ → stays opt-in. docs/DETAILS.md:2028 says fused IQ3 packs are **on by default** on gfx12 in a later note — tension with AMD_HIP “flag stays opt-in” for quality reasons.
5. **NVFP4 HIP gap.** docs/NVFP4.md:189-190: decode kernels portable; **NVFP4 MMQ / W4A8 unit CUDA-only**.
6. **Windows ready-made HIP zip.** bench/results/2026-10-03-community-r9700-windows/README.md:34-39 quotes docs/AMD_HIP.md:106: ready-made zip “has not run a model on a discrete card yet”; that community run is the report — still not “project-validated.”
7. **HIP greedy nondeterminism.** docs/AMD_HIP.md:370-371: ~1/10 starts differ at some greedy token; unexplained.
8. **ctest on R9700:** 42/45 pass; hipblaslt test skips without table; ple_parity / expert_multi need fixtures/AVX-512 (docs/AMD_HIP.md:281-283).
9. **bench/results scale:** 2,086 files; trimmed community folders carry a TRIMMED.md listing omitted raw logs/binaries (bench/results/COMMUNITY.md:3).

