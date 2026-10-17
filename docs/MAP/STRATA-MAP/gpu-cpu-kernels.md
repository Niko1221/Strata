# Strata map: GPU and CPU kernels (`src/kernels/`)

Source: Strata commit `296b59d2`. Paths are relative to the Strata repo root.

## 1. Overview

- `src/kernels/` holds every compute kernel of the Strata engine: `cuda/` (47 .cu/.cuh files, compiled as CUDA, or as HIP for AMD builds — there are no .hip files anywhere in src/), `cpu/` (AVX1/AVX2/AVX512 expert kernels + a thread pool for CPU-offloaded MoE experts), and 52 top-level `*_parity.cpp` / `*_bench.cpp` test/bench programs.
- Public interfaces live in `include/strata/kernels/*.hpp` (a same-named header for most .cu).
- Much arithmetic is transcribed from llama.cpp/ggml pinned at 3cf03257 so that GPU, CPU and ggml agree bit-for-bit.
- Ops: quantized GEMV (MMVQ on q8_1 activations), dequant, MoE router/top-k, grouped & shared experts, GDN (gated delta-net / linear attention), QSA attention (cache, indexer, select, prompt + decode attention), flash-attn, RMSNorm/gated-residual "GR", RoPE/mRoPE, PLE, KV quant (Q8/Q4) and KV streaming, sampler, speculative verify kernels.
- Total: 53,760 lines in src/kernels.
- Other .cu outside src/kernels (18 files) belong to engine-core.md (src/core/device.cu, src/core/pinned.cu, src/core/remote_expert_opt.cu) and prefill.md (src/prefill/*.cu: src/prefill/gemm.cu, src/prefill/kernels.cu, src/prefill/mmq_kq_lib.cu, src/prefill/wmma_gemm.cu, src/prefill/moe_fused_iq.cu, …) and qwen35-engine.md (src/qwen35/q36_engine.cu, src/qwen35/q36ref.cu) — not mapped in depth here.

## 2. File table (GPU, src/kernels/cuda)

| path | lines | purpose |
|---|---|---|
| src/kernels/cuda/iq_kernels.cu | 3824 | vec_dot_*_q8_1 dot products for i-quants/Q2_0/Q8_0/NVFP4, dequantizers, native gate-up/down MoE expert kernels |
| src/kernels/cuda/native_mmvq.cu | 2817 | llama.cpp MMVQ port: quantized GEMV for dense projections; per-arch rows-per-block choice |
| src/kernels/cuda/fused_gr.cu | 2333 | fused gated-residual / hyper-connection + norm kernels |
| src/kernels/cuda/verify_kernels.cu | 1528 | speculative-decoding verify (batched) kernels, GDN split, multi-block argmax |
| src/kernels/cuda/qsa_prompt_attn.cu | 1527 | QSA prompt (prefill) attention; tensor-core v2 kernel + RDNA4 WMMA int8-KV kernel |
| src/kernels/cuda/sampler.cu | 1393 | sampler chain penalties→top_k→top_p→min_p→temp→pick |
| src/kernels/cuda/qsa_select.cu | 1368 | QSA top-k cell selection; warp scorer + gfx11/12 WMMA scorer |
| src/kernels/cuda/s2_expert_grouped.cu | 1238 | grouped GPU expert (arithmetic mirrors src/kernels/cpu/expert.cpp) |
| src/kernels/cuda/qsa.cu | 821 | QSA cache, indexer and attention |
| src/kernels/cuda/qsa_decode_attn.cu | 760 | QSA decode attention (incl. gfx906 variants) |
| src/kernels/cuda/s_gemv.cu | 694 | naive S-family GEMV, one thread per row |
| src/kernels/cuda/gr.cu | 500 | gr_read / gr_write (gated residual) |
| src/kernels/cuda/router_top10.cu | 497 | MoE router: BF16 GEMV + softmax/top-k/renorm; AMD fast variant |
| src/kernels/cuda/elementwise.cu | 461 | glue kernels, doorbell publish (volatile store) |
| src/kernels/cuda/dequant_bf16.cu | 428 | dequant to BF16 (from ggml-quants.c) |
| src/kernels/cuda/shared_expert.cu | 419 | shared expert (ref/moe.py::shared_expert) |
| src/kernels/cuda/ple.cu | 390 | PLE block GPU path |
| src/kernels/cuda/native_bf16.cu | 383 | BF16 GEMV (mmvf) |
| src/kernels/cuda/native_qsa_indexer.cu | 375 | QSA indexer (set-rows/norm/rope) |
| src/kernels/cuda/quantize_act.cu | 352 | quantize activations to q8_1/Q8_0 like ggml src1 |
| src/kernels/cuda/native_router.cu | 329 | topk-moe port, 512-expert / 10-output |
| src/kernels/cuda/native_ple_postops.cu | 291 | PLE post-ops |
| src/kernels/cuda/kv_stream.cu | 286 | KV streaming |
| src/kernels/cuda/gdn.cu | 248 | gated delta-net non-projection parts |
| src/kernels/cuda/native_rope.cu | 245 | rope_multi / yarn (IMRoPE) + fused norm-rope |
| src/kernels/cuda/kv_q4.cu | 241 | Q4_0 KV with Walsh-Hadamard rotation |
| src/kernels/cuda/native_moe.cu | 229 | MoE weighted reduction |
| src/kernels/cuda/native_qsa_score.cu | 216 | QSA score (mmf/mma port) |
| src/kernels/cuda/native_flash_attn.cu | 212 | flash-attn vec (fattn-vec port) |
| src/kernels/cuda/native_gdn_preprocess.cu | 199 | GDN preprocess: norm, conv, scale |
| src/kernels/cuda/fused_gdn.cu | 186 | fused GDN |
| src/kernels/cuda/cvec.cu | 174 | control vectors |
| src/kernels/cuda/native_qsa.cu | 167 | QSA norm/unary |
| src/kernels/cuda/native_gr_postops.cu | 161 | GR post-ops |
| src/kernels/cuda/s2_gemv_fast.cu | 153 | experiment — header says measured negative/neutral |
| src/kernels/cuda/kv_q8.cu | 147 | Q8 KV |
| src/kernels/cuda/bf16_gemv.cu | 141 | BF16 GEMV |
| src/kernels/cuda/rope.cu | 136 | NEOX partial RoPE |
| src/kernels/cuda/native_gdn.cu | 124 | GDN recurrence (gated_delta_net.cu port) |
| src/kernels/cuda/native_gr_norm.cu | 109 | weighted F32 RMSNorm |
| src/kernels/cuda/q8_1_il.cuh | 99 | interleaved q8_1 helpers |
| src/kernels/cuda/s2_gemv_q8.cu | 96 | S2 (Q2_0 canonical) GEMV variant |
| src/kernels/cuda/s2_gemv_quads.cu | 93 | S2 (Q2_0 canonical) GEMV variant |
| src/kernels/cuda/s2_gemv.cu | 63 | S2 (Q2_0 canonical) GEMV variant |
| src/kernels/cuda/dequant_s2.cu | 51 | S2 (Q2_0 canonical) naive dequant |
| src/kernels/cuda/route_prior.cu | 79 | STRATA_ROUTE_PRIOR cache-conditioned routing (opt-in, changes output) |
| src/kernels/cuda/s26_tsum.cuh | 71 | tree-sum helper; permlanex16/DPP on AMD gfx10+ |

CPU files (src/kernels/cpu):

| path | lines | purpose |
|---|---|---|
| src/kernels/cpu/expert.cpp | 633 | CPU expert compute |
| src/kernels/cpu/expert_layout.cpp | 495 | ISA pick |
| src/kernels/cpu/native_expert.cpp | 281 | format dispatch |
| src/kernels/cpu/iq_avx1.cpp | 511 | IQ kernels, AVX |
| src/kernels/cpu/iq_avx2.cpp | 522 | IQ kernels, AVX2 |
| src/kernels/cpu/iq_avx512.cpp | 269 | IQ kernels, AVX-512 |
| src/kernels/cpu/kq_avx1.cpp | 71 | K-quant kernels, AVX |
| src/kernels/cpu/kq_avx2.cpp | 263 | K-quant kernels, AVX2 |
| src/kernels/cpu/q2_avx1.cpp | 294 | Q2_0 kernels, AVX |
| src/kernels/cpu/q2_avx2.cpp | 271 | Q2_0 kernels, AVX2 |
| src/kernels/cpu/nvfp4_avx2.cpp | 199 | NVFP4 kernels, AVX2 |
| src/kernels/cpu/nvfp4_avx512.cpp | 200 | NVFP4 kernels, AVX-512 |
| src/kernels/cpu/pool.cpp | 907 | thread pool |
| src/kernels/cpu/pool_affinity_linux.hpp | 87 | thread-pool CPU affinity, Linux |
| src/kernels/cpu/pool_affinity_win.hpp | 64 | thread-pool CPU affinity, Windows |

Tests: all `*_parity.cpp`, `*_test.cpp`, `*_bench.cpp` at src/kernels/ and src/kernels/cpu/ (tests-bench-docs.md area).

## 3. Where it does X

- **Arch detection** — include/strata/kernels/gfx_arch.hpp:16 `gfx_arch_is`, include/strata/kernels/gfx_arch.hpp:23 `gfx_arch_is_gfx11_wmma`, include/strata/kernels/gfx_arch.hpp:30 `gfx_arch_is_gfx12_wmma`, include/strata/kernels/gfx_arch.hpp:36 `gfx_arch_is_gfx12` (both = gfx1200||gfx1201), include/strata/kernels/gfx_arch.hpp:33 `gfx_arch_is_gfx1151`. String compare on `gcnArchName`; used across kernels, prefill.md and program.
- **gfx12 default switches** — src/core/arch_defaults.cpp:14-31 (engine-core.md). On gfx12, unless STRATA_GFX12_DEFAULTS=0, sets STRATA_GDN_HEAD=1, GDN_PP=2, GDN_CONVL2=1, GDN_NOY=1, CVEC_FUSE=1 ("prompt half" of the gfx1151 table; decode half measured slower on gfx1201). Called from src/program/generate.cpp:2252 `apply_arch_defaults` before switches are first read; user settings kept.
- **gfx12 prefill chunk** — src/program/generate.cpp:2256-2273 (generate-program.md). Bare `--prefill auto` on gfx12 Linux HIP raises chunk/auto_max to 16384; STRATA_GFX12_CHUNK=0 or STRATA_GFX12_DEFAULTS=0 keeps 8192. These two are the ONLY STRATA_GFX12_* env vars in the tree.
- **Quantized GEMV dispatch (MMVQ)**:
  - src/kernels/cuda/native_mmvq.cu:1270 `dev_class()` (1=gfx12, 2=sm_86) and `q8_0_rows1` just below.
  - Q8_0 dense projections use one row per block on gfx1201 when n_in<=4096 && n_out>=4096 (measured 76-86% of 644 GB/s); else two rows.
  - Override STRATA_NO_MMVQ_ROWS2 (1=one row everywhere, 0=two).
  - Other knobs: STRATA_MMVQ_WAVE src/kernels/cuda/native_mmvq.cu:1623, STRATA_Q8_PACKED src/kernels/cuda/native_mmvq.cu:1992, STRATA_Q6_PACKED src/kernels/cuda/native_mmvq.cu:2027, STRATA_MMVQ_IL_ROWS src/kernels/cuda/native_mmvq.cu:2678.
- **NVFP4 format/dot**:
  - src/kernels/cuda/iq_kernels.cu:570-600.
  - Block = 64 values = 4 sub-blocks of 16, one UE4M3 scale each, E2M1 codes split-half like Q4_0 (ggml `block_nvfp4`; ModelOpt is repacked by tools/nvfp4_convert.py).
  - `kvalues_fp4` holds doubled E2M1 values so `ue4m3_to_f32_half` halves the scale; decoded in software (NaN→0) for bit-exactness with ggml-cpu.
  - Dot is dp4a against q8_1 activations (w4a8), template `Fmt<40>` at src/kernels/cuda/iq_kernels.cu:655, `Split<40>` src/kernels/cuda/iq_kernels.cu:1007, sub-warp K=20 paths src/kernels/cuda/iq_kernels.cu:1120, src/kernels/cuda/iq_kernels.cu:1167 for n_ff=640.
- **NVFP4 per-expert global scale** — src/kernels/cuda/iq_kernels.cu:1348-1352 & src/kernels/cuda/iq_kernels.cu:1391, src/kernels/cuda/iq_kernels.cu:1431. The blob tail (`L.tail_off`) holds FP32 s_gate/s_up/s_down; applied to FP32 outputs, s_down on the down output (folding into up made hidden ~1e-5 → FP16-subnormal q8_1 scale).
- **NVFP4 prefill (hand-off prefill.md)** — include/strata/prefill/moe_mmq.hpp:23-27 `nvfp4_mode()`, STRATA_PREFILL_NVFP4: w4a8 default (int8 MMQ), w4a4 Blackwell FP4 only, fp16 dequantize + FP16 GEMM reference.
- **CPU NVFP4 experts (hand-off moe-experts.md)** — src/kernels/cpu/native_expert.cpp:206, src/kernels/cpu/native_expert.cpp:213, src/kernels/cpu/native_expert.cpp:260, src/kernels/cpu/native_expert.cpp:265 (STRATA_NO_NVFP4_512/256 pick AVX512 vs AVX2), src/kernels/cpu/nvfp4_avx2.cpp:51, src/kernels/cpu/nvfp4_avx512.cpp:52 (STRATA_NVFP4_PREFETCH). ISA forced via src/kernels/cpu/expert_layout.cpp:29 STRATA_FORCE_ISA, src/kernels/cpu/expert_layout.cpp:43 STRATA_FORCE_AVX2.
- **Attention (prompt, RDNA4)** — src/kernels/cuda/qsa_prompt_attn.cu:1131-1181 WMMA kernel on `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`, int8 KV only; gate `hip_wmma_usable()` src/kernels/cuda/qsa_prompt_attn.cu:1404 needs STRATA_HIP_WMMA=1 and gfx12/gfx11 (output differs in last bits). Fragment layout documented at src/kernels/cuda/qsa_prompt_attn.cu:1136.
- **QSA select (RDNA4)** — src/kernels/cuda/qsa_select.cu:288-330 BF16 WMMA scorer, `sel_gfx12_device()` src/kernels/cuda/qsa_select.cu:447, opt-in STRATA_SELECT_WMMA=1 at src/kernels/cuda/qsa_select.cu:1129 (selects slightly differently from warp scorer). Hardcoded `kRegMinBlocks = 7168` src/kernels/cuda/qsa_select.cu:1333 tuned on gfx1201.
- **Router** — src/kernels/cuda/router_top10.cu:267 AMD bit-identical variant measured on gfx1201; switches STRATA_HIP_ROUTER_FAST/OLD src/kernels/cuda/router_top10.cu:429, src/kernels/cuda/router_top10.cu:431, STRATA_ROUTER_BLOCK src/kernels/cuda/router_top10.cu:462. src/kernels/cuda/native_router.cu: 512 experts / top-10 path.
- **GDN / linear attention** — src/kernels/cuda/gdn.cu, src/kernels/cuda/native_gdn.cu, src/kernels/cuda/native_gdn_preprocess.cu, src/kernels/cuda/fused_gdn.cu; state layout in include/strata/kernels/gdn.hpp; verify-time split STRATA_GDN_SPLIT / STRATA_GDN_COMMIT_SPLIT src/kernels/cuda/verify_kernels.cu:1024-1094.
- **Norm/RoPE** — src/kernels/cuda/native_gr_norm.cu (RMSNorm), src/kernels/cuda/native_rope.cu:154 STRATA_ROPE_TABLE, src/kernels/cuda/native_rope.cu:177, src/kernels/cuda/native_rope.cu:180 STRATA_NO_NORM_ROPE / STRATA_NORM_ROPE (fused norm+rope).
- **Sampling** — src/kernels/cuda/sampler.cu (llama.cpp order), STRATA_ARGMAX_MULTI src/kernels/cuda/sampler.cu:1256; verify argmax STRATA_MULTI_BLOCK_ARGMAX src/kernels/cuda/verify_kernels.cu:930.
- **Build (build-system.md)** — CMakeLists.txt:348 `strata_kernels` STATIC library of the .cu sources; CMakeLists.txt:59 same .cu sources compiled as HIP for gfx906 via a compat layer. HIP build = same sources, `__HIPCC__` / `__HIP_PLATFORM_AMD__` guards.
- **See also:** engine-core.md (src/core/device.cu, arch_defaults, include/strata/core/emulate.hpp, include/strata/core/geometry.hpp), prefill.md (src/prefill: MMQ, src/prefill/wmma_gemm.cu:52, src/prefill/moe_fused_iq.cu:1033 picks gfx12 WMMA), moe-experts.md (CPU experts/pool), generate-program.md (src/program/generate.cpp), tests-bench-docs.md (parity programs), model-spec-ngram-platform.md (qwen35, PLE/QSA semantics from ref/*.py).

## 4. Key types and env vars

- Types: `block_nvfp4`, `block_q8_1`, `block_q8_0` (ggml layouts, used src/kernels/cuda/iq_kernels.cu); `Fmt<TD>` / `Split<TD>` traits (TD 40 = NVFP4, 42 = Q2_0) src/kernels/cuda/iq_kernels.cu:655, src/kernels/cuda/iq_kernels.cu:1007; `SmallTraits<Q80Block,8>` (src/kernels/cuda/native_mmvq.cu).

| env var | where |
|---|---|
| `STRATA_GFX12_DEFAULTS` | src/core/arch_defaults.cpp:21, src/program/generate.cpp:2263 |
| `STRATA_GFX12_CHUNK` | src/program/generate.cpp:2262 |
| `STRATA_HIP_WMMA` | src/kernels/cuda/qsa_prompt_attn.cu:1406 |
| `STRATA_SELECT_WMMA` | src/kernels/cuda/qsa_select.cu:1129 |
| `STRATA_NO_MMVQ_ROWS2` | src/kernels/cuda/native_mmvq.cu:1297 |
| `STRATA_TSUM` | src/kernels/cuda/native_mmvq.cu:1160, src/kernels/cuda/iq_kernels.cu:3657, src/kernels/cuda/fused_gr.cu:1907 |
| `STRATA_EXPERT_V2 / V2K` | src/kernels/cuda/iq_kernels.cu:3654, src/kernels/cuda/iq_kernels.cu:3672 |
| `STRATA_EXP_MODE` | src/kernels/cuda/iq_kernels.cu:3593 |
| `STRATA_IQ_STAGE_GRID` | src/kernels/cuda/iq_kernels.cu:2137 |
| `STRATA_IQ_STAGE_GRID_MMVQ` | src/kernels/cuda/iq_kernels.cu:2144 |
| `STRATA_DEQUANT_WIDE` | src/kernels/cuda/dequant_bf16.cu:269 |
| `STRATA_S2_SWIGLU_Q8` | src/kernels/cuda/s2_expert_grouped.cu:1157 |
| `STRATA_OLD_GROUPED` | src/kernels/cuda/s2_expert_grouped.cu:529 |
| `STRATA_FUSED_SWIGLU_Q81` | src/kernels/cuda/shared_expert.cu:207 |
| `STRATA_DEC_BATCH` | src/kernels/cuda/shared_expert.cu:246 |
| `STRATA_PROMPT_ATTN_Q4 / _V1` | src/kernels/cuda/qsa_prompt_attn.cu:1500, src/kernels/cuda/qsa_prompt_attn.cu:1517 |
| `STRATA_PA_FAST` | src/kernels/cuda/qsa_prompt_attn.cu:1433 |
| `STRATA_DF_PDL` | src/kernels/cuda/verify_kernels.cu:1460 |

The full list is every `getenv("STRATA_…")` call under src/kernels.

## 5. Gotchas / surprises

- No .hip files: AMD runs the CUDA .cu sources through HIP; arch specialization is by `#if defined(__gfx1200__)||defined(__gfx1201__)` and runtime gcnArchName checks.
- RDNA4 WMMA paths (prompt attn, select) are OPT-IN and not bit-identical to default; default decode on gfx1201 is dp4a MMVQ, not matrix cores.
- NVFP4 decode is w4a8 (q8_1 activations), not FP4×FP4; FP4 MMA only on Blackwell prefill.
- src/kernels/cuda/elementwise.cu:206-212: on RDNA4 a plain store to mapped pinned memory can sit in L2 until stream sync — store is volatile.
- src/kernels/cuda/verify_kernels.cu:1402: wall_clock64 assumed 100 MHz on gfx10.3/11/12.
- src/kernels/cuda/s2_gemv_fast.cu is named "fast" but its header says both levers measured negative/neutral.
- src/kernels/qsa_prompt_attn_parity.cpp:327: RDNA4 kernel takes int8 KV only; FP16-KV check skipped on AMD.
- Hardcoded tuning: kRegMinBlocks=7168 (src/kernels/cuda/qsa_select.cu:1333), gfx1201 MMVQ thresholds (src/kernels/cuda/native_mmvq.cu:1266), prefill 16384 (src/program/generate.cpp:2267).
