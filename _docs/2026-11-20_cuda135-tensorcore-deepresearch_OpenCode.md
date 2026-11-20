# CUDA 13.5 / Tensor Core deep research for Q38 (2026-11-20, OpenCode)

## Objective and scope

Determine whether the Q38 native runner can be made faster than the installed
PrismML/llama.cpp on this RTX 5060 Ti (SM120) using CUDA 13.5 Tensor Core
features, and select a safe implementable change. Scope: 9B and 27B dense
`qwen35`, prefill and decode, Windows, CUDA 13.5.37 / cuBLAS 13.1, driver state
as found on 2026-11-20. Source quality: NVIDIA official docs/issues/forums and
the ggml/llama.cpp repository were treated as primary; secondary commentary
only for context.

## Verified facts from primary sources (checked 2026-11-20)

1. **SM120 consumer Blackwell has no `tcgen05.mma`.** The NVIDIA CCCL issue
   [#8146](https://github.com/NVIDIA/cccl/issues/8146) (opened 2026-03-24,
   maintainer reply 2026-04-13) states that on SM120 consumer GPUs the only
   Tensor Core path is warp-level `mma.sync`; the cited FP4 instruction is
   `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32...`.
   Therefore CUTLASS TCGen05 / TMEM techniques aimed at datacenter SM100 do not
   apply to this GPU.

2. **CUDA 13.5 cuBLAS supports block-scaled FP8/FP4 GEMM**, documented in the
   [cuBLAS 13.5 guide](https://docs.nvidia.com/cuda/cublas/index.html)
   (sections 3.1.4.4/3.1.4.5: 16/32/128-element 1D block scaling for FP8 and
   FP4). The same guide warns that forward compatibility for narrow precisions
   (FP4/FP8) is limited and that new hardware may be slower despite higher
   theoretical peaks.

3. **This exact GPU has a documented low-precision hang signature.** NVIDIA
   developer forum post
   [384753](https://forums.developer.nvidia.com/t/rtx-5060-ti-sm120-gpu-engine-hang-device-loss-under-sustained-low-precision-tensor-core-workloads-reproducible-in-2-5-minutes/384753)
   (2026-09-30, drivers 591.86/616.56/616.92, Windows 11 25H2) reports
   reproducible GPU engine hang/device loss under sustained int8 attention and
   `torch._scaled_mm` FP8 GEMM (cuBLASLt/CUTLASS backend), with a boot-scoped
   aftereffect. BF16 SDPA / cuBLASLt BF16 GEMM and int8 GEMM negative controls
   showed zero events. This is a field report, not an NVIDIA-confirmed root
   cause, so it is treated as a risk signal, not a proven driver bug.

4. **llama.cpp uses dequantize + cuBLAS for large-batch prefill**, not the MMQ
   quantized kernel. In
   [`ggml-cuda/mmq.cu`](https://github.com/ggml-org/llama.cpp/blob/665dac0f110f730af851ab27bdb597ab1d94e3ad/ggml/src/ggml-cuda/mmq.cu)
   `ggml_cuda_should_use_mmq()` returns true on NVIDIA with FP16 MMA hardware
   only when `ne11 < MMQ_DP4A_MAX_BATCH_SIZE` (small batch / decode); larger
   batches go through the dequant + BLAS path. It also shows native MXFP4/NVFP4
   MMQ only when `blackwell_mma_available(cc)` and the weight is already FP4.
   Implication: our dequant + cuBLAS architecture matches llama.cpp's prefill
   strategy; the remaining gap is kernel/implementation efficiency, not a
   fundamentally wrong approach.

5. The current Q38 prefill already uses Tensor Cores: `GEMM_STATS` reports 848
   FP16 tensor-op GEMM calls for a 1024-token 9B prefill, with FP16 accumulate
   for quantized weight types (`CUBLAS_COMPUTE_16F`, FP16 output tile scaled to
   FP32). Sweeps of tile buffer (20/64/128/256 MiB) and chunk (1024/512/256)
   changed call count 848 -> 248 but not time (571-581 ms), so call count and
   dequant tile count are not the limiting factor.

## Measured baseline (this machine, 2026-11-20)

- 9B 1024-token prefill, ctx 2048, `--kv f16`: FP16 565-585 ms warm; FP8
  (`STRATA_FP8_PREFILL=1`, aligned 1024 tokens) 616 ms with 784 FP8 GEMMs —
  slower, not adopted.
- 9B 140-token comparison: Prism ML 0.4.1 prompt 204.25 ms / decode 14.84
  ms/token; native prefill 305.08 ms / decode 16.03 ms/token.
- Stage profile after batched attention: `gdn_mm=179.00 attn_mm=18.19
  attn_kv=149.22 ffn=171.11 fence=1.02` ms. GEMM work is about 370 ms of 618 ms.

## Inferences (not directly proven here)

- The remaining ~1.5x prefill gap is most likely in the dequantized-weight GEMM
  implementation and the GDN/FFN projection schedule, not in launch overhead
  (already fixed) or in KV attention (already 8x faster).
- A direct Q4_K -> `mma.sync` (warp-level) Tensor Core GEMM that keeps weights
  compressed and dequantizes in registers could reduce both memory traffic and
  GEMM time, but it is a new kernel family, not a tuning knob. llama.cpp does
  not use such a kernel for large-batch prefill, so it is not a proven path to
  beat llama.cpp either.
- Persistent FP16 dequantized weights would need ~10 GB for 9B (and ~22 GB for
  27B), which does not fit alongside the 5-11 GB compressed weights and the
  768 MiB reserve in 16 GB VRAM. Persistent FP8 (~5 GB for 9B) fits but was
  measured slower and carries the hang risk in fact 3.

## Decisions

- Do not adopt FP8/FP4 for production on this GPU. It is slower in the measured
  path, SM120 support is limited to `mma.sync` intrinsics via inline PTX, and
  there is a reproduced field failure signature on this exact card.
- Do not adopt TCGen05/TMEM techniques: unavailable on SM120 (fact 1).
- Keep the already-implemented batched f16 attention and head-geometry fix from
  `_docs/2026-11-20_q38-tensorcore-tuning_OpenCode.md` as the safe speedup.
- Any further attempt to beat llama.cpp should start with a microbenchmark of
  the existing dequant + cuBLAS GEMM against ggml's dequant + cuBLAS path on the
  exact projection shapes, before writing new kernels.

## Unresolved uncertainty / skipped checks

- Did not build or run a `mma.sync` FP4/FP8 prototype: high implementation cost
  and the documented hang risk; not justified without a GEMM microbenchmark.
- Did not profile with Nsight Compute: not available in this session. Absolute
  tensor-core utilization therefore remains unmeasured.
- Did not test a newer driver branch: the forum report lists 591.86/616.56/616.92
  all affected; changing system drivers during a coding session is out of scope.
- The llama.cpp commit fetch returned HTTP 503; the MMQ facts above come from
  the raw source at the same pinned revision, which was fetched successfully.

## Sources

- https://docs.nvidia.com/cuda/cublas/index.html (CUDA 13.5 cuBLAS guide, checked 2026-11-20)
- https://github.com/NVIDIA/cccl/issues/8146 (SM120 mma.sync issue, opened 2026-03-24)
- https://forums.developer.nvidia.com/t/rtx-5060-ti-sm120-gpu-engine-hang-device-loss-under-sustained-low-precision-tensor-core-workloads-reproducible-in-2-5-minutes/384753 (2026-09-30)
- https://github.com/ggml-org/llama.cpp/blob/665dac0f110f730af851ab27bdb597ab1d94e3ad/ggml/src/ggml-cuda/mmq.cu
- https://developer.nvidia.com/blog/new-cublas-12-0-features-and-matrix-multiplication-performance-on-nvidia-hopper-gpus (secondary context for cuBLASLt heuristics)
