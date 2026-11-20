# FP8 / NVFP4 / MXFP4 feasibility for Q38 (2026-11-20, OpenCode)

User asked to try FP8 and NVFP4/MXFP4. This record separates API availability
from a proven speedup.

## Verified API facts (CUDA 13.5 toolkit installed on this machine)

- `H:/Strata/toolchains/cuda-13.5.0/toolkit/include/library_types.h:87-93`:
  `CUDA_R_8F_E4M3` (FP8 E4M3) and `CUDA_R_4F_E2M1` (NVFP4) are defined.
- `cublasLt.h:937-963`: block-scaling modes exist, including
  `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`, `...VEC32_UE8M0`,
  `...VEC128_32F` and MN-K4 variants — the API surface for MXFP8/MXFP4 GEMM.
- CUDA 13.5 cuBLAS guide (web, 2026-11-20): block-scaled FP8/FP4 matmul is
  documented; the same guide warns that narrow-precision forward compatibility
  is limited and new hardware may be slower despite higher theoretical peaks.

## Existing implementation in Q38

- FP8 prefill is implemented behind `STRATA_FP8_PREFILL=1`
  (`src/qwen38/prefill_gemm.cu:892-974`): dequant FP16 tile -> block-scaled FP8
  quant -> cuBLASLt GEMM -> rescale. It requires `n_tokens % 16 == 0`.
- Measured 9B (1024 tokens, aligned chunk): FP8 616 ms with 784 FP8 GEMMs
  versus FP16 565-585 ms — **slower, not adopted**.
- 27B first attempt used 10 prompt tokens, so `n_tokens % 16 != 0` and the FP8
  path was not taken (`GEMM_STATS fp16=2592 fp8=0`, 914.5 ms).
- 27B aligned 32-token rerun on the GQA build: FP8 `fp8=2592`, 729.030 ms,
  versus FP16 `fp16=2592`, 740.861 ms on the same prompt — about 1.6% in
  FP8's favour, within single-run noise and far smaller than the 5-6% from the
  numerical algorithmic changes. FP8 remains opt-in and unadopted.
- The NIM-driven Shinka v2 evolution was stopped because the NIM endpoint
  returned repeated HTTP 429/529 and MiniMax failed to produce extractable
  patches; no evolved configuration win was demonstrated.

## Why NVFP4/MXFP4 is not a drop-in replacement

1. The model weights are Q4_K, Q6_K, IQ2_S, IQ3_S and PQ2_0, not FP4. A
   conversion to NVFP4 with 16-element block scales would need a new quantizer
   and would change the numerical semantics beyond the 3e-5/4e-5 contract
   unless error-bounded and validated.
2. No FP4 path exists in `PrefillGemmWorkspace`; only FP16 and FP8 are wired.
   Implementing block-scaled FP4 GEMM would require new cuBLASLt descriptors,
   scale layouts and a correctness oracle.
3. The literature result (FineQuant, 2023; llama.cpp MMQ) says weight-only
   low-precision GEMM helps memory-bound decode, not compute-bound prefill —
   the opposite of the intended target.
4. Hardware risk: NVIDIA forum post 384753 reports GPU hangs on this exact
   RTX 5060 Ti under sustained low-precision tensor-core workloads, including
   FP8 GEMM via cuBLASLt. Adopting FP4 for a production path is not justified
   without a long soak test.

## Decision

FP8 remains an opt-in diagnostic and is not enabled by default. NVFP4/MXFP4 is
API-available but not implemented; a proof-of-concept would need a new
quantizer, a new cuBLASLt plan, a numeric oracle and a stability soak before it
could be considered. The accepted speedups in this session come from geometry
and algorithmic changes that preserve numerics (5.4% on 9B, 6.0% on 27B).
