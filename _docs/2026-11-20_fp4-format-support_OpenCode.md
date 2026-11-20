# NVFP4 / MXFP4 / FP8 format support (2026-11-20)

## What was added

- `include/strata/qwen38/admission.hpp`: `matrix_type()` now accepts GGUF types
  39 (MXFP4) and 40 (NVFP4). Payload sizes come from the shared
  `block_geometry()` table, which already defines 39 = 32 values / 17 bytes and
  40 = 64 values / 36 bytes.
- `src/qwen38/prefill_gemm.cu`: `dequant_elem()` gained E2M1 decode and FP4
  block-scale handling:
  - MXFP4: one E8M0 scale (2^(e-127), e in 1..254) plus 16 bytes of packed
    4-bit E2M1 codes per 32 values.
  - NVFP4: four unsigned E4M3 sub-block scales (16 values each) plus 32 bytes
    of packed E2M1 codes per 64 values.
  - E2M1 magnitude table {0, 0.5, 1, 1.5, 2, 3, 4, 6} with sign bit.
  - `launch_dequant_tile()` dispatches types 39/40 through the generic
    scalar-per-element kernel, so FP4 weights are dequantized to FP16 and use
    the existing cuBLAS Tensor Core prefill path.

FP8 (CUDA `CUDA_R_8F_E4M3`) was already supported internally through the
opt-in `STRATA_FP8_PREFILL=1` block-scaled cuBLASLt GEMM path in
`prefill_gemm.cu`. It is measured slower on this GPU and remains opt-in.

## Verification status

- Build `H:/Strata/builds/q38-9b-fp4-20261120` (CUDA 13.5 / SM120 / five jobs)
  passed CTest 18/18.
- Regression check on the pinned 9B fixture (1024 tokens, f16, seed 42):
  strict build token `15217` / 417.4 ms, FP4-support build token `15217` /
  410.2 ms (within noise). No behavioural regression on existing Q4_K models.
- No real NVFP4/MXFP4 GGUF fixture was available in this workspace, so the
  dequantizer is implemented from the documented block layout but **not**
  numerically validated against an external reference. Do not use it for
  model-quality claims until such a fixture is obtained.
- The decode (`matvec`) path does not yet dispatch types 39/40; FP4 support is
  currently for the prefill GEMM path only.

## Why not native block-scaled FP4 GEMM

CUDA 13.5 exposes `CUDA_R_4F_E2M1` and cuBLASLt scale modes
`CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` / `VEC32_UE8M0`, but:
- no FP4 GGUF fixture exists here to validate against;
- prior INT8/FP8 Tensor Core pilots on this exact GPU were 1.9-4.5x slower than
  cuBLAS, and the same RTX 5060 Ti has a field-reported hang under sustained
  low-precision Tensor Core load;
- the shared-227 KiB / no-tcgen05 SM120 constraints limit tile sizes.

Therefore the first supported path is dequantize-to-FP16 + existing Tensor
Core GEMM, which is conservative and numerically transparent.
