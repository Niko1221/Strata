# Q38 algorithmic acceleration plan (2026-11-20, OpenCode)

## Objective

Reduce the 9B/27B native prefill and decode cost by changing the algorithms,
not by switching numeric formats or driver knobs. Companion research:
`_docs/2026-11-20_cuda135-tensorcore-deepresearch_OpenCode.md`.

## Current measured cost (9B, 1024-token prefill, ctx 2048, f16 KV)

| Stage | ms | Algorithm in use |
|---|---|---|
| gdn_mm | 179 | dequant -> tiled FP16 GEMM for qkv / gate / alpha / beta / ssm_out / FFN |
| gdn_core | 46 | fused conv+L2 + recurrent delta step |
| attn_mm | 18 | Q/K/V projections |
| attn_kv | 149 | batched score -> global score buffer -> batched softmax/aggregate |
| ffn | 171 | dequant -> FP16 GEMM for gate, up, down + SiLU |
| total | 618 | |

Decode: ~16.0 ms/token, dominated by reading ~5 GB of compressed weights per
token (bandwidth bound), versus Prism 14.84 ms/token.

## Algorithmic changes, ordered by value / risk

### 1. Fused GQA prefill attention with shared K/V loads (high value, medium risk)
Current `score_batch` launches one block per (token, Q head), so for 16 Q heads
and 4 KV heads each K/V history vector is read four times, and scores are
written to and read from a 32 MiB global buffer.
Change: one block per (token, KV head), 256 threads. Keep `heads_per_kv` Q
vectors in registers/shared, load each K and V history vector once, compute the
`heads_per_kv` dot products with one warp reduction each, and run an online
softmax (running max/sum/weighted accumulator) per Q head. No global score
buffer, no separate score kernel, 4x less K/V traffic.
Expected: attn_kv 149 ms -> roughly 40-60 ms. Numerics differ in summation
order, so validate against the current path with forced-token logits at
tolerance 3e-5/4e-5 before adopting.

### 2. Fused dual projection GEMMs for prefill (medium value, low risk)
Decode already uses `matvec_q8_1_dual/triple` for gate+up and Q/K/V, but prefill
calls `matmul_batch` once per projection. Build one dequantized FP16 tile that
concatenates gate and up (2 x ffn rows) and issue one GEMM, then apply SwiGLU.
Same for GDN qkv+gate+alpha+beta where shapes permit. This raises the GEMM N
dimension and halves the number of tiling passes over the activation buffer.
Expected: 5-15% on gdn_mm + ffn. Numerics are unchanged (same dot products).

### 3. Recurrent-state blocking for GDN (low value, low risk)
`gdn_core` is only 46 ms of 618 ms. Increasing the recurrence block size in
`fused_gdn_step_norm_silu_batch` trades registers for fewer launches. Do not
prioritise until items 1 and 2 are measured.

### 4. MTP speculative decoding for 27B (high value for decode, file-dependent)
The 27B checkpoint has 15 real `blk.64.nextn.*` tensors and the engine already
reports `mtp_ready=1` with `--spec-draft-n-max 1`. Measure accepted-token rate
with the exact serve protocol before claiming a decode speedup. The local 9B
GGUF declares MTP but has no predictor tensors, so this cannot apply to 9B
without a different checkpoint.

### 5. Bandwidth reduction for decode (medium value, high risk)
Decode is bounded by reading compressed weights. Packing Q4_K blocks into a
layout better matched to 128-bit loads, or using a persistent FP8 copy (~5 GB
extra, fits 16 GB), could help, but FP8 was measured slower here and the exact
GPU has a field-reported hang under sustained FP8/INT8 Tensor Core loads. Not
recommended without Nsight Compute evidence.

## Validation gates for any change

1. Five-job build and CTest 18/18, without relaxing tolerances.
2. Forced-token FP32 logits comparison against the current binary on both 9B
   and 27B; report max absolute/relative error.
3. Same 1024-token fixed prompt, warm, three runs, median prefill time.
4. 27B regression tokens must remain `4183,344,248046` with `--seed 42`.
5. Resource floors unchanged; no competing user process stopped.

## Status

Item 1 is the next implementation target. No source changes for this plan are
included yet; the current working tree contains the already-validated batched
attention and head-geometry fixes from the previous records.
