# State-of-the-art technique applicability to Q38 (2026-11-20)

Primary sources checked this date. Scope: CUDA 13.5 / SM120 / RTX 5060 Ti,
Qwen3.5 dense 9B/27B with head_dim 256 and 3:1 GDN:attention layers.

## FlashAttention-4 on SM120

Sources: FA4 paper (arXiv 2603.05451, MLSys 2026), Dao-AILab PRs #2336/#2348/
#2349/#2484, `SecondNatureComputing/flash-attn-4-sm120` README.

Verified facts:
- SM120 has **no tcgen05/TMEM, no WGMMA**, only single-CTA TMA and SM80-era
  `mma.sync.aligned.m16n8k16`; 99 KiB shared memory per block.
- FA4's main speed path (TMEM, 2-CTA MMA) therefore does not apply. The SM120
  bundle reaches only 1.04-1.10x over FA2 at S=512..4096, and is 1.71x slower
  at S=128 due to dispatch overhead.
- **Critical limitation: head_dim > 128 is unsupported on SM120** because the
  99 KiB SMEM budget cannot hold the Q tile. Qwen3.5-9B/27B use D=256.
- Split-KV is clamped to 1 on SM12x in the bundle.

Decision: FA4 cannot be used for these checkpoints. Our own BR-tiled kernel is
the appropriate design point for D=256 on this hardware. This also explains
why the 55% gain came from barrier/tile restructuring rather than MMA.

## Gated DeltaNet chunkwise (WY) algorithms

Sources: Gated Delta Networks (Yang et al., ICLR 2025, arXiv 2412.06464),
ReWY (arXiv 2611.01940), Systolic Linear Attention (arXiv 2611.06828).

Facts:
- GDN admits a chunkwise parallel algorithm: within a chunk of size C the
  delta-rule interactions are expressed as dense matmuls using the WY
  (Householder) representation, with a sequential state scan across chunks.
- This is a **training/prefix-parallel** formulation. For autoregressive decode
  the recurrence remains sequential; for prefill it could replace the current
  per-token fused recurrence with matmuls.

Applicability assessment:
- The recurrence cost is only ~1.0 ms/step decode (`2026-11-19_q38-gdn-graph-profile...:42-46`)
  and ~46-73 ms per 1024-token prefill, versus 179-252 ms projection GEMMs.
- A chunkwise rewrite would change the FP32 accumulation order and requires new
  WY matrices and a state-scan kernel; it is a research-grade change with a
  small measurable target. The rank-1 update is dependency-coupled
  (`kv -> delta -> S -> o`), so naive mma.sync does not help (confirmed by the
  GDN analysis subagent).
- Decision: not recommended now; record as a future direction if GDN ever
  becomes the dominant stage.

## What the expert analysis endorses

1. Keep the BR-tiled GQA attention (already 1.554x on 9B) as the main path.
2. Keep decode CUDA graphs, including the newly merged attention QKV graph.
3. Do not pursue FP8/NVFP4, tcgen05, or GDN mma.sync on this GPU:
   narrow-precision hangs, no hardware support, or too little arithmetic.
4. The remaining prefill bottleneck is dequant+cuBLAS data movement; a fused
   dequant-in-MMA kernel is the only large lever, but prior local pilots were
   1.9-4.5x slower and FA4's SM120 experience confirms the hardware ceiling.
