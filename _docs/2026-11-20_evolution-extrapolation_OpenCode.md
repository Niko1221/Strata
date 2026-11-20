# Extrapolating the optimisation trajectory (2026-11-20)

## Observed generations and fitness

| Gen | Change | Mechanism | 9B prefill | Gain |
|---|---|---|---|---|
| G0 | baseline | dequant+cuBLAS, per-token attention | 563-714 ms | - |
| G1 | softmax probability cache | remove 256x redundant expf | ~558 ms | 1.2% (noise) |
| G2 | RoPE table + memset | remove constant recomputation / write | 534 ms | 5.4% cumulative |
| G3 | GQA-fused attention | share K/V across 4-6 Q heads | 523 ms (27B) | 6.0% on 27B |
| G4 | BR-tiled attention | collapse H*BR reductions into 1 barrier | 445 ms | **1.554x cumulative** |
| G5 | decode attn QKV graph | fewer graph launches | decode 15.3 ms | 8.3% decode |
| G6 | `score_mb=64` (evolution) | sub-batch granularity | held-out 3/4 shapes | opt-in |

The gains are not additive in a simple way: G4's 55% dominates because it
changed the *dependency depth*, not the FLOP count.

## Roofline-based ceiling for the current data path

Measured after G4 (9B, 1024 tokens): total ~445 ms.
Stage decomposition before G4: GEMM ~368 ms, attention ~149 ms, GDN ~46 ms.
After G4, attention is estimated ~60 ms (the 1.554x total gain is consistent
with attention falling from 149 to ~60 ms).

Attention bandwidth floor = 2.15 GB / 448 GB/s = **4.8 ms**.
Remaining attention headroom ≈ 55 ms (about 12% of total prefill).
Even a perfect attention kernel would bring 9B prefill to ~390 ms.

For reference, PrismML/llama.cpp measured 393 ms on the same 1024-token fixture.
**Conclusion: perfecting attention alone only reaches parity, not superiority.**

## Where a >2x gain must come from

The GEMM stage is the only remaining large term:
- 14.44 TFLOP in 368 ms = 27 TFLOPS = **14% of FP16 peak**.
- Doubling utilisation to 28% would save ~184 ms -> total ~260 ms, about
  1.5x faster than Prism. Reaching 50% (the 27B level) gives ~200 ms total,
  roughly 2x faster than Prism.

The theoretical mechanism is a fused dequantize-in-MMA GEMM (llama.cpp MMQ
style Q8_1 activations + int8 mma.sync), which removes the FP16 weight
materialisation and second global read. Prior local pilots of fused WMMA and
INT8 projections were 1.9-4.5x slower than cuBLAS, and FA4's SM120 experience
confirms the hardware ceiling (no tcgen05, 99 KiB SMEM). So this is a
*research-grade mutation*, not a tuning knob.

## Extrapolated next mutations (ranked by expected value)

1. **Tile Q/K/V in shared memory with `mma.sync.m16n8k16` and online softmax.**
   Target: attention 60 ms -> 10-20 ms; total 445 -> 395 ms. Risk: high
   implementation cost, but the hardware supports the instruction and our own
   BR tiling already proved the dependency-depth hypothesis.
2. **Fused dequant + GEMM (MMQ-style).** Target: GEMM 368 -> 200 ms; total
   ~280 ms, beating Prism by ~1.4x. Risk: previous pilots failed; needs a new
   layout and exhaustive numeric validation.
3. **Chunkwise GDN WY.** Target: GDN 46 -> 10 ms. Low value (8% of total) and
   high complexity; defer.
4. **Further graph fusion.** Decode 15.3 ms is already bandwidth-bound
   (71% of peak); at most ~1.4x available. Low value.

## Diminishing-returns law observed

Each generation moved a different bottleneck: recomputation -> memory sharing
-> dependency depth -> launch overhead. The remaining terms are hardware
data-path limits. Extrapolation of the *measured* trend (5%, 6%, 55%, 8%)
is meaningless; the roofline model, not the trend, determines the ceiling.
The honest forecast is: ~390 ms with perfect attention (parity with llama.cpp),
~260-280 ms only with a new GEMM data path (1.4-1.5x faster).
