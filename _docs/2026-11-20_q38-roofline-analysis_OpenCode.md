# Roofline and operation-count analysis of Q38 (2026-11-20)

Mathematical analysis, no GPU calls. Script:
`C:\Users\downl\AppData\Local\Temp\opencode\roofline.py`.

## Assumptions

- RTX 5060 Ti 16 GB: ~448 GB/s memory bandwidth, ~190 TFLOPS dense FP16
  tensor peak (derived from the published 759 TOPS INT8 class), SM120.
- 1024-token prefill, F16 KV, geometry from `admission.hpp`.
- FLOP count includes GDN qkv/gate/alpha/beta/ssm_out/FFN and attention Q/K/V/output/FFN.

## Results

| Metric | 9B | 27B |
|---|---|---|
| GEMM FLOPs (1024 tokens) | 14.44 TFLOP | 50.90 TFLOP |
| Compute roof (FP16 peak) | 76 ms | 268 ms |
| Weight-read roof | 11.3 ms | 25.0 ms |
| Attention K/V traffic, GQA | 2.15 GB | 2.15 GB |
| Attention K/V traffic, naive Q-per-KV | 8.60 GB | 12.90 GB |
| Attention traffic roof (GQA / naive) | 4.8 / 19.2 ms | 4.8 / 28.8 ms |
| Measured prefill (accepted build) | 534.4 ms | 522.6 ms |
| Achieved GEMM throughput | **27 TFLOPS (14% of peak)** | **97 TFLOPS (51% of peak)** |
| Decode roof (one weight pass) | 11.3 ms | 25.0 ms |
| Measured decode | 16.0 ms (316 GB/s, ~71% of peak) | not measured in this session |

## Interpretation

1. **Decode is bandwidth-bound and near practical peak** (71% of theoretical
   bandwidth). Only ~1.4x is available even with a perfect kernel; this matches
   the small gap to Prism (16.0 vs 14.8 ms/token).

2. **Attention memory traffic was the dominant *algorithmic* waste**, exactly as
   the GQA-fused kernel assumed: naive Q-per-KV loads 8.6 GB (9B) / 12.9 GB
   (27B) versus 2.15 GB with GQA sharing. The roofline saving is ~14 ms (9B)
   and ~24 ms (27B), consistent with the measured 6% (33 ms) on 27B once launch
   overhead is included.

3. **The 9B prefill is far from the compute roof** (14% utilization) while 27B
   reaches 51%. With a 3.5x smaller FLOP count but the same wall time, 9B is
   dominated by fixed overheads: dequantization of ~5 GB of weights into a
   20 MiB L2 tile window, repeated cuBLAS calls (848 per 1024-token prefill),
   and launch/fence latency. This is the mathematical reason tile-size sweeps
   (20-256 MiB) and call-count reductions (848 -> 248) produced no gain: the
   cost is in the dequantize-then-GEMM data movement, not in the tile count.

4. The only remaining large lever for prefill is therefore a **fused
   dequantize-in-MMA** kernel (llama.cpp MMQ style, Q8_1 activations, int8
   `mma.sync`), which avoids materializing the FP16 weight in global memory.
   Prior local pilots of fused WMMA and INT8 projection were 2-4x slower than
   cuBLAS (see `2026-11-18_q38-f16-resident-benchmark_OpenCode.md`), so this is
   a hypothesis requiring a new design, not a promised gain. FineQuant (2023)
   also reports weight-only GEMM losing to FP16 GEMM in compute-bound prefill.

## Conclusion

The accepted 5.4% / 6.0% improvements are consistent with the roofline: they
remove redundant attention traffic and transform work, not GEMM compute.
Further large prefill gains on 9B require changing the GEMM data path, which is
beyond a tuning change and has failed in earlier local pilots.
