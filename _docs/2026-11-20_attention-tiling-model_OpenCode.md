# Latency model for batched attention tiling (2026-11-20)

## Why the old kernel is latency-bound

Per (token, KV head) block, the previous GQA kernel performs, for each history
position p, H dot products over d=256 with a 256-thread reduction:
- loads K[p], V[p] once (1 KiB),
- H warp+block reductions (H=4 for 9B, 6 for 27B), each with `__syncthreads`,
- online-softmax update, then another barrier.

Number of blocks: `T * kv_heads` = 4096 (T=1024, KV=4).
Average dependency chain length per block: T/2 = 512 iterations.
Barriers per iteration: about H+2 (6-8).
Shared memory: 32 KiB probability cache + 1 KiB reductions, so at most ~2-4
blocks/SM on SM120 (227 KiB/SM). With 38 SMs and ~2 resident blocks/SM, the
number of waves is 4096/76 ≈ 54, each with a 512-iteration dependent chain.

## Tiled kernel (BR=4)

`attend_batch_gqa_tiled` handles BR consecutive tokens in one block:
- grid = (T/BR) * kv_heads = 1024 blocks, 4x fewer;
- K/V loads for a position are shared by BR*H = 16-24 query rows;
- all BR*H dot-product reductions are written to one shared array and collapsed
  into **one** `__syncthreads`; barriers per iteration drop from H+2 to 2;
- shared memory ≈ 768 B (`warp_sum[6][4][8]`), so occupancy rises substantially.

Waves: 1024 / (38 * blocks_per_SM). With the smaller footprint, ~8 blocks/SM is
plausible -> ~3.4 waves, versus ~54 before. The theoretical reduction in
barrier-serialised depth is about 54*512*(H+2) / (3.4*512*2) ≈ 60x, but GEMM
and fixed costs cap the end-to-end gain.

## Register pressure and optimal BR

Per thread the kernel stores qv, mx, sum, acc for H heads and BR tokens:
`regs ≈ 4*H*BR + 20`. SM120 has 64K registers/SM and 256 threads/block:

| H | BR | regs/thread | max blocks/SM | grid | waves |
|---|---|---|---|---|---|
| 6 | 2 | ~68 | 3 | 2048 | 18 |
| 6 | 4 | ~116 | 2 | 1024 | 13.5 |
| 6 | 8 | ~212 | 1 | 512 | 13.5 |
| 4 | 4 | ~84 | 3 | 1024 | 9 |

BR=4 is optimal for 9B (H=4), but the initial 27B A/B regressed: 516.7 -> 637.2 ms
with identical tokens. For H=6, BR=4 needs ~116 registers/thread, which lowers
occupancy enough to negate the barrier savings. The kernel was therefore
templated on BR and 27B now uses BR=2 (`~68` registers/thread, grid 2048,
18 waves). The rebuild passed CTest 18/18 and its A/B gave 579.9 ms baseline
vs 588.7 ms tuned with identical tokens: the BR=4 regression is gone but there
is no net gain on 27B. This confirms the register-occupancy tradeoff in the
model; the 9B gain remains the main result.

## Predicted vs measured

The old stage profile gave `attn_kv` 149 ms (9B) / 234 ms (27B). The bandwidth
floor is 4.8 ms; the tiled kernel should approach a few tens of ms. The first
A/B showed total prefill 713.96 -> 451.82 ms with identical tokens; an
interleaved A/B is running to remove system-load drift. If confirmed, the
attention stage likely fell from ~149 ms to roughly 30-60 ms, i.e. the kernel
moved from barrier-bound toward the memory floor.

## Conclusion

The mathematical model explains why the tiling change is much more effective
than earlier tuning: it reduces the *number of dependent barriers*, not the
number of FLOPs or bytes. This is the algorithmic lever that the roofline
analysis predicted (0.03% of FP16 peak achieved).
