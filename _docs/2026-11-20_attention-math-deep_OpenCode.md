# Deep mathematical analysis of Q38 attention (2026-11-20)

Script: `C:\Users\downl\AppData\Local\Temp\opencode\attention_math.py`.

## Complexity

For T=1024 causal tokens, Q heads q, head dim d=256, GQA KV heads 4:

- Score/value FLOPs = 2 * [T(T+1)/2] * q * d * 2 = 8.60 GFLOP (9B) /
  12.90 GFLOP (27B).
- Compressed K/V traffic = [T(T+1)/2] * 4 * 256 * 2 bytes = 2.15 GB for both.
- Compute roof at 190 TFLOPS = 0.045 / 0.068 ms.
- Bandwidth roof at 448 GB/s = 4.80 ms.
- Measured `attn_kv` = 149.2 ms (9B) / 233.7 ms (27B).

**Achieved arithmetic throughput = 0.058 / 0.055 TFLOPS = 0.03% of peak.**
The kernel is neither compute-bound nor bandwidth-bound; it is **latency-bound
by a long serial dependency chain**.

## Why the current kernel is slow (structural, not tuning)

`score_batch` / `attend_batch_gqa` assign one block per (token, head) and loop
over all positions p=0..pos. Each iteration performs a 256-thread warp
reduction with `__syncthreads`, then (in the GQA version) 4-6 sequential
reductions. With 4096 blocks, ~32 KiB shared memory and SM120 occupancy limits,
only a fraction of blocks are resident; each executes ~500 dependent iterations.

Roofline comparison: 2.15 GB / 448 GB/s = 4.8 ms if the K/V loads were
streamed; 149 ms means a ~31x latency penalty from serialisation and low
occupancy. This explains why the earlier local changes (shared probability
cache, GQA load sharing) gave 5-6% but could not close the gap: they reduce
work, not the dependency depth.

## Tiled MMA (FlashAttention-style) design space

With SM120 shared memory 227 KiB per block, a tile of Q block BR and KV block
BC at head_dim 256 needs `2*256*(BR + 2*BC)` bytes. Feasible pairs:

| BR | BC | shared (KiB) |
|---|---|---|
| 32 | 64 | 80 |
| 64 | 64 | 96 |
| 128 | 64 | 128 |
| 32 | 128 | 144 |
| 64 | 128 | 160 |
| 128 | 128 | 192 |

A block handling BR=64 query tokens and looping over 8 KV tiles of BC=128
would give (1024/64)*q = 256 (9B) / 384 (27B) blocks, each with only 8 MMA tile
iterations instead of ~500 scalar iterations. That reduces the dependency depth
by ~60x and should approach the 4.8 ms bandwidth roof.

SM120 has no tcgen05/TMEM (NVIDIA CCCL issue 8146), so the tiles must use
warp-level `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (4096 FLOP per
instruction) with FP16 operands in registers and K/V staged through shared
memory. Online softmax keeps the running max/sum and rescales the accumulator,
which is numerically different from the current two-pass softmax but valid
under the 3e-5/4e-5 tolerance contract.

## GDN recurrence (Qwen3.5-specific)

The gated-delta recurrence is a rank-1 state update with d_v=d_k=state=128:
per layer/token ~2*128^2*2 = 65k FLOP; for 48 layers and 1024 tokens = 3.22
GFLOP (27B), i.e. 0.017 ms at peak. The measured `gdn_core` is ~46-73 ms, again
latency/launch bound, but its absolute FLOP content is ~1/16,000 of the GEMM
work. Optimising the recurrence arithmetic cannot yield large gains; the
projection GEMMs (already Tensor Core) dominate.

## Conclusion

The single largest mathematical opportunity in this engine is replacing the
per-token scalar attention loops with a tiled `mma.sync` online-softmax kernel.
It has a theoretical 4.8 ms floor versus 149-234 ms today. This is a new kernel
family, not a tuning knob, and requires careful numerical validation. It is
recorded here as the top-priority algorithmic direction.
