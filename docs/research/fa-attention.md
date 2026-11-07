# Prompt attention on tensor cores (FA pilot)

Reference machine: RTX 3060 Ti 8 GB, 32 GiB RAM, KAT IQ4_XS (qwen35moe), q4_0 KV.
Branch `fa-mma`, base `main` 8ee1f346. Status: pilot, gated by G1-G5 below.

## Why

On a 96k prompt the cuBLAS attention branch (`q36_engine.cu`, the `T > kGemmT` path)
materialises the scores and slices the queries so that the score buffer fits: with
`--att-mb 256` a chunk of 2048 becomes ~43 slices at the tail of an 86k prefix, about
900 slices per layer and ~45 000 launches over 10 attention layers. The profile says
61.0 s for attention (10.3 TFLOPS, ~8 % of the 3060 Ti's ~130 TFLOPS FP16 tensor peak).
`--att-mb 512` (tq 96) recovers +5.5 % pp (1018.4 → 1074.0 at 96k N=4) — the narrow-N
symptom — but the rest is score materialisation and re-reading K/V per slice.

## What

`src/qwen35/q36_fa.cuh` — a flash-style prompt attention on `mma.m16n8k16`: one launch
per attention layer per sub-chunk, no score buffer, online softmax in the exp2 domain,
per-head max in the quad's registers (a first version kept it in shared and had a
read/write race — see the file comment), parametric causality `j <= p0 + i`, `oob` for
`P % kSlice != 0`. K and V share one staged buffer (`S.kv`); the barrier between the
score MMA and the V load makes the reuse safe.

Four tile variants (`STRATA_Q36_FA_TILE`): 4x32, 8x32, 4x64, 8x64 (queries x KV slice),
shared 41 216 / 65 536 / 64 256 / 94 720 B. The launcher re-derives the shape and
declines foreign geometry, so the caller keeps the cuBLAS branch.

## Wiring

`STRATA_Q36_FA_LAYER`: -1 = cuBLAS (default), N = layer N only, `all` = every attention
layer. `STRATA_Q36_FA_TILE` 0..3. `STRATA_Q36_FA_LOG=1` prints the launch line;
`STRATA_Q36_FA_CHECK=1` runs both branches on the FA layer and prints per-head
`max_abs` / `cos` against cuBLAS before the gate.

## Gates

| Gate | Metric | Expectation |
| --- | --- | --- |
| G1 | per-head `max abs` and `cos` vs cuBLAS-att on 4k, P=2048 and P=9 | `cos >= 0.9999`, `max abs <= 2e-2` |
| G2 | diagonal-query softmax sum vs cuBLAS | <= 1e-6 relative |
| G3 | greedy tokens at 4k and 96k | identical to base; near-ties logged by top-2 |
| G4 | 96k profile attention | <= 39 s on all layers (pilot target; ~12-20 s if 25-40 % peak) |
| G5 | needle 96k | hit no worse than base |

Numbers from these gates are added here when the pilot closes.
