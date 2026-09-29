# Prefill and decode speed (2026-09-29, branch prefill-decode-perf)

Qwen3.8-Flash-Next Q2_0 on the Tesla V100-PCIE-16GB (`sm_70`), Ryzen 5 3600, 48 GB
DDR4-3200, CUDA 12.8. Each row is a separate, uncached (`reused=0`)
OpenAI-compatible chat-completion request with a unique-prefix repeated-text
prompt and 64 generated tokens; the engine's own timings come from `/metrics`
(`bench/run_v100_bench.py`). All rows measured 2026-09-29 on the same machine
and binary set (engine 0.1.22 + these commits).

## What changed

| Change | Area | Effect |
| --- | --- | --- |
| Batch the verify window's per-token kernels (upstream PR #109, cherry-picked onto the fork) | decode | One launch per window for the router, indexer projections, RoPE/norm/gate, attention and MoE rows instead of one launch per token per layer (`STRATA_DEC_BATCH=0` restores the old loop); bit-identical |
| Tile the QSA block scorer's key reads across queries | prefill + decode | The indexer scored each 512 B pooled key once per (query, block) pair - `O(nq * n_bid)` DRAM traffic, ~54 TB over a 256K prompt. Each warp now holds its key row in registers and scores a tile of 8 queries against it; the per-(query, block) arithmetic is unchanged, so the scores are bit-identical (`qsa_parity --selftest` passes, including the tie-cut selection check). Microbench at the engine's real shapes (256 queries x 65K blocks): 3.5 ms -> 1.3 ms per call |
| `--spec 8` in the machine config | decode | Longer draft windows amortize the fixed per-window traffic (same engine, +3.5% on an identical 64-token request; draft floor stays 0.70 - floors 0.55/0.45 measured slower) |

## Results

### Prefill (prompt tok/s)

| | main, same day (2026-09-29) | this branch | delta |
| --- | ---: | ---: | ---: |
| ~4K | 681.5* | 1073.1 | *(see note) |
| ~8K | 557.3* | 1064.0 | *(see note) |
| 32K | 576.2* | 1271.2 | *(see note) |
| 128K | 805.5 | 1036.1 | +28.6% |
| 256K | 396.3 | 466.7 | +17.8% |

The * short-row baselines were measured right after the two long rows, on a
hot card; the passively-cooled V100 swings the short rows by ~50% with
temperature, so the fair short-row comparison is the published 2026-09-28
table, which this baseline binary reproduced (989.1 / 966.5 / 1136.8 /
659.3 / 414.1). Against those cool-card rows this branch reads +8.5% / +10.1%
/ +11.8% at 4K / 8K / 32K. The long rows were re-measured from the same
cool-card start on both builds (the published 128K row measured low on a warm
session; 659.3 -> 1036.1 is +57%, 414.1 -> 466.7 is +12.7%).

### Output (tok/s), same rows

| | same-day baseline | this branch | delta |
| --- | ---: | ---: | ---: |
| ~4K | 47.7 / 49.9* | 56.9 / 54.7 | *(see note) |
| ~8K | 45.3* | 55.5 | *(see note) |
| 32K | 45.6* | 50.6 | *(see note) |
| 128K | 33.6 | 37.1-38.2 | +10-14% |
| 256K | 34.7 | 38.5-40.4 | +11-16% |

The * short-row baselines are the hot-card same-day rows; against the
published cool-card rows (49.3 / 50.6 / 50.5) this branch reads 56.9 / 55.5 /
50.6 in the same conditions. The 128K/256K decode gain comes roughly 2/3 from
the batched verify windows and 1/3 from the tiled scorer.

## Where prefill time goes now (`STRATA_PREFILL_TIMING`, 24K prompt)

| stage | share |
| --- | ---: |
| qsa attn (FP32 fallback kernel on sm_70) | 24.4% |
| gemm gate/up + gemm down + dequant (MoE experts) | 36.2% |
| gdn (dense recurrent layers) | 13.8% |
| hc read | 7.8% |
| qsa select (scores + topk, after the tiling) | 2.3% |

The block-scorer tiling removed the long-context collapse (that stage is now
2.3% of the prompt); the remaining flat tax is the prompt attention, which on
Volta falls back to the FP32 split-K kernel because the tensor-core
(`mma.m16n8k16`) prompt attention requires sm_80. Porting it to Volta's
`mma.m16n8k8` is the next step.

## Quality

- `qsa_parity --selftest`: 0 failures (the modified kernel's scores are
  bit-identical; the reference selection and tie-cut checks pass).
- Full parity suite: qsa, kv_q4, kv_q8, kv_stream, bf16_gemv, s_gemv,
  s_gemv_q8k, s2_gemv, s2_gemv_q8, router_top10, gr, gdn, native_expert
  (against the real shard), ple, cvec, dequant_s2, quantize_act, rope,
  elementwise, sampler, shared_expert, prefill_gemm: all pass.
  (iq_parity reports missing fixtures: the test needs data files this checkout
  does not carry - pre-existing, unrelated to these changes.)
- `tests/diagnose_openai.py`: 5/5 pass; `tests/diagnose_hermes_flow.py`:
  complete native-tool workflow passes.
- The batched verify and the tiled scorer are both bit-identical by design
  (same arithmetic per element); the selection tie-cut test in qsa_parity pins
  the scorer to the reference exactly.

## Raw rows

- `2026-09-29-baseline-long/`, `2026-09-29-baseline-short/`: main binary, same order.
- `2026-09-29-new-long/`, `2026-09-29-new-short/`: this branch, spec 4.
- `2026-09-29-cool-short/`: this branch, spec 8, cool card (the published-row comparison).
- `2026-09-29-final-long/`, `2026-09-29-final-s8-128k/`: this branch, spec 8.
