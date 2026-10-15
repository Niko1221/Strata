# Community benchmark: 2x RTX 2080 Ti on a slow dual-socket CPU — single vs split, concurrency, and vLLM

Measured on 2026-10-16 on an Ubuntu 24.04 server. This fills three gaps the earlier 2x2080 Ti
reports left open: a single-GPU vs layer-split comparison on the same cards, concurrent-request
throughput with `--batch`, and a same-workload comparison against vLLM on the same hardware.

## Hardware and software

- **GPUs:** 2x NVIDIA GeForce RTX 2080 Ti, 22,528 MiB reported VRAM each, NVLink (P2P measured
  44 GB/s). Driver 580.173.02, CUDA 13.0. Power limit 180 W for most of the session, then raised
  to 250 W (the card default) for the last table.
- **CPU and RAM:** dual-socket Intel Xeon E5-2696 v4, 2x 22 physical cores (44 cores / 88 logical
  CPUs), AVX2 only, DDR4. 94 GiB RAM, 8 GiB swap. GPU 0 sits on NUMA node 0, GPU 1 on node 1.
- **Storage:** model on a local WD Green SN350 NVMe (DRAM-less).
- **Engine:** Strata `main` @ v0.1.43.1, locally compiled for SM 75.
- **Model:** Qwen3.8-Flash-Next `unsloth/UD-IQ4_XS` (3 shards, SHA-256 verified), 131,072-token
  context, INT8 KV, `--vision` on, `--expert-cache auto`, `--prefill auto`, `--spec 4 --spec-min-p 0.5`,
  bundled MTP, `--resident-budget-gib 55`.
- **Comparison engine:** vLLM fork (`weicj/vLLM-2080Ti-Definitive`), vLLM `0.29.1rc0` + torch
  2.13.0+cu130, the same model family in FP8 (`Qwen3.8-27B-FP8`), MTP 3, TP=2, 256K, FP8 KV.

## Method

- Decode timing is the engine's own per-request log (`prompt N tokens = ... generated in Y ms`),
  read with the harness's `ab_run.py`. Single-request numbers are the median of 3 repeats of each
  prompt size (424 / 2,944 / 11,582 tokens, 200 generated).
- Concurrent throughput is aggregate completion tokens from the OpenAI-compatible API divided by
  wall time, with the same 2,944-token prompt repeated across the concurrent requests and the
  prompt cache warmed first, so it is decode-bound. GPU utilisation and power are `nvidia-smi`
  sampled at 4 Hz.
- Quality: the 31-case suite (instruction following, Chinese, math, long-context needle 128K, tool
  calls, vision) run against the final config.

## Canonical suite (`tools/strata_bench.py --long --clients 4 --runs 3`)

The repo's own benchmark tool, run against this install on the final config (2-GPU layer split +
`--batch 4 --batch-groups 2`, engine 0.1.43.1). Medians with ranges; every request is in
`results.json`.

| Configuration | Prompt tok/s | Decode tok/s | TTFT s |
| --- | ---: | ---: | ---: |
| short chat | — | 42.8 (31.9–49.7) | 0.67 |
| 4K prompt, fresh | 678.6 (618.7–683.8) | — | 5.96 |
| 16K prompt, fresh | 1102.4 (1080.2–1111.1) | — | 14.62 |
| 32K prompt, fresh | 1298.1 (1294.1–1303.7) | — | 24.76 |
| agent session, follow-up turns | — | 50.3 (39.0–65.5) | 5.26 |
| 4 clients, 128 new tokens each | — | 41.3 total (37.6–45.5) | 2.68 p50 |

## Results — single GPU vs layer split (`"gpu": [0, 1]`)

| metric | single GPU | layer split |
| --- | ---: | ---: |
| decode 424 / 2,944 / 11,582 tok/s | 28.4 / 29.8 / 31.3 | **52.4 / 57.9 / 57.3** |
| prompt 424 / 2,944 / 11,582 tok/s | 77 / 365 / 512 | **206 / 632 / 955** |
| window | 70.06 ms | **37.96 ms** |
| CPU pool time | 9.52 ms | **0.79 ms** |
| GPU-reach wait | 28.32 ms | 15.79 ms |

The decode gain is almost entirely the CPU pool collapsing (9.52 -> 0.79 ms): on a CPU this slow,
the second card does not just add memory — it removes expert work the CPU pool was the bottleneck on.

## Results — concurrency and `--batch 4 --batch-groups 2`

The layer split alone is a serialised two-stage pipeline (one hand-off per window), so a single
request leaves each card ~50% idle and concurrent requests only interleave, they do not fill the
pipeline. `--batch 4 --batch-groups 2` packs several requests' tokens into one verify window.

| concurrency | layer split only | + `--batch 4 --batch-groups 2` |
| ---: | ---: | ---: |
| 1 | 22.3 | 36.7 |
| 2 | 60.4 | 57.0 |
| 4 | 62.2 | **75.1** |
| 8 | 64.1 | **82.9** |

At concurrency 4/8 the batch config reaches 83%/99% GPU utilisation and 175-177 W (the 180 W cap).
Quality is 31/31 with batch enabled.

## Results — power limit 180 W vs 250 W

Raised the limit to the card default (250 W). vLLM does **not** move; Strata does:

| concurrency | Strata `--batch 4` @ 180 W | Strata `--batch 4` @ 250 W | vLLM @ 250 W |
| ---: | ---: | ---: | ---: |
| 1 | 36.7 | 43.9 | 37.1 |
| 2 | 57.0 | 62.7 | 37.1 |
| 4 | 75.1 | **90.8** | 37.1 |
| 8 | 82.9 | **92.0** | 37.0 |
| 16 | — | 93.1 | — |

vLLM saturates its SMs at concurrency 1 (86-90% util) and produces the same 37 tok/s at 180 W and
250 W (drawing 210-244 W): it is compute/latency-bound on SM 75, where FP8 has no hardware support
and is emulated. Strata is latency-bound first, fills the GPU with concurrency + `--batch`, and
only then does the power limit bind — raising it adds +21% at concurrency 4.

## Strata vs vLLM, same cards, same prompt

| concurrency | Strata + `--batch 4` (250 W) | vLLM (250 W) | Strata / vLLM |
| ---: | ---: | ---: | ---: |
| 1 | 43.9 | 37.1 | 1.18x |
| 4 | 90.8 | 37.1 | **2.45x** |
| 8 | 92.0 | 37.0 | **2.49x** |

## Limitations

- One host, one quant (UD-IQ4_XS). The vLLM side runs the FP8 weights, so the comparison is the
  deployment a user would actually make with each engine, not a quant-matched one.
- The `--batch` and 250 W tables are single batches, not the 10-round protocol of the earlier
  community reports; single-request decode numbers are medians of 3.
- No thermal soak; the long 128K needle cases were run at 180 W only.
- vLLM was started for this comparison, then stopped; both engines never ran simultaneously.
