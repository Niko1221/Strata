# Community benchmark: 2x RTX 2080 Ti on a slow dual-socket CPU — single vs split, concurrency, and vLLM

Measured on 2026-10-24 on an Ubuntu 24.04 server, engine 0.1.49 built from source for SM 75. This fills
three gaps the earlier 2x2080 Ti reports left open: a single-GPU vs layer-split comparison on the same
cards, what `--batch` actually buys on concurrent requests, and a same-workload comparison against vLLM
on the same hardware.

## Hardware and software

- **GPUs:** 2x NVIDIA GeForce RTX 2080 Ti, 22,528 MiB reported VRAM each, NVLink (P2P measured
  44 GB/s, not used by the engine). Driver 580.173.02, CUDA 13.0. **Power limit 180 W** for every table
  except the one marked 250 W, which is an earlier run on this host at the card default.
- **CPU and RAM:** dual-socket Intel Xeon E5-2696 v4, 2x 22 physical cores (44 cores / 88 logical
  CPUs), AVX2 only, DDR4. 94 GiB RAM, 8 GiB swap. GPU 0 sits on NUMA node 0, GPU 1 on node 1.
- **Storage:** model on a local WD Green SN350 NVMe (DRAM-less).
- **Engine:** Strata `main` @ `06e51dfa`, engine **0.1.49**, locally compiled for SM 75. `src/`,
  `include/` and `serve/` are identical between that commit and the commit this branch is rebased on
  (`git diff --stat 06e51dfa..8183c6b6 -- src/ include/ serve/` is empty; only the CMake version string
  moves to 0.1.49.1), so these numbers measure the current engine sources.
- **Model:** Qwen3.8-Flash-Next `unsloth/UD-IQ4_XS` (3 shards, SHA-256 verified), 131,072-token
  context, INT8 KV, `--vision` on, `--expert-cache auto`, `--prefill auto`, `--spec 4 --spec-min-p 0.5`,
  bundled MTP, `--resident-budget-gib 55`, `--vram-reserve-mib 700`, `--prompt-cache-every 4096`. The
  exact argument list is in `config.json`.
- **Comparison engine:** vLLM fork (`weicj/vLLM-2080Ti-Definitive`), vLLM `0.29.1rc0` + torch
  2.13.0+cu130, the same model family in FP8 (`Qwen3.8-27B-FP8`), MTP 3, TP=2, 256K, FP8 KV.

## Method

- **Decode timing** is the engine's own per-request log line (`prompt N tokens = ... generated in Y ms`).
  The single-vs-split table uses six 300-token coding prompts per config and reports the median of the
  six; cache hit and fresh-prefill speed come from the same log lines.
- **Concurrent throughput** is aggregate completion tokens from the OpenAI-compatible API divided by
  wall time, with the same 2.9K-token prompt repeated across the concurrent requests and the prompt
  cache warmed first, so the level is decode-bound. Every concurrency level was run **four times**
  (one cold, three warm) and the table reports the median; the per-round values are in
  `concurrency.json`. GPU utilisation and power are `nvidia-smi` sampled at 4 Hz.
- **The canonical suite** is the repo's own `tools/strata_bench.py --long --clients 4 --runs 3`;
  `results.json` holds every request.
- **Quality** is a 31-case suite (instruction following, Chinese, math, long-context needle at 128K,
  tool calls, vision) run against the final config.
- The three harnesses are in `scripts/`: `bench_code.py` (single vs split, reads the engine log),
  `bench_conc.py` (Strata concurrency) and `vllm_concurrency.py` (the vLLM side, which starts and stops
  vLLM itself). They are plain scripts against the OpenAI-compatible API, kept so the numbers can be
  re-run rather than taken on trust.

## Single GPU vs layer split `"gpu": [0, 1]`

Two identical cards at the same power limit. The split is not paying for extra VRAM here:

| metric | single GPU | layer split | change |
| --- | ---: | ---: | ---: |
| decode, 6 code prompts (median) | 35.2 tok/s | **51.0 tok/s** | **+45%** |
| expert cache hit rate | 85.7% | 95.8% | +10.1 pt |
| 11.6K-token prompt, fresh | 672 tok/s | **908 tok/s** | **+35%** |
| wall time for one 300-token answer | 9.6 s | **7.0 s** | -27% |

`STRATA_DECODE_TIMING=1` shows where the 45% comes from — this per-window breakdown was measured on this
host with the same cards:

| window component | single GPU | layer split |
| --- | ---: | ---: |
| whole verify window | 70.06 ms | 37.96 ms |
| CPU pool time in the window | **9.52 ms** | **0.79 ms** |
| GPU-reach wait | 28.32 ms | 15.79 ms |

On a CPU this slow the single card waits on its own expert work: the CPU pool is the bottleneck, and the
second card does not add memory so much as remove work the pool was serialising. That is the opposite of
the "on par with the faster card" result `docs/MULTI_GPU.md` records for a fast desktop CPU, and it is
consistent with it — there the pool is not what the window waits on, so a second card adds a hand-off
and no gain.

## Concurrency and `--batch 4 --batch-groups 2`

A layer split is a serialised two-stage pipeline: one hand-off per window, so a single request leaves
each card about half idle. **Concurrency alone does not fill it** — utilisation stays at 49-51% per card
from 1 to 8 concurrent requests without `--batch`. `--batch` packs several requests' tokens into one
verify window, and that is what raises utilisation:

| concurrency | split only | + `--batch 4 --batch-groups 2` | change |
| ---: | ---: | ---: | ---: |
| 1 | **58.6** | 54.4 | -7% |
| 2 | 59.7 | 64.3 | +8% |
| 4 | 59.8 | **82.9** | **+39%** |
| 8 | 59.9 | **85.0** | **+42%** |

| concurrency | GPU 0/1 util, split only | GPU 0/1 util, + batch | power W, split only | power W, + batch |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 50 / 50% | 49 / 50% | 92 / 167 | 91 / 161 |
| 2 | 49 / 51% | 94 / 100% | 92 / 163 | 121 / 176 |
| 4 | 49 / 50% | 95 / 99% | 94 / 164 | 125 / 174 |
| 8 | 49 / 50% | 90 / 99% | 92 / 167 | 125 / 173 |

The cost is one request at a time: at concurrency 1 `--batch` waits to fill a window and gives back 7%.
From 2 upward it earns that back, and at 4-8 it is worth about 40%. So: leave `--batch` off for
one-at-a-time chat, turn it on when several conversations share the engine.

## The canonical suite

The repo's own tool on the recommended config (layer split + `--batch 4 --batch-groups 2`), medians with
ranges; every request is in `results.json`.

| Configuration | Prompt tok/s | Decode tok/s | TTFT s |
| --- | ---: | ---: | ---: |
| short chat | — | 46.1 (36.4-54.2) | 0.84 |
| 4K prompt, fresh | 658.4 (587.7-666.3) | — | 6.14 |
| 16K prompt, fresh | 1059.3 (1055.2-1066.1) | — | 15.22 |
| 32K prompt, fresh | 1218.3 (1211.8-1257.5) | — | 26.38 |
| agent session, follow-up turns | — | 52.1 (38.3-67.4) | 5.40 |
| 4 clients, 128 new tokens each | — | 43.5 total (36.4-47.0) | 3.45 p50 |

The agent rows also report a 69.7% cache hit on the follow-up turns with 0 tokens re-read from storage,
which is what `--prompt-cache-every 4096` is for.

## Standard protocol pilot (draft)

The same config was also run through `strata bench run` (`docs/BENCHMARK_PROTOCOL.md`), which writes a
protocol-format result rather than this report's format. It lives beside this report in
`../2026-10-24-standard-2x-rtx-2080-ti-20261024T062225Z-375ff2de/`. Protocol
**`strata-qwen38-thinking-v1-draft.1`, status draft**, track **`custom`** (layer split + `--batch`,
which is not the release default) — this is draft pilot data, not a standard result, and the approved
manifest described in the protocol does not exist yet.

| Case | Prompt tok/s | Decode tok/s | TTFT s |
| --- | ---: | ---: | ---: |
| short (~80 token prompt) | n/a | 46.9 (45.8-48.5) | 1.68 |
| prompt-4k | 696.3 (664.8-697.6) | 47.1 (44.6-51.9) | 5.91 |
| prompt-16k | 1113.5 (1113.4-1114.2) | 50.7 (48.6-52.3) | 14.77 |

21 of 21 required requests finished, none failed or were skipped, 12/12 sanity checks passed
(instruction 4/4, retrieval 4/4, tool calls 4/4), and four requests at once gave 57.4 / 62.4 / 70.0 tok/s
total per round across three rounds. 3 min 52 s after the model had loaded. The decode figures agree with
the canonical suite above (46.1-52.1). The four-at-once figure is lower than the 82.9 in the concurrency
table because the protocol's requests generate up to 512 tokens with thinking on rather than ~200 here,
so less of each round is spent decoding — the two are not the same measurement and should not be read as
a cross-check of each other.

## Power limit 180 W vs 250 W

An earlier run on this host (engine 0.1.43.1, `concurrency.py`, one round per level), so treat it as a
direction, not a same-day number. vLLM does **not** move; Strata does:

| concurrency | Strata `--batch 4` @ 180 W | Strata `--batch 4` @ 250 W | vLLM @ 250 W |
| ---: | ---: | ---: | ---: |
| 1 | 36.7 | 43.9 | 37.1 |
| 2 | 57.0 | 62.7 | 37.1 |
| 4 | 75.1 | **90.8** | 37.1 |
| 8 | 82.9 | **92.0** | 37.0 |

vLLM saturates its SMs at concurrency 1 (86-90% util) and produces the same ~37 tok/s at 180 W and at
250 W, drawing 210-244 W at the higher limit: it is compute/latency-bound on SM 75, where FP8 has no
hardware support and is emulated. Strata is latency-bound first, fills the GPU with concurrency +
`--batch`, and only then does the power limit bind — raising it added +21% at concurrency 4.

## Strata vs vLLM, same cards, same prompt

Same day, same power limit (180 W), same prompt, same concurrency gradient. vLLM's numbers are one round
per level, as its harness runs them:

| concurrency | Strata + `--batch 4` | vLLM | Strata / vLLM |
| ---: | ---: | ---: | ---: |
| 1 | 54.4 (58.6 without `--batch`) | 34.7 | 1.57x (1.69x) |
| 2 | 64.3 | 35.9 | 1.79x |
| 4 | 82.9 | 35.9 | **2.31x** |
| 8 | 85.0 | 35.8 | **2.37x** |

vLLM is flat from concurrency 1 to 8 (34.7 -> 35.8) at 87-94% GPU utilisation: adding requests does not
add throughput because it has nothing left to fill. That is the honest shape of the comparison — it is a
statement about this hardware at this quant, not about vLLM in general.

## Quality

31/31 on the 31-case suite with the final config (layer split + `--batch 4`, 128K context, `--vision`),
the same score the single-GPU config gets. Nothing here is bit-identical to a single-GPU run; the
hand-off is the source of the difference and the suite was run to confirm it does not change answers.

## Limits

- One host, one quant (UD-IQ4_XS). The vLLM side runs FP8 weights, so the comparison is the deployment a
  user would actually make with each engine, not a quant-matched one.
- **Every table is at 180 W except "Power limit 180 W vs 250 W" and its vLLM column**, which are an
  earlier run on this host with engine 0.1.43.1 and a different harness. Raising the limit needs root and
  was not repeated on this revision.
- Concurrency numbers are medians of four rounds; the canonical suite follows its own protocol
  (`--runs 3`). The per-window breakdown in the first section is from `STRATA_DECODE_TIMING=1`, which
  serialises a little and is not directly comparable to the API-level numbers around it.
- No thermal soak, and no 128K-prompt throughput sweep: the longest fresh prompt here is 32K. The 128K
  long-context cases in the quality suite are needle checks, not throughput measurements.
- The measure of "the split pays" is one prompt family (coding, 300 generated tokens). A workload with
  much longer generations may land the hand-off differently.
