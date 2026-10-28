# 2x RTX 4060 Ti 16 GB, Qwen3.8-Flash-Next UD-IQ4_XS, v0.2.7.1

Two runs of `tools/strata_bench.py --long` (suite 1) on the same machine and the same engine, on the same evening.
The first is my production config. The second uses the start flags from #3888 (2x RTX 3090, same model), so I could
see how far apart the two machines really are.

- 2x RTX 4060 Ti 16 GB (sm_89, PCIe 4.0 x8 each), `--layer-split auto`, Threadripper PRO 3975WX, 128 GB DDR4-2667, Linux
- Strata `v0.2.7.1` (`46c152a1`, the report calls it project 0.1.53) built here with CUDA 12.6 for sm_89
- Both runs have the same nine sm_86 switches as #4290 plus `STRATA_PROMPT_ATTN_IMMA=1` and `STRATA_HC_UPMIX=1` in the
  config's `env`

## Results

| suite 1, median (range) | production | #3888 flags | difference |
|---|---|---|---|
| short chat, decode tok/s | 58.2 (53.7-79.1) | 45.8 (35.1-53.8) | -21% |
| short chat, first token | 0.44 s | 0.79 s | |
| 4K prompt tok/s | 910.9 (870.0-914.9) | 836.0 (832.1-848.0) | -8% |
| 16K prompt tok/s | 1787.7 (1781.5-1793.6) | 1534.3 (1530.0-1560.8) | -14% |
| 32K prompt tok/s | 2074.3 (2073.8-2077.6) | 1848.0 (1839.3-1862.4) | -11% |
| agent decode tok/s | 66.5 (57.9-71.1) | 54.6 (50.8-61.9) | -18% |
| agent turn first token | 4.41 s | 4.29 s | |
| agent cache hit | 69.3% | 69.3% | |

0 failed requests in both. Per-request data and the generated reports are in `production/` and `issue-3888-flags/`.

## What is different between the two

| | production | #3888 flags |
|---|---|---|
| context | 262144 | 131072 |
| KV | `--kv int8 --kv-resident 32768` (32768 cells per layer in VRAM, the rest in pinned RAM) | default, all in VRAM |
| `--remote-expert-opt`, `--pcie-frac 0.00` | yes | no (the engine probed `pcie_frac 0.37`) |
| `--spec-min-p` | 0.50 | default |
| MTP draft subset | Italian+English, 54675 tokens (109.5 MiB) | default `cjk`, 106299 tokens (212.9 MiB) |
| `--vision` | yes, on the CPU | no |
| resident experts (engine log) | **7615** | **6410** |
| expert cache, CUDA0 + CUDA1 | 9.07 + 8.09 GiB | 7.69 + 6.75 GiB |
| layer split | K=23 | K=24 |

The VRAM reserve (700 MiB) and the conversation cache (8192 MiB, 4 slots) are the same in both: in the second run
they came from the defaults.

My reading: on 16 GB cards the KV options leave room for about 19% more experts in VRAM, so fewer experts are read
from RAM over PCIe, and that is where most of the difference comes from. That is my reading of the log lines in
`engine-log-excerpt.txt`, not a measurement: I changed seven things at once and did not test them one by one. On
48 GB the #3888 flags probably cost little, because almost everything fits anyway.

## Next to #3888

| suite 1 | 2x RTX 3090 (#3888, 0.1.48.1) | 2x RTX 4060 Ti, production (0.2.7.1) | ratio |
|---|---|---|---|
| short chat, decode tok/s | 117.5 | 58.2 | 50% |
| 4K prompt tok/s | 1292.4 | 910.9 | 70% |
| 16K prompt tok/s | 2415.7 | 1787.7 | 74% |
| agent decode tok/s | 120.8 | 66.5 | 55% |

Different engine versions and different configs, so this is "each machine as it is used", not card against card.

## Limits

- One run of the suite per config (3 runs per test inside it), no alternated order.
- The production run used the server that had been serving my office assistant and Codex since 16:43, so its expert
  cache was already adapted to real use. The second run was a fresh server.
- `bench.sh` is the quick check, not `strata bench run` from BENCHMARK_PROTOCOL.md. No correctness check.
- The chat questions are in English, so the Italian draft subset helps little here.
- Both runs start with a LOW warning on CUDA1 (381 and 277 MiB free).

**Not run:** `strata bench run`, KL or the quality suite, and the seven differences one at a time.

I did this with an AI coding assistant: it ran the benchmark tool, put this folder together and drafted this text. I
chose what to measure, stopped and restarted the server, and checked the numbers.
