# Community benchmark on 4x AMD Radeon (gfx1201, Device 0x7551)

Measured on 2026-10-16 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: 4x AMD Radeon (gfx1201, Device 0x7551), 127 GiB VRAM total
- CPU: AMD EPYC 7452 32-Core Processor (32 cores, 64 threads); installed RAM: 220.0 GiB; storage: not measured
- OS: Linux 7.2.10-200.fc44.x86_64; GPU driver: 7.2.10-200.fc44.x86_64; CUDA: not read
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: none during the run; PCIe host->device probe 27.8 GB/s; CPU has no AVX-512 (AVX2 expert kernels)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-q5_k_xl
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 45.1 (37.4-82.5) | 0.54 (0.50-0.58) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1228.5 (1166.3-1229.1) | not measured | 3.31 (3.31-3.49) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2047.6 (2020.9-2051.7) | not measured | 7.89 (7.88-8.00) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2686.7 (2676.8-2719.5) | not measured | 12.00 (11.86-12.08) |
| agent session, follow-up turns | 14570 (14539-14582) at the last turn | see below | up to 64 | 3 | see below | 44.9 (31.6-52.5) | 3.14 (2.08-3.68) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.6); earlier-turn tokens read again per session: 74 (74-74) (earlier turns that the cache did not hold; the 13728 (13728-13728) new tokens of the files are read once); first turn (nothing cached) first token: 1.97 s (1.88-1.97).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 330.3 | 1.881 | 41.6 |
| 1 | 3755 | 635 | 3120 | 916.8 | 3.449 | 42.3 |
| 2 | 5288 | 3748 | 1540 | 689.7 | 2.279 | 52.5 |
| 3 | 8732 | 5351 | 3381 | 931.1 | 3.682 | 46.9 |
| 4 | 11637 | 8775 | 2862 | 871.0 | 3.339 | 48.3 |
| 5 | 14570 | 11671 | 2899 | 893.0 | 3.302 | 31.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 316 s.
