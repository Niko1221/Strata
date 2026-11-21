# Community benchmark on AMD Radeon

Measured on 2026-11-21 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 16 GiB VRAM
- CPU: AMD Ryzen 5 3600 6-Core Processor (6 cores, 12 threads); installed RAM: 31.2 GiB; storage: not measured
- OS: Linux 7.3.5-2-cachyos; GPU driver: 7.3.5-2-cachyos; CUDA: not read
- Strata: project 0.1.44.1, commit 71a5118d; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-coder-iq1_m
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --clients 4 --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K / 32K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 88 (85-93) | 0 | 100 (100-100) | 9 | not measured | 29.2 (25.4-43.6) | 1.44 (1.40-1.51) |
| 4K prompt, fresh | 4021 (4021-4021) | 0 (0-0) | 24 cap | 3 | 344.3 (342.3-346.4) | not measured | 11.72 (11.65-11.79) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 371.5 (370.7-371.7) | not measured | 43.27 (43.25-43.36) |
| 32K prompt, fresh | 32050 (32050-32050) | 0 (0-0) | 24 cap | 3 | 368.3 (368.0-368.5) | not measured | 87.09 (87.04-87.15) |
| agent session, follow-up turns | 13700 (13662-13715) at the last turn | see below | up to 64 | 3 | see below | 33.7 (25.0-39.2) | 10.04 (6.38-12.09) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 67.6 % (67.4-67.7); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 12797 (12796-12797) new tokens of the files are read once); first turn (nothing cached) first token: 4.00 s (3.92-4.13).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 647 | 0 | 647 | 158.1 | 4.128 | 12.1 |
| 1 | 2804 | 650 | 2154 | 248.0 | 8.724 | 32.8 |
| 2 | 4372 | 2867 | 1505 | 219.5 | 6.895 | 32.6 |
| 3 | 7779 | 4399 | 3380 | 319.8 | 10.609 | 31.2 |
| 4 | 10706 | 7842 | 2864 | 271.1 | 10.605 | 36.5 |
| 5 | 13662 | 10763 | 2899 | 275.6 | 10.562 | 39.2 |

### Concurrency: 4 clients

Total 23.3 tok/s (22.8-23.3); first token p50 9.837 s, p95 18.343 s; request latency p50 13.698 s, p95 22.182 s; errors 0. Server parallel slots: 1 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 731 s.
