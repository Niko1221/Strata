# Community benchmark on AMD Radeon

Measured on 2026-11-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 16 GiB VRAM
- CPU: AMD Ryzen 5 3600 6-Core Processor (6 cores, 12 threads); installed RAM: 31.2 GiB; storage: not measured
- OS: Linux 7.3.4-1-cachyos; GPU driver: 7.3.4-1-cachyos; CUDA: not read
- Strata: project 0.1.44.1, commit 9ae82ef9; engine 0.1.44.1
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
| short chat | 88 (85-93) | 0 | 100 (100-100) | 9 | not measured | 29.8 (22.7-40.2) | 1.49 (1.44-1.59) |
| 4K prompt, fresh | 4021 (4021-4021) | 0 (0-0) | 24 cap | 3 | 339.0 (330.0-339.7) | not measured | 11.90 (11.88-12.23) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 365.4 (364.7-366.7) | not measured | 44.00 (43.84-44.08) |
| 32K prompt, fresh | 32050 (32050-32050) | 0 (0-0) | 24 cap | 3 | 363.3 (363.1-364.1) | not measured | 88.28 (88.10-88.33) |
| agent session, follow-up turns | 13643 (13640-13701) at the last turn | see below | up to 64 | 3 | see below | 32.9 (24.8-39.9) | 9.88 (6.24-10.86) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 67.5 % (67.5-67.6); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 12796 (12796-12797) new tokens of the files are read once); first turn (nothing cached) first token: 3.90 s (3.88-3.92).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 647 | 0 | 647 | 166.7 | 3.918 | 27.4 |
| 1 | 2822 | 668 | 2154 | 254.9 | 8.493 | 28.9 |
| 2 | 4390 | 2885 | 1505 | 227.0 | 6.669 | 30.8 |
| 3 | 7798 | 4418 | 3380 | 312.4 | 10.861 | 32.0 |
| 4 | 10716 | 7853 | 2863 | 274.5 | 10.471 | 38.5 |
| 5 | 13640 | 10741 | 2899 | 280.7 | 10.369 | 37.8 |

### Concurrency: 4 clients

Total 22.4 tok/s (21.7-23.2); first token p50 10.026 s, p95 19.225 s; request latency p50 14.185 s, p95 23.194 s; errors 0. Server parallel slots: 1 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 768 s.
