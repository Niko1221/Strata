# Community benchmark on NVIDIA GeForce RTX 5090

Measured on 2026-10-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090, 32 GiB VRAM
- CPU: 12th Gen Intel(R) Core(TM) i7-12700KF (12 cores, 20 threads); installed RAM: 95.8 GiB; storage: not measured
- OS: Windows 11 (10.0.26300); GPU driver: 617.42; CUDA: not read
- Strata: project 0.1.46, commit c093ac26; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: swift-1.5-qwen3.8-flash-next-iq3_s
- Context: 524288 tokens; parallel slots: 1
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 156.7 (127.9-166.6) | 0.29 (0.28-0.37) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 3084.4 (2786.5-3111.6) | not measured | 1.34 (1.32-1.48) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 5497.6 (5353.2-5515.2) | not measured | 2.97 (2.94-3.06) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 5572.8 (5496.9-5592.4) | not measured | 5.80 (5.79-5.88) |
| agent session, follow-up turns | 14701 (14685-14714) at the last turn | see below | up to 64 | 3 | see below | 189.6 (137.0-256.8) | 1.34 (1.18-1.66) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.2 % (69.2-69.3); earlier-turn tokens read again per session: 143 (74-144) (earlier turns that the cache did not hold; the 13827 (13826-13828) new tokens of the files are read once); first turn (nothing cached) first token: 1.20 s (1.09-1.24).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 570.3 | 1.093 | 113.0 |
| 1 | 4102 | 639 | 3463 | 2120.0 | 1.656 | 137.0 |
| 2 | 5635 | 4095 | 1540 | 1078.1 | 1.458 | 170.0 |
| 3 | 9068 | 5699 | 3369 | 2188.4 | 1.568 | 168.1 |
| 4 | 11840 | 9061 | 2779 | 1696.9 | 1.658 | 189.6 |
| 5 | 14701 | 11882 | 2819 | 2016.6 | 1.418 | 188.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 74 s.
