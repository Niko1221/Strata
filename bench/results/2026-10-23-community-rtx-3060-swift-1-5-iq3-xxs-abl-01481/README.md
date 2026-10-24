# Community benchmark on NVIDIA GeForce RTX 3060

Measured on 2026-10-23 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3060, 12 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2697 v2 @ 2.70GHz (24 cores, 48 threads); installed RAM: 127.9 GiB; storage: not measured
- OS: Windows 10 (10.0.26200); GPU driver: 610.88; CUDA: not read
- Strata: project 0.1.48.1, commit 7ddc17a3; engine 0.1.48.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: swift-1.5-iq3_xxs-abl-01481
- Context: 128000 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 44.1 (37.7-53.0) | 0.74 (0.66-0.77) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 475.8 (467.0-476.5) | not measured | 8.50 (8.47-8.65) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 745.0 (742.8-745.1) | not measured | 21.61 (21.61-21.69) |
| agent session, follow-up turns | 14610 (14588-14616) at the last turn | see below | up to 64 | 3 | see below | 50.9 (39.9-66.6) | 7.01 (5.31-7.95) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.6 % (69.5-69.6); earlier-turn tokens read again per session: 1 (1-72) (earlier turns that the cache did not hold; the 13732 (13729-13733) new tokens of the files are read once); first turn (nothing cached) first token: 4.31 s (4.16-4.37).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 148.0 | 4.159 | 29.6 |
| 1 | 4097 | 627 | 3470 | 438.9 | 7.954 | 42.5 |
| 2 | 5630 | 4161 | 1469 | 267.1 | 5.56 | 53.1 |
| 3 | 9063 | 5694 | 3369 | 429.1 | 7.897 | 39.9 |
| 4 | 11835 | 9126 | 2709 | 385.2 | 7.096 | 53.8 |
| 5 | 14616 | 11899 | 2717 | 390.5 | 7.015 | 42.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 268 s.
