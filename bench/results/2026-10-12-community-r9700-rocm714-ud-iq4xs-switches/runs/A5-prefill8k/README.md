# Community benchmark on AMD Radeon AI PRO R9700 + AMD Radeon AI PRO R9700

Measured on 2026-10-12 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon AI PRO R9700 + AMD Radeon AI PRO R9700, 32 GiB VRAM
- CPU: AMD Ryzen 9 3900 12-Core Processor (12 cores, 24 threads); installed RAM: 125.7 GiB; storage: not measured
- OS: Linux 6.8.0-139-generic; GPU driver: not read; CUDA: not read
- Strata: project 0.1.43, commit 7695859a; engine 0.1.43
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-ud-iq4_xs
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 57.9 (49.4-73.5) | 0.40 (0.34-0.47) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1318.1 (1116.1-1327.0) | not measured | 3.08 (3.06-3.63) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1650.0 (1600.7-1668.7) | not measured | 9.77 (9.66-10.07) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1642.7 (1640.5-1650.7) | not measured | 19.56 (19.46-19.58) |
| agent session, follow-up turns | 14462 (14461-14520) at the last turn | see below | up to 64 | 3 | see below | 67.0 (35.5-90.0) | 2.82 (2.13-3.59) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.7-68.8); earlier-turn tokens read again per session: 72 (3-73) (earlier turns that the cache did not hold; the 13634 (13633-13637) new tokens of the files are read once); first turn (nothing cached) first token: 1.60 s (1.58-1.65).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 374.9 | 1.651 | 45.2 |
| 1 | 3713 | 633 | 3080 | 909.4 | 3.42 | 65.5 |
| 2 | 5246 | 3776 | 1470 | 592.0 | 2.514 | 55.4 |
| 3 | 8690 | 5310 | 3380 | 950.9 | 3.59 | 53.6 |
| 4 | 11614 | 8683 | 2931 | 884.1 | 3.351 | 46.8 |
| 5 | 14462 | 11616 | 2846 | 917.8 | 3.139 | 35.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 188 s.
