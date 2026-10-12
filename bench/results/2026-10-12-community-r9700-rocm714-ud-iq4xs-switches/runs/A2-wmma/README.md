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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 58.9 (45.6-70.7) | 0.41 (0.34-0.49) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1616.5 (1455.1-1635.3) | not measured | 2.51 (2.48-2.79) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2547.0 (2246.7-2570.3) | not measured | 6.34 (6.28-7.18) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2686.5 (1871.5-2706.7) | not measured | 11.98 (11.89-17.17) |
| agent session, follow-up turns | 14499 (14487-14505) at the last turn | see below | up to 64 | 3 | see below | 71.7 (54.9-84.8) | 2.24 (1.81-2.94) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.5-68.8); earlier-turn tokens read again per session: 5 (3-143) (earlier turns that the cache did not hold; the 13634 (13632-13635) new tokens of the files are read once); first turn (nothing cached) first token: 1.59 s (1.58-1.63).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 379.5 | 1.632 | 43.9 |
| 1 | 3713 | 633 | 3080 | 1078.8 | 2.887 | 57.5 |
| 2 | 5246 | 3706 | 1540 | 708.7 | 2.204 | 60.8 |
| 3 | 8690 | 5239 | 3451 | 1185.4 | 2.945 | 54.9 |
| 4 | 11601 | 8741 | 2860 | 1069.5 | 2.71 | 74.7 |
| 5 | 14505 | 11661 | 2844 | 1110.2 | 2.598 | 64.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 153 s.
