# Community benchmark on AMD Radeon

Measured on 2026-10-31 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 24 GiB VRAM
- CPU: Intel(R) Core(TM) Ultra 7 255H (16 cores, 16 threads); installed RAM: 62.3 GiB; storage: not measured
- OS: Linux 6.18.15; GPU driver: not read; CUDA: not read
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.54
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: b028
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 68.3 (49.3-91.5) | 0.40 (0.34-0.43) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 784.8 (784.4-785.3) | not measured | 5.17 (5.16-5.17) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1211.5 (1194.3-1235.4) | not measured | 13.29 (13.04-13.47) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1199.3 (1193.1-1224.8) | not measured | 26.76 (26.20-26.90) |
| agent session, follow-up turns | 14626 (14602-14642) at the last turn | see below | up to 64 | 3 | see below | 75.6 (40.4-102.4) | 5.20 (4.82-5.96) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.5-68.7); earlier-turn tokens read again per session: 143 (73-144) (earlier turns that the cache did not hold; the 13729 (13729-13730) new tokens of the files are read once); first turn (nothing cached) first token: 3.95 s (3.90-3.99).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 157.1 | 3.903 | 60.4 |
| 1 | 3776 | 656 | 3120 | 603.1 | 5.202 | 75.6 |
| 2 | 5309 | 3839 | 1470 | 278.5 | 5.327 | 73.6 |
| 3 | 8753 | 5373 | 3380 | 596.9 | 5.71 | 72.4 |
| 4 | 11679 | 8816 | 2863 | 569.0 | 5.06 | 69.3 |
| 5 | 14642 | 11672 | 2970 | 502.3 | 5.956 | 59.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 266 s.
