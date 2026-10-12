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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 66.0 (54.5-78.9) | 0.38 (0.33-0.46) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1625.9 (1374.1-1634.9) | not measured | 2.50 (2.49-2.95) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2563.3 (2226.7-2610.8) | not measured | 6.30 (6.19-7.25) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2720.2 (2663.4-2760.0) | not measured | 11.83 (11.66-12.08) |
| agent session, follow-up turns | 14512 (14473-14524) at the last turn | see below | up to 64 | 3 | see below | 76.0 (61.9-95.8) | 2.23 (1.81-2.92) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.8); earlier-turn tokens read again per session: 73 (3-73) (earlier turns that the cache did not hold; the 13636 (13636-13637) new tokens of the files are read once); first turn (nothing cached) first token: 1.57 s (1.57-1.63).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 380.3 | 1.628 | 44.1 |
| 1 | 3701 | 619 | 3082 | 1065.9 | 2.925 | 71.4 |
| 2 | 5234 | 3694 | 1540 | 695.3 | 2.248 | 66.4 |
| 3 | 8678 | 5298 | 3380 | 1174.5 | 2.914 | 61.9 |
| 4 | 11602 | 8741 | 2861 | 1057.3 | 2.744 | 95.8 |
| 5 | 14473 | 11627 | 2846 | 1128.6 | 2.559 | 64.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 143 s.
