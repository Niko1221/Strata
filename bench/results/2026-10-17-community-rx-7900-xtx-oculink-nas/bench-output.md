# Community benchmark on AMD Radeon

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 24 GiB VRAM
- CPU: Intel(R) Core(TM) Ultra 7 255H (16 cores, 16 threads); installed RAM: 62.3 GiB; storage: not measured
- OS: Linux 6.18.15; GPU driver: not read; CUDA: not read
- Strata: project 0.1.44.1, commit not a git checkout; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: arbeit
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 63.7 (47.9-68.4) | 0.47 (0.37-0.55) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 741.4 (734.4-746.7) | not measured | 5.45 (5.41-5.50) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1195.1 (1176.6-1226.9) | not measured | 13.47 (13.12-13.68) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1224.1 (1218.1-1226.2) | not measured | 26.22 (26.17-26.35) |
| agent session, follow-up turns | 14594 (14567-14596) at the last turn | see below | up to 64 | 3 | see below | 74.0 (48.2-83.3) | 5.31 (5.04-5.51) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.5-68.8); earlier-turn tokens read again per session: 73 (4-144) (earlier turns that the cache did not hold; the 13729 (13729-13730) new tokens of the files are read once); first turn (nothing cached) first token: 4.04 s (3.83-4.24).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 160.2 | 3.83 | 55.2 |
| 1 | 3759 | 637 | 3122 | 582.9 | 5.393 | 57.5 |
| 2 | 5292 | 3752 | 1540 | 305.5 | 5.07 | 60.5 |
| 3 | 8736 | 5356 | 3380 | 630.5 | 5.393 | 63.9 |
| 4 | 11651 | 8789 | 2862 | 541.8 | 5.314 | 64.5 |
| 5 | 14567 | 11668 | 2899 | 547.9 | 5.322 | 54.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 269 s.
