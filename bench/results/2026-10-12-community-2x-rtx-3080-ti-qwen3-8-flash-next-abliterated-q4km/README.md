# Community benchmark on NVIDIA GeForce RTX 3080 Ti + NVIDIA GeForce RTX 3080 Ti

Measured on 2026-10-12 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3080 Ti + NVIDIA GeForce RTX 3080 Ti, 24 GiB VRAM
- CPU: 13th Gen Intel(R) Core(TM) i5-13600K (14 cores, 20 threads); installed RAM: 76.1 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 595.91.07; CUDA: 13.2
- Strata: project 0.1.43, commit 3ef5e6c5; engine 0.1.43
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-abliterated-q4km
- Context: 196608 tokens; parallel slots: 1
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 104.5 (98.7-122.2) | 0.35 (0.32-0.38) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 458.6 (458.5-458.8) | not measured | 8.78 (8.78-8.78) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1198.2 (1194.0-1198.6) | not measured | 13.42 (13.42-13.47) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1456.4 (1448.9-1459.9) | not measured | 22.03 (21.98-22.15) |
| agent session, follow-up turns | 14479 (14422-14496) at the last turn | see below | up to 64 | 3 | see below | 104.1 (68.7-131.9) | 8.71 (8.25-9.01) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.7); earlier-turn tokens read again per session: 73 (5-75) (earlier turns that the cache did not hold; the 13635 (13633-13635) new tokens of the files are read once); first turn (nothing cached) first token: 5.67 s (5.54-5.68).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 110.2 | 5.545 | 73.5 |
| 1 | 3705 | 624 | 3081 | 348.6 | 8.856 | 97.8 |
| 2 | 5238 | 3698 | 1540 | 183.8 | 8.395 | 95.5 |
| 3 | 8682 | 5301 | 3381 | 381.1 | 8.89 | 95.2 |
| 4 | 11590 | 8730 | 2860 | 328.2 | 8.736 | 113.7 |
| 5 | 14496 | 11652 | 2844 | 327.4 | 8.708 | 117.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 312 s.
