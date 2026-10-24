# Community benchmark on NVIDIA GeForce RTX 4090

Measured on 2026-10-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4090, 22 GiB VRAM
- CPU: AMD Ryzen 9 7950X 16-Core Processor (16 cores, 32 threads); installed RAM: 45.4 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 615.71.09; CUDA: not read
- Strata: project 0.1.45.3, commit e88ee39d; engine 0.1.45.3
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Context: 204800 tokens; parallel slots: 1
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 105.9 (81.3-130.1) | 0.28 (0.24-0.37) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1294.7 (620.1-2075.4) | not measured | 3.12 (1.95-6.50) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3241.6 (3131.2-3263.4) | not measured | 4.97 (4.94-5.15) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 3307.1 (3261.4-3311.9) | not measured | 9.72 (9.70-9.85) |
| agent session, follow-up turns | 14690 (14684-14702) at the last turn | see below | up to 64 | 3 | see below | 121.0 (106.5-158.6) | 1.80 (1.64-2.02) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.4 % (69.4-69.4); earlier-turn tokens read again per session: 74 (72-74) (earlier turns that the cache did not hold; the 13812 (13811-13813) new tokens of the files are read once); first turn (nothing cached) first token: 1.23 s (1.20-1.35).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 515.9 | 1.195 | 86.5 |
| 1 | 4119 | 633 | 3486 | 1769.6 | 1.988 | 118.4 |
| 2 | 5652 | 4182 | 1470 | 866.5 | 1.711 | 111.1 |
| 3 | 9085 | 5645 | 3440 | 1720.3 | 2.021 | 111.4 |
| 4 | 11822 | 9139 | 2683 | 1464.0 | 1.849 | 141.9 |
| 5 | 14690 | 11883 | 2807 | 1448.3 | 1.954 | 111.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 113 s.
