# Community benchmark on NVIDIA GeForce RTX 5070

Measured on 2026-11-05 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5070, 12 GiB VRAM
- CPU: AMD Ryzen 7 9700X 8-Core Processor (8 cores, 16 threads); installed RAM: 125.6 GiB; storage: Samsung SSD 990 EVO 1TB SSD/NVMe; Samsung SSD 990 PRO 2TB SSD/NVMe
- OS: Windows 11 (10.0.26300); GPU driver: 621.14; CUDA (driver supports up to): 13.5
- Strata: project 0.1.56, commit ffa80d73; engine ds4-0.1
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=81.04, pcie=gen4 x16, pstate=P1, temp_c=49.48
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: deepseek-v4.1-flash
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --clients 4 --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 32 (29-38) | 0 | 100 (100-100) | 9 | not measured | 15.3 (6.8-16.1) | 1.14 (0.94-1.32) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 3 | 390.5 (386.9-405.6) | not measured | 10.59 (10.18-10.67) |
| 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 3 | 771.4 (767.9-774.2) | not measured | 21.49 (21.43-21.61) |
| 32K prompt, fresh | 33126 (33126-33126) | 0 (0-0) | 24 cap | 3 | 819.6 (812.7-832.9) | not measured | 40.46 (39.80-40.81) |
| agent session, follow-up turns | 12856 (12855-12873) at the last turn | see below | up to 64 | 3 | see below | 15.3 (13.1-16.6) | 8.54 (6.54-9.47) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.5 % (69.4-69.5); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 11992 (11991-11993) new tokens of the files are read once); first turn (nothing cached) first token: 4.89 s (4.82-4.93).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 602 | 0 | 602 | 123.8 | 4.892 | 13.2 |
| 1 | 3364 | 639 | 2725 | 301.0 | 9.066 | 13.4 |
| 2 | 4872 | 3418 | 1454 | 213.5 | 6.823 | 13.1 |
| 3 | 7977 | 4929 | 3048 | 322.3 | 9.474 | 14.4 |
| 4 | 10219 | 8038 | 2181 | 283.5 | 7.711 | 15.3 |
| 5 | 12856 | 10268 | 2588 | 299.5 | 8.688 | 14.6 |

### Concurrency: 4 clients

Total 12.5 tok/s (12.0-12.9); first token p50 16.817 s, p95 32.8 s; request latency p50 25.204 s, p95 41.726 s; errors 0. Server parallel slots: 1 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 659 s.
