# Community benchmark on NVIDIA GeForce RTX 5070

Measured on 2026-11-05 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5070, 12 GiB VRAM
- CPU: AMD Ryzen 7 9700X 8-Core Processor (8 cores, 16 threads); installed RAM: 125.6 GiB; storage: Samsung SSD 990 EVO 1TB SSD/NVMe; Samsung SSD 990 PRO 2TB SSD/NVMe
- OS: Windows 11 (10.0.26300); GPU driver: 621.14; CUDA (driver supports up to): 13.5
- Strata: project 0.1.56, commit ffa80d73; engine ds4-0.1
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=79.12, pcie=gen4 x16, pstate=P1, temp_c=49.28
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: deepseek-v4.1-flash
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking on for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --clients 4 --long --think` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 58 (55-64) | 0 | 100 (100-100) | 9 | not measured | 15.9 (13.7-16.4) | 1.84 (1.74-1.99) |
| 4K prompt, fresh | 4151 (4151-4151) | 0 (0-0) | 24 cap | 3 | 389.9 (387.3-393.8) | not measured | 10.67 (10.55-10.73) |
| 16K prompt, fresh | 16588 (16588-16588) | 0 (0-0) | 24 cap | 3 | 749.9 (746.2-750.5) | not measured | 22.14 (22.12-22.25) |
| 32K prompt, fresh | 33152 (33152-33152) | 0 (0-0) | 24 cap | 3 | 836.0 (834.8-840.2) | not measured | 39.71 (39.49-39.77) |
| agent session, follow-up turns | 12854 (12835-12872) at the last turn | see below | up to 64 | 3 | see below | 15.8 (13.2-16.8) | 8.59 (6.62-10.12) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.4 % (69.4-69.5); earlier-turn tokens read again per session: 69 (69-69) (earlier turns that the cache did not hold; the 11932 (11932-11933) new tokens of the files are read once); first turn (nothing cached) first token: 4.71 s (4.64-5.01).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 627 | 0 | 627 | 113.1 | 5.012 | 13.6 |
| 1 | 3356 | 626 | 2730 | 282.1 | 9.754 | 13.8 |
| 2 | 4868 | 3414 | 1454 | 199.6 | 7.377 | 13.2 |
| 3 | 7967 | 4919 | 3048 | 304.2 | 10.118 | 14.3 |
| 4 | 10211 | 8030 | 2181 | 260.3 | 8.488 | 14.8 |
| 5 | 12854 | 10266 | 2588 | 283.3 | 9.231 | 14.2 |

### Concurrency: 4 clients

Total 12.8 tok/s (12.3-13.1); first token p50 17.528 s, p95 33.147 s; request latency p50 25.228 s, p95 40.812 s; errors 0. Server parallel slots: 1 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 639 s.
