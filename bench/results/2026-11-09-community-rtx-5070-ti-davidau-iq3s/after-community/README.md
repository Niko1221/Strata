# Community benchmark on NVIDIA GeForce RTX 5070 Ti

Measured on 2026-11-09 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5070 Ti, 16 GiB VRAM
- CPU: AMD Ryzen 7 7700X 8-Core Processor (8 cores, 16 threads); installed RAM: 63.1 GiB; model storage: Samsung SSD 980 500GB SSD/NVMe (from the standard-run inventory); unrelated drive inventory omitted
- OS: Windows 11 (10.0.26200); GPU driver: 621.14; CUDA (driver supports up to): 13.5
- Strata: project ?, commit not a git checkout; engine 0.1.58
- GPU detail (after the run, near idle): power_limit_w=300.00, power_draw_w=178.87, pcie=gen5 x16, pstate=P1, temp_c=56.41
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 131072 tokens; parallel slots: 2
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 65.6 (63.3-86.9) | 0.55 (0.48-0.58) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 2590.2 (2504.4-2591.1) | not measured | 1.60 (1.57-1.64) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2734.4 (2729.5-2736.5) | not measured | 5.91 (5.91-5.93) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2929.2 (2927.1-2932.0) | not measured | 10.99 (10.99-11.01) |
| agent session, follow-up turns | 9088 (9051-9095) at the last turn | see below | up to 64 | 3 | see below | 70.6 (59.3-87.5) | 1.11 (1.07-1.26) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.8 % (70.6-70.9); earlier-turn tokens read again per session: 74 (73-144) (earlier turns that the cache did not hold; the 8208 (8206-8208) new tokens of the files are read once); first turn (nothing cached) first token: 1.21 s (1.14-1.25).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 548.7 | 1.142 | 64.2 |
| 1 | 2273 | 631 | 1642 | 1561.4 | 1.072 | 63.3 |
| 2 | 3979 | 2337 | 1642 | 1548.5 | 1.095 | 71.1 |
| 3 | 5685 | 3972 | 1713 | 1470.1 | 1.207 | 69.8 |
| 4 | 7382 | 5740 | 1642 | 1546.4 | 1.086 | 79.0 |
| 5 | 9088 | 7375 | 1713 | 1404.1 | 1.256 | 69.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 115 s.
