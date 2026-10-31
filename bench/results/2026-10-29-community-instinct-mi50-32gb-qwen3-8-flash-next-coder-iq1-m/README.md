# Community benchmark on Radeon Instinct MI50 32GB

Measured on 2026-10-29 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Radeon Instinct MI50 32GB, 32 GiB VRAM
- CPU: AMD Ryzen Threadripper PRO 3975WX 32-Cores (32 cores, 64 threads); installed RAM: 247.3 GiB; storage: nvme0n1 Lexar SSD NM620 1TB        nvme; nvme1n1 SAMSUNG MZVLB1T0HBLR-000L7 nvme
- OS: Linux 7.0.0-42-generic; GPU driver: 7.1.3.31500000; CUDA (driver supports up to): not read
- Strata: project 0.1.53, commit fd91cebb; engine 0.1.53
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-coder-iq1_m
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 61.7 (55.0-74.9) | 0.64 (0.61-0.70) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 462.8 (461.7-464.0) | not measured | 8.71 (8.69-8.74) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 492.7 (492.7-494.1) | not measured | 32.63 (32.54-32.64) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 487.0 (484.9-487.9) | not measured | 65.86 (65.75-66.15) |
| agent session, follow-up turns | 12919 (12917-12949) at the last turn | see below | up to 64 | 3 | see below | 63.7 (56.6-74.7) | 6.31 (4.09-7.40) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.5 % (69.4-69.5); earlier-turn tokens read again per session: 2 (1-3) (earlier turns that the cache did not hold; the 12065 (12064-12066) new tokens of the files are read once); first turn (nothing cached) first token: 3.02 s (2.74-3.59).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 224.7 | 2.74 | 46.9 |
| 1 | 3356 | 628 | 2728 | 432.5 | 6.344 | 65.0 |
| 2 | 4889 | 3420 | 1469 | 360.8 | 4.105 | 61.5 |
| 3 | 8133 | 4952 | 3181 | 431.8 | 7.403 | 58.9 |
| 4 | 10202 | 8196 | 2006 | 394.7 | 5.117 | 74.7 |
| 5 | 12919 | 10236 | 2683 | 419.8 | 6.426 | 63.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 468 s.
