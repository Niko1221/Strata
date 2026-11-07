# Community benchmark on 4x NVIDIA GeForce RTX 3090

Measured on 2026-11-06 with `tools/strata_bench.py` (suite 1, 5 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: 4x NVIDIA GeForce RTX 3090, 24 GiB VRAM each (layer split across all four cards)
- CPU: 13th Gen Intel(R) Core(TM) i7-13700K (16 cores, 22 threads); installed RAM: 58.0 GiB; storage: nvme1n1 CT1000T500SSD8                           nvme; nvme0n1 Samsung SSD 990 EVO Plus 4TB             nvme
- OS: Linux 7.0.14-6-pve; GPU driver: 595.84; CUDA (driver supports up to): 13.2
- <host>: project 0.1.57, commit 9f7d977d; engine 0.1.57
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=141.50, pcie=gen4 x16, pstate=P2, temp_c=47
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next
- Context: 262144 tokens; parallel slots: 2
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --clients 1 --long --runs 5` against the HTTP API of a <host> server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 5 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 5 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 5 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 15 | not measured | 102.6 (86.6-143.6) | 0.28 (0.26-0.36) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 5 | 1012.9 (1009.7-1015.8) | not measured | 3.99 (3.98-4.00) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 5 | 2125.7 (2124.3-2129.0) | not measured | 7.58 (7.57-7.58) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 5 | 2768.0 (2750.6-2778.2) | not measured | 11.61 (11.57-11.69) |
| agent session, follow-up turns | 13379 (13351-13389) at the last turn | see below | up to 64 | 5 | see below | 114.6 (99.3-132.0) | 3.56 (2.98-3.84) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.4 % (69.2-69.6); earlier-turn tokens read again per session: 72 (2-143) (earlier turns that the cache did not hold; the 12512 (12511-12513) new tokens of the files are read once); first turn (nothing cached) first token: 1.92 s (1.90-2.04).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 323.2 | 1.901 | 82.4 |
| 1 | 3611 | 620 | 2991 | 821.8 | 3.658 | 110.4 |
| 2 | 5144 | 3675 | 1469 | 493.6 | 2.994 | 111.4 |
| 3 | 8388 | 5208 | 3180 | 843.8 | 3.788 | 117.6 |
| 4 | 10615 | 8451 | 2164 | 647.9 | 3.36 | 106.4 |
| 5 | 13389 | 10678 | 2711 | 767.2 | 3.554 | 120.9 |

### Concurrency: 1 clients

Total 66.1 tok/s (63.5-73.1); first token p50 0.377 s, p95 0.418 s; request latency p50 1.936 s, p95 2.012 s; errors 0. Server parallel slots: 2 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 262 s.
