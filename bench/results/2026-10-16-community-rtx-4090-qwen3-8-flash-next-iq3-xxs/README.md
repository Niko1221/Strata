# Community benchmark on NVIDIA GeForce RTX 4090

Measured on 2026-10-16 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4090, 24 GiB VRAM
- CPU: 13th Gen Intel(R) Core(TM) i7-13700K (16 cores, 24 threads); installed RAM: 127.8 GiB; storage: not measured
- OS: Windows 10 (10.0.26200); GPU driver: 610.88; CUDA: not read
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --pcie-frac 0.35 --spec-min-p 0.70 --pool-workers 7 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97
```

Environment of the config: `STRATA_ROUTE_TAIL_SKIP=0`

## Method

Run by `python tools/strata_bench.py --clients 4 --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 109.2 (98.4-121.4) | 0.28 (0.25-0.31) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 2634.9 (2604.3-2637.0) | not measured | 1.54 (1.54-1.58) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 4622.9 (4612.0-4634.1) | not measured | 3.50 (3.49-3.51) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 4771.2 (4733.1-4850.1) | not measured | 6.77 (6.66-6.80) |
| agent session, follow-up turns | 14579 (14544-14607) at the last turn | see below | up to 64 | 3 | see below | 132.3 (99.0-166.3) | 1.56 (1.46-1.59) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.4-68.7); earlier-turn tokens read again per session: 75 (74-144) (earlier turns that the cache did not hold; the 13729 (13728-13731) new tokens of the files are read once); first turn (nothing cached) first token: 0.87 s (0.87-0.87).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 722.4 | 0.872 | 111.4 |
| 1 | 3760 | 637 | 3123 | 2023.7 | 1.564 | 107.8 |
| 2 | 5293 | 3823 | 1470 | 1006.5 | 1.493 | 125.3 |
| 3 | 8737 | 5356 | 3381 | 2173.3 | 1.577 | 127.6 |
| 4 | 11663 | 8730 | 2933 | 1904.9 | 1.561 | 145.8 |
| 5 | 14607 | 11708 | 2899 | 1880.1 | 1.565 | 111.3 |

### Concurrency: 4 clients

Total 95.3 tok/s (85.5-96.6); first token p50 2.641 s, p95 4.651 s; request latency p50 3.67 s, p95 5.65 s; errors 0. Server parallel slots: 1 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 99 s.
