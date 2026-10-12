# Community benchmark on NVIDIA GeForce RTX 5070 Ti

Measured on 2026-10-12 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5070 Ti, 16 GiB VRAM
- CPU: AMD Ryzen 5 5600G with Radeon Graphics (6 cores, 12 threads); installed RAM: 58.8 GiB; storage: not measured
- OS: Linux 7.0.14-17-pve; GPU driver: 615.71.09; CUDA: not read
- Strata: project 0.1.43, commit 7695859a; engine 0.1.43
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 65536 --kv int8 --resident-experts --vram-reserve-mib 1024 --kv-resident 32768
```

## Method

Run by `python tools/strata_bench.py --clients 2 --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 66.3 (55.4-75.1) | 0.49 (0.43-0.56) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1236.6 (1204.7-1242.6) | not measured | 3.27 (3.25-3.35) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1927.2 (1920.7-1931.4) | not measured | 8.36 (8.34-8.39) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1939.8 (1934.1-1940.2) | not measured | 16.56 (16.55-16.61) |
| agent session, follow-up turns | 14498 (14458-14509) at the last turn | see below | up to 64 | 3 | see below | 72.7 (40.4-88.5) | 3.13 (2.90-3.37) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.4-68.8); earlier-turn tokens read again per session: 74 (3-215) (earlier turns that the cache did not hold; the 13633 (13633-13636) new tokens of the files are read once); first turn (nothing cached) first token: 1.47 s (1.47-1.48).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 416.5 | 1.481 | 42.5 |
| 1 | 3704 | 624 | 3080 | 922.8 | 3.361 | 69.9 |
| 2 | 5237 | 3767 | 1470 | 496.7 | 2.982 | 71.6 |
| 3 | 8681 | 5301 | 3380 | 1009.3 | 3.371 | 72.0 |
| 4 | 11595 | 8735 | 2860 | 880.2 | 3.274 | 72.8 |
| 5 | 14458 | 11612 | 2846 | 897.7 | 3.195 | 41.9 |

### Concurrency: 2 clients

Total 48.7 tok/s (43.0-49.4); first token p50 1.997 s, p95 3.876 s; request latency p50 4.303 s, p95 5.777 s; errors 0. Server parallel slots: 1 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 190 s.
