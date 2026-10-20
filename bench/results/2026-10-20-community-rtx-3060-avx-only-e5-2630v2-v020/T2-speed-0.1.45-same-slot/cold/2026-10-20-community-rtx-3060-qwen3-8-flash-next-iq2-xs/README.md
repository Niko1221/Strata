# Community benchmark on NVIDIA GeForce RTX 3060

Measured on 2026-10-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3060, 12 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2630 v2 @ 2.60GHz (12 cores, 24 threads); installed RAM: 251.8 GiB; storage: not measured
- OS: Linux 7.2.5-200.fc44.x86_64; GPU driver: 610.57.04; CUDA: not read
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq2_xs
- Context: 40960 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack iq2_xs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 40960 --kv int8
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 23.6 (20.4-25.3) | 1.72 (1.56-2.32) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 632.1 (599.9-634.8) | not measured | 6.40 (6.38-6.75) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 950.9 (941.9-965.4) | not measured | 16.94 (16.68-17.11) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 972.7 (966.7-974.5) | not measured | 33.04 (32.97-33.24) |
| agent session, follow-up turns | 14552 (14532-14579) at the last turn | see below | up to 64 | 3 | see below | 27.5 (15.8-39.6) | 6.17 (5.88-6.98) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.4-68.7); earlier-turn tokens read again per session: 73 (3-143) (earlier turns that the cache did not hold; the 13728 (13727-13728) new tokens of the files are read once); first turn (nothing cached) first token: 3.84 s (3.82-3.87).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 159.1 | 3.867 | 21.1 |
| 1 | 3760 | 640 | 3120 | 450.1 | 6.976 | 19.9 |
| 2 | 5293 | 3824 | 1469 | 240.8 | 6.145 | 21.7 |
| 3 | 8716 | 5336 | 3380 | 499.0 | 6.824 | 29.3 |
| 4 | 11616 | 8754 | 2862 | 431.9 | 6.679 | 39.6 |
| 5 | 14532 | 11633 | 2899 | 455.5 | 6.419 | 15.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 380 s.
