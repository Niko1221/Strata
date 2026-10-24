# Community benchmark on NVIDIA GeForce RTX 3060

Measured on 2026-10-16 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 17.1 (15.1-19.4) | 2.18 (1.94-2.56) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 178.3 (177.7-179.0) | not measured | 22.59 (22.50-22.66) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 342.1 (341.8-342.3) | not measured | 46.99 (46.97-47.03) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 352.4 (351.8-352.4) | not measured | 91.03 (91.03-91.18) |
| agent session, follow-up turns | 14561 (14559-14565) at the last turn | see below | up to 64 | 3 | see below | 20.5 (15.1-26.6) | 21.89 (20.91-22.36) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.4-68.7); earlier-turn tokens read again per session: 73 (3-144) (earlier turns that the cache did not hold; the 13729 (13729-13729) new tokens of the files are read once); first turn (nothing cached) first token: 8.86 s (8.76-8.98).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 69.9 | 8.756 | 12.4 |
| 1 | 3741 | 620 | 3121 | 141.1 | 22.165 | 15.1 |
| 2 | 5274 | 3734 | 1540 | 73.6 | 20.968 | 18.7 |
| 3 | 8718 | 5338 | 3380 | 152.7 | 22.181 | 20.2 |
| 4 | 11627 | 8765 | 2862 | 131.4 | 21.838 | 24.6 |
| 5 | 14565 | 11666 | 2899 | 133.1 | 21.827 | 16.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 983 s.
