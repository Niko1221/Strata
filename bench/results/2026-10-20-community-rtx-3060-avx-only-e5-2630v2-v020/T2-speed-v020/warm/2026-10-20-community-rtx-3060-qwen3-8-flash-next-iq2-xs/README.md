# Community benchmark on NVIDIA GeForce RTX 3060

Measured on 2026-10-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3060, 12 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2630 v2 @ 2.60GHz (12 cores, 24 threads); installed RAM: 251.8 GiB; storage: not measured
- OS: Linux 7.2.5-200.fc44.x86_64; GPU driver: 610.57.04; CUDA: not read
- Strata: project 0.1.45.3, commit 3d1fb480; engine 0.1.45.3
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 25.8 (22.0-29.6) | 1.68 (1.50-2.29) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 633.1 (605.0-634.8) | not measured | 6.39 (6.37-6.69) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 983.2 (976.1-989.7) | not measured | 16.39 (16.27-16.52) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1002.3 (1001.2-1002.9) | not measured | 32.06 (32.05-32.09) |
| agent session, follow-up turns | 14586 (14584-14599) at the last turn | see below | up to 64 | 3 | see below | 30.1 (20.1-39.1) | 6.21 (5.85-6.96) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.4 % (69.3-69.4); earlier-turn tokens read again per session: 4 (4-74) (earlier turns that the cache did not hold; the 13744 (13743-13744) new tokens of the files are read once); first turn (nothing cached) first token: 3.78 s (3.74-3.79).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 164.5 | 3.742 | 17.7 |
| 1 | 4048 | 630 | 3418 | 494.4 | 6.96 | 20.1 |
| 2 | 5581 | 4111 | 1470 | 242.3 | 6.105 | 23.2 |
| 3 | 9014 | 5644 | 3370 | 496.8 | 6.827 | 31.3 |
| 4 | 11736 | 9053 | 2683 | 412.8 | 6.542 | 36.9 |
| 5 | 14584 | 11777 | 2807 | 441.2 | 6.405 | 29.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 371 s.
