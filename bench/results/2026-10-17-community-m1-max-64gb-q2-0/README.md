# Community benchmark on Apple M1 Max (Metal), Q2_0 resident

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Apple M1 Max, 32-core GPU, unified memory (Metal)
- CPU: Apple M1 Max, 10 cores (16-inch MacBook Pro); installed RAM: 64.0 GiB; storage: not measured
- OS: macOS (Darwin 27.0.0); no CUDA
- Strata: project 0.1.44.1, commit 38978a6d; engine 0.1.44.1-metal
- Background workloads: none during the runs (no other model servers); power limits not recorded; plugged-in state not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-q2_0
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (metal backend); thinking off for the suite; temperature 0

```text
strata --pack q2_0 --native Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache 64 --prefill auto --spec 2 --mtp rt --max-context 32768 --mmap-experts
```

Environment of the config: `STRATA_METAL_STREAM=0`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 35.1 (33.7-39.2) | 0.88 (0.81-0.96) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 387.5 (368.3-389.7) | not measured | 10.40 (10.35-10.95) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 380.4 (376.7-380.9) | not measured | 42.26 (42.20-42.67) |
| agent session, follow-up turns | 14526 (14503-14528) at the last turn | see below | up to 64 | 3 | see below | 32.7 (21.8-36.9) | 8.68 (4.41-10.13) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.7-68.7); earlier-turn tokens read again per session: 4 (3-4) (earlier turns that the cache did not hold; the 13730 (13730-13730) new tokens of the files are read once); first turn (nothing cached) first token: 2.08 s (2.04-2.23).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 276.8 | 2.233 | 32.9 |
| 1 | 3750 | 627 | 3123 | 334.2 | 9.385 | 36.8 |
| 2 | 5283 | 3814 | 1469 | 306.4 | 4.836 | 34.9 |
| 3 | 8727 | 5346 | 3381 | 335.3 | 10.13 | 26.7 |
| 4 | 11608 | 8746 | 2862 | 326.2 | 8.82 | 29.3 |
| 5 | 14526 | 11627 | 2899 | 333.1 | 8.748 | 26.1 |

### Skipped and failed

- skipped 32K: the context is 32768 tokens

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 349 s.


## Needle-in-a-haystack

`python tools/needle_bench.py --lengths 16k,30k --depths 10,50,90` (128K not run: context was configured to 32768). Raw: `needles.json`.

```text
16k depth  10%: FOUND  (15,156 prompt tokens, 45 s)
  16k depth  50%: FOUND  (15,156 prompt tokens, 41 s)
  16k depth  90%: FOUND  (15,157 prompt tokens, 41 s)
  30k depth  10%: FOUND  (30,614 prompt tokens, 86 s)
  30k depth  50%: FOUND  (30,612 prompt tokens, 84 s)
  30k depth  90%: FOUND  (30,612 prompt tokens, 84 s)
6 of 6 found
```
