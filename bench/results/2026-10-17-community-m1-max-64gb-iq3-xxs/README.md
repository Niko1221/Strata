# Community benchmark on Apple M1 Max (Metal), IQ3_XXS resident

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Apple M1 Max, 32-core GPU, unified memory (Metal)
- CPU: Apple M1 Max, 10 cores (16-inch MacBook Pro); installed RAM: 64.0 GiB; storage: not measured
- OS: macOS (Darwin 27.0.0); no CUDA
- Strata: project 0.1.44.1, commit 38978a6d; engine 0.1.44.1-metal
- Background workloads: none during the runs (no other model servers); power limits not recorded; plugged-in state not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (metal backend); thinking off for the suite; temperature 0

```text
strata --pack iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache 64 --prefill auto --spec 2 --suffix-draft 0 --max-context 32768 --mmap-experts
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 32.0 (31.6-32.4) | 1.08 (1.01-1.19) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 321.2 (279.8-321.2) | not measured | 12.55 (12.55-14.41) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 320.3 (317.4-320.4) | not measured | 50.18 (50.17-50.64) |
| agent session, follow-up turns | 14590 (14588-14593) at the last turn | see below | up to 64 | 3 | see below | 31.2 (30.8-32.2) | 10.54 (5.36-12.01) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.8-68.8); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 13731 (13731-13731) new tokens of the files are read once); first turn (nothing cached) first token: 2.64 s (2.56-2.66).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 234.0 | 2.64 | 31.6 |
| 1 | 3745 | 622 | 3123 | 282.5 | 11.099 | 31.2 |
| 2 | 5278 | 3808 | 1470 | 264.9 | 5.591 | 31.2 |
| 3 | 8722 | 5341 | 3381 | 282.5 | 12.011 | 31.1 |
| 4 | 11648 | 8785 | 2863 | 272.0 | 10.569 | 31.1 |
| 5 | 14593 | 11694 | 2899 | 275.2 | 10.575 | 31.0 |

### Skipped and failed

- skipped 32K: the context is 32768 tokens

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 425 s.


## Needle-in-a-haystack

`python tools/needle_bench.py --lengths 16k,30k --depths 10,50,90` (128K not run: context was configured to 32768). Raw: `needles.json`.

```text
16k depth  10%: FOUND  (15,156 prompt tokens, 59 s)
  16k depth  50%: FOUND  (15,156 prompt tokens, 50 s)
  16k depth  90%: FOUND  (15,157 prompt tokens, 50 s)
  30k depth  10%: FOUND  (30,614 prompt tokens, 103 s)
  30k depth  50%: FOUND  (30,612 prompt tokens, 101 s)
  30k depth  90%: FOUND  (30,612 prompt tokens, 101 s)
6 of 6 found
```
