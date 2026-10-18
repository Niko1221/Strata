# Community benchmark on Apple M1 Max (Metal), IQ3_S streamed

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Apple M1 Max, 32-core GPU, unified memory (Metal)
- CPU: Apple M1 Max, 10 cores (16-inch MacBook Pro); installed RAM: 64.0 GiB; storage: not measured
- OS: macOS (Darwin 27.0.0); no CUDA
- Strata: project 0.1.44.1, commit 38978a6d; engine 0.1.44.1-metal
- Background workloads: none during the runs (no other model servers); power limits not recorded; plugged-in state not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (metal backend); thinking off for the suite; temperature 0

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache 64 --prefill auto --spec 2 --suffix-draft 0 --max-context 32768 --mmap-experts
```

Environment of the config: `STRATA_METAL_STREAM=1`, `STRATA_METAL_ARENA_GB=38`, `STRATA_METAL_LEAVE_SHARE=0.3`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 23.1 (19.6-24.1) | 1.55 (1.27-1.68) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 290.3 (282.8-290.5) | not measured | 13.89 (13.88-14.26) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 290.0 (288.5-291.0) | not measured | 55.43 (55.23-55.72) |
| agent session, follow-up turns | 14569 (14544-14574) at the last turn | see below | up to 64 | 3 | see below | 23.2 (19.5-23.9) | 11.65 (6.29-13.95) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.7-68.8); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 13730 (13730-13730) new tokens of the files are read once); first turn (nothing cached) first token: 2.88 s (2.88-3.03).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 204.7 | 3.026 | 19.2 |
| 1 | 3743 | 620 | 3123 | 240.8 | 13.025 | 19.5 |
| 2 | 5276 | 3806 | 1470 | 226.8 | 6.533 | 22.3 |
| 3 | 8720 | 5339 | 3381 | 243.2 | 13.952 | 22.6 |
| 4 | 11618 | 8756 | 2862 | 246.8 | 11.647 | 22.9 |
| 5 | 14544 | 11645 | 2899 | 249.2 | 11.684 | 22.6 |

### Skipped and failed

- skipped 32K: the context is 32768 tokens

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 501 s.


## Needle-in-a-haystack

`python tools/needle_bench.py --lengths 16k,30k --depths 10,50,90` (128K not run: context was configured to 32768). Raw: `needles.json`.

```text
16k depth  10%: FOUND  (15,156 prompt tokens, 78 s)
  16k depth  50%: FOUND  (15,156 prompt tokens, 57 s)
  16k depth  90%: FOUND  (15,157 prompt tokens, 57 s)
  30k depth  10%: FOUND  (30,614 prompt tokens, 117 s)
  30k depth  50%: FOUND  (30,612 prompt tokens, 115 s)
  30k depth  90%: FOUND  (30,612 prompt tokens, 115 s)
6 of 6 found
```
