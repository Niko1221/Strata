# Community benchmark on AMD Instinct MI50 32GB + AMD Radeon VII 16GB

Measured on 2026-10-11 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Instinct MI50 32GB + AMD Radeon VII 16GB, 48 GiB VRAM
- CPU: 12th Gen Intel(R) Core(TM) i5-12400F (6 cores, 12 threads); installed RAM: 61.6 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 7.0.0-38-generic; CUDA: not read
- Strata: project 0.1.43, commit not a git checkout; engine 0.1.43
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: Qwen3.8-Flash-Next-Uncensored-IQ3_XXS
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --clients 1` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 44.0 (41.9-62.1) | 0.59 (0.54-0.75) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 546.9 (546.5-549.5) | not measured | 7.38 (7.35-7.39) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 621.8 (619.7-622.0) | not measured | 25.87 (25.86-25.96) |
| agent session, follow-up turns | 9043 (9024-9067) at the last turn | see below | up to 64 | 3 | see below | 41.7 (34.9-62.0) | 4.18 (3.86-10.85) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 71.0 % (71.0-71.0); earlier-turn tokens read again per session: 4 (4-4) (earlier turns that the cache did not hold; the 8206 (8204-8207) new tokens of the files are read once); first turn (nothing cached) first token: 2.57 s (2.54-2.62).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 244.8 | 2.572 | 36.2 |
| 1 | 2264 | 624 | 1640 | 425.6 | 3.892 | 41.5 |
| 2 | 3970 | 2327 | 1643 | 395.9 | 4.192 | 42.8 |
| 3 | 5676 | 4033 | 1643 | 399.6 | 4.152 | 43.1 |
| 4 | 7382 | 5739 | 1643 | 396.8 | 4.184 | 41.6 |
| 5 | 9067 | 7425 | 1642 | 396.4 | 4.184 | 41.7 |

### Concurrency: 1 clients

Total 31.7 tok/s (13.3-32.0); first token p50 0.816 s, p95 0.868 s; request latency p50 4.035 s, p95 9.077 s; errors 0. Server parallel slots: 1 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 254 s.
