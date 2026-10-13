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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 41.2 (34.8-53.0) | 0.69 (0.58-0.76) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 473.6 (375.1-477.0) | not measured | 8.52 (8.46-10.75) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 543.9 (543.2-546.1) | not measured | 29.56 (29.45-29.60) |
| agent session, follow-up turns | 9038 (8988-9045) at the last turn | see below | up to 64 | 3 | see below | 44.7 (33.7-53.2) | 4.76 (4.51-4.95) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.7 % (70.7-70.9); earlier-turn tokens read again per session: 73 (4-73) (earlier turns that the cache did not hold; the 8206 (8204-8207) new tokens of the files are read once); first turn (nothing cached) first token: 3.43 s (3.42-3.45).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 180.2 | 3.416 | 35.7 |
| 1 | 2263 | 622 | 1641 | 366.7 | 4.511 | 33.7 |
| 2 | 3956 | 2314 | 1642 | 351.2 | 4.711 | 38.6 |
| 3 | 5662 | 4020 | 1642 | 344.4 | 4.804 | 37.0 |
| 4 | 7368 | 5655 | 1713 | 349.1 | 4.945 | 44.4 |
| 5 | 9045 | 7403 | 1642 | 340.4 | 4.862 | 52.2 |

### Concurrency: 1 clients

Total 32.5 tok/s (29.9-32.6); first token p50 0.832 s, p95 0.941 s; request latency p50 3.932 s, p95 4.239 s; errors 0. Server parallel slots: 1 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 269 s.
