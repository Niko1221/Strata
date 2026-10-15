# Community benchmark on AMD Radeon

Measured on 2026-10-15 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 20 GiB VRAM
- CPU: AMD EPYC 7K62 48-Core Processor (48 cores, 48 threads); installed RAM: 128.0 GiB; storage: not measured
- OS: Linux 7.0.14-19-pve; GPU driver: 7.0.14-19-pve; CUDA: not read
- Strata: project 0.1.44.1, commit not a git checkout; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-q5_k_xl
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 22.7 (18.2-31.3) | 1.47 (0.98-1.69) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 576.7 (263.2-603.2) | not measured | 7.01 (6.74-15.33) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 887.7 (766.7-1123.0) | not measured | 18.14 (14.35-21.00) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1115.9 (1001.6-1116.4) | not measured | 28.79 (28.77-32.07) |
| agent session, follow-up turns | 14559 (14539-14589) at the last turn | see below | up to 64 | 3 | see below | 26.1 (11.9-43.2) | 5.36 (4.43-12.94) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.7-68.8); earlier-turn tokens read again per session: 3 (2-5) (earlier turns that the cache did not hold; the 13729 (13728-13730) new tokens of the files are read once); first turn (nothing cached) first token: 3.35 s (3.29-3.63).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 169.6 | 3.63 | 20.3 |
| 1 | 3741 | 621 | 3120 | 482.9 | 6.518 | 12.2 |
| 2 | 5274 | 3805 | 1469 | 257.5 | 5.749 | 11.9 |
| 3 | 8718 | 5338 | 3380 | 494.0 | 6.888 | 20.1 |
| 4 | 11623 | 8761 | 2862 | 472.5 | 6.107 | 43.2 |
| 5 | 14539 | 11640 | 2899 | 501.6 | 5.831 | 24.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 381 s.
