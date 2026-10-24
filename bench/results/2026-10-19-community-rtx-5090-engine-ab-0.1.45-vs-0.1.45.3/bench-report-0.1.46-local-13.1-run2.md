# Community benchmark on NVIDIA GeForce RTX 5090

Measured on 2026-10-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090, 32 GiB VRAM
- CPU: 12th Gen Intel(R) Core(TM) i7-12700KF (12 cores, 20 threads); installed RAM: 95.8 GiB; storage: not measured
- OS: Windows 11 (10.0.26300); GPU driver: 617.42; CUDA: not read
- Strata: project 0.1.46, commit c093ac26; engine 0.1.46
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: swift-1.5-qwen3.8-flash-next-iq3_s
- Context: 524288 tokens; parallel slots: 1
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 161.2 (130.4-183.5) | 0.29 (0.26-0.35) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 3064.2 (2818.2-3104.0) | not measured | 1.35 (1.32-1.45) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 5993.8 (5886.4-6012.0) | not measured | 2.73 (2.70-2.77) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 6188.3 (6076.9-6198.6) | not measured | 5.23 (5.23-5.32) |
| agent session, follow-up turns | 14688 (14675-14698) at the last turn | see below | up to 64 | 3 | see below | 219.6 (139.5-260.5) | 1.23 (1.17-1.61) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.3-69.5); earlier-turn tokens read again per session: 73 (3-74) (earlier turns that the cache did not hold; the 13826 (13826-13828) new tokens of the files are read once); first turn (nothing cached) first token: 1.17 s (1.12-1.18).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 553.6 | 1.119 | 102.1 |
| 1 | 4086 | 622 | 3464 | 2282.4 | 1.553 | 139.5 |
| 2 | 5619 | 4150 | 1469 | 1167.6 | 1.3 | 163.5 |
| 3 | 9052 | 5682 | 3370 | 2301.4 | 1.497 | 158.8 |
| 4 | 11824 | 9045 | 2779 | 1756.9 | 1.614 | 219.6 |
| 5 | 14688 | 11869 | 2819 | 2102.8 | 1.38 | 179.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 69 s.
