# Community benchmark on NVIDIA GeForce RTX 3060

Measured on 2026-10-23 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3060, 12 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2697 v2 @ 2.70GHz (24 cores, 48 threads); installed RAM: 127.9 GiB; storage: not measured
- OS: Windows 10 (10.0.26200); GPU driver: 610.88; CUDA: not read
- Strata: project 0.1.48.1, commit 7ddc17a3; engine 0.1.47
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: swift-1.5-iq3_xxs-abl-pw11
- Context: 128000 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 39.2 (34.6-45.1) | 1.03 (0.86-1.17) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 505.2 (503.0-514.0) | not measured | 7.99 (7.86-8.05) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 790.3 (788.1-792.6) | not measured | 20.39 (20.34-20.44) |
| agent session, follow-up turns | 14555 (14503-14589) at the last turn | see below | up to 64 | 3 | see below | 44.5 (19.4-52.0) | 6.31 (4.57-7.25) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.5 % (69.3-69.6); earlier-turn tokens read again per session: 4 (3-144) (earlier turns that the cache did not hold; the 13731 (13730-13731) new tokens of the files are read once); first turn (nothing cached) first token: 4.07 s (3.81-4.38).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 162.2 | 3.811 | 27.0 |
| 1 | 4094 | 624 | 3470 | 482.5 | 7.248 | 38.3 |
| 2 | 5627 | 4157 | 1470 | 302.7 | 4.897 | 41.3 |
| 3 | 9060 | 5690 | 3370 | 478.3 | 7.105 | 40.2 |
| 4 | 11801 | 9093 | 2708 | 424.8 | 6.417 | 52.0 |
| 5 | 14555 | 11838 | 2717 | 434.1 | 6.313 | 43.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 254 s.
