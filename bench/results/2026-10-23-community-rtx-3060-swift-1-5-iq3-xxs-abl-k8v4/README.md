# Community benchmark on NVIDIA GeForce RTX 3060

Measured on 2026-10-23 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3060, 12 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2697 v2 @ 2.70GHz (24 cores, 48 threads); installed RAM: 127.9 GiB; storage: not measured
- OS: Windows 10 (10.0.26200); GPU driver: 610.88; CUDA: not read
- Strata: project 0.1.48.1, commit 7ddc17a3; engine 0.1.47
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: swift-1.5-iq3_xxs-abl-k8v4
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 48.1 (39.2-54.0) | 0.77 (0.69-0.82) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 466.0 (463.5-467.0) | not measured | 8.66 (8.64-8.72) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 730.5 (722.8-732.4) | not measured | 22.06 (22.00-22.27) |
| agent session, follow-up turns | 14598 (14554-14604) at the last turn | see below | up to 64 | 3 | see below | 50.7 (41.1-65.2) | 7.17 (5.32-8.22) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.5 % (69.4-69.6); earlier-turn tokens read again per session: 73 (3-74) (earlier turns that the cache did not hold; the 13731 (13730-13732) new tokens of the files are read once); first turn (nothing cached) first token: 4.44 s (4.25-4.46).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 145.0 | 4.245 | 30.9 |
| 1 | 4094 | 624 | 3470 | 434.7 | 8.041 | 45.7 |
| 2 | 5627 | 4157 | 1470 | 264.5 | 5.616 | 54.5 |
| 3 | 9060 | 5690 | 3370 | 422.6 | 8.018 | 41.1 |
| 4 | 11823 | 9115 | 2708 | 376.4 | 7.239 | 62.7 |
| 5 | 14604 | 11887 | 2717 | 381.3 | 7.171 | 48.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 270 s.
