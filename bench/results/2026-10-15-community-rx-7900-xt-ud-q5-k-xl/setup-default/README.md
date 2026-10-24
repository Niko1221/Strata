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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 17.2 (8.2-23.0) | 1.61 (1.34-5.62) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 334.9 (233.3-888.1) | not measured | 12.05 (4.57-17.27) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 564.6 (470.8-579.1) | not measured | 31.30 (27.93-34.32) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1018.9 (1018.7-1028.5) | not measured | 31.52 (31.23-31.53) |
| agent session, follow-up turns | 14562 (14543-14596) at the last turn | see below | up to 64 | 3 | see below | 15.1 (5.5-41.8) | 7.17 (4.26-10.69) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.7); earlier-turn tokens read again per session: 72 (2-73) (earlier turns that the cache did not hold; the 13728 (13728-13728) new tokens of the files are read once); first turn (nothing cached) first token: 5.52 s (4.63-6.19).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 99.0 | 6.192 | 9.6 |
| 1 | 3741 | 621 | 3120 | 311.5 | 10.062 | 10.6 |
| 2 | 5274 | 3805 | 1469 | 204.0 | 7.244 | 5.5 |
| 3 | 8718 | 5338 | 3380 | 318.2 | 10.675 | 7.2 |
| 4 | 11611 | 8749 | 2862 | 310.6 | 9.264 | 17.5 |
| 5 | 14543 | 11644 | 2899 | 331.3 | 8.798 | 9.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 535 s.
