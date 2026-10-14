# Community benchmark on NVIDIA GeForce RTX 5090

Measured on 2026-10-14 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090, 32 GiB VRAM
- CPU: Intel(R) Core(TM) Ultra 7 265K (20 cores, 20 threads); installed RAM: 125.2 GiB 5200 MT/s; storage: btrfs WD_BLACK SN850X 4000GB
- OS: Linux 7.1.8-200.fc44.x86_64; GPU driver: 610.57.04; CUDA: CUDA UMD Version: 13.3
- Strata: project 0.1.44, commit not a git checkout; engine 0.1.44
- Background workloads and power limits: not recorded
- Power Limit 575W

## Model and configuration

- Model and quantization: qwen3.8-flash-next-nvfp4-nvfp4-nvidia
- Context: 65536 tokens; parallel slots: 1
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 112.4 (94.6-133.5) | 0.30 (0.25-0.33) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 3151.2 (3080.2-3156.6) | not measured | 1.29 (1.29-1.32) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 6664.7 (6394.3-6668.6) | not measured | 2.43 (2.42-2.53) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 6740.2 (6718.6-6761.4) | not measured | 4.78 (4.76-4.79) |
| agent session, follow-up turns | 14548 (14521-14554) at the last turn | see below | up to 64 | 3 | see below | 150.9 (98.1-233.8) | 1.35 (1.24-1.46) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.5-68.7); earlier-turn tokens read again per session: 2 (2-74) (earlier turns that the cache did not hold; the 13728 (13728-13729) new tokens of the files are read once); first turn (nothing cached) first token: 0.70 s (0.69-0.71).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 879.0 | 0.707 | 82.6 |
| 1 | 3741 | 620 | 3121 | 2173.9 | 1.452 | 98.1 |
| 2 | 5274 | 3734 | 1540 | 1186.2 | 1.314 | 111.8 |
| 3 | 8682 | 5302 | 3380 | 2344.6 | 1.459 | 139.9 |
| 4 | 11608 | 8745 | 2863 | 2046.2 | 1.417 | 150.9 |
| 5 | 14554 | 11655 | 2899 | 2108.8 | 1.393 | 135.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 68 s.
