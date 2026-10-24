# Strata v0.1.44.1, RX 7900 XT 20 GB (same-day baseline)

Measured on 2026-10-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XT (gfx1100), 20 GiB VRAM
- CPU: AMD EPYC 7K62 48-Core Processor (48 cores, 48 threads); installed RAM: 128.0 GiB; storage: not measured
- OS: Linux 7.0.14-19-pve; GPU driver: not read; CUDA: not read
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 16.2 (8.4-21.5) | 1.99 (1.69-4.04) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 226.2 (172.1-311.1) | not measured | 17.84 (12.97-23.45) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 833.3 (676.2-1049.3) | not measured | 19.33 (15.39-23.82) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 901.7 (582.1-999.1) | not measured | 35.61 (32.17-55.13) |
| agent session, follow-up turns | 14582 (14570-14582) at the last turn | see below | up to 64 | 3 | see below | 22.3 (5.8-30.0) | 5.79 (4.46-11.02) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.7-68.8); earlier-turn tokens read again per session: 3 (1-4) (earlier turns that the cache did not hold; the 13728 (13727-13728) new tokens of the files are read once); first turn (nothing cached) first token: 3.87 s (3.46-5.10).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 179.5 | 3.458 | 14.1 |
| 1 | 3741 | 621 | 3120 | 529.6 | 5.952 | 6.2 |
| 2 | 5274 | 3804 | 1470 | 181.8 | 8.131 | 5.8 |
| 3 | 8718 | 5337 | 3381 | 308.9 | 11.018 | 11.8 |
| 4 | 11638 | 8776 | 2862 | 434.2 | 6.697 | 22.5 |
| 5 | 14582 | 11683 | 2899 | 478.2 | 6.12 | 14.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 541 s.
