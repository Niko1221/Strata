# Community benchmark on NVIDIA L40S-48Q

Measured on 2026-10-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA L40S-48Q, 48 GiB VRAM
- CPU: INTEL(R) XEON(R) PLATINUM 8580 (32 cores, 32 threads); installed RAM: 503.7 GiB; storage: not measured
- OS: Linux 6.8.0-138-generic; GPU driver: 550.127.05; CUDA: 12.4
- Strata: project 0.1.45, commit 477ac13c; engine 0.1.46
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-teste-021-ud-q2_k_xl-256k
- Context: 262144 tokens; parallel slots: 1
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 136.9 (121.6-201.2) | 0.22 (0.21-0.25) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 2736.4 (2189.2-2737.6) | not measured | 1.49 (1.49-1.86) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3464.6 (3267.7-3471.8) | not measured | 4.66 (4.65-4.94) |
| agent session, follow-up turns | 14564 (14561-14587) at the last turn | see below | up to 64 | 3 | see below | 149.8 (131.1-182.3) | 1.23 (0.86-2.47) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.8); earlier-turn tokens read again per session: 71 (3-74) (earlier turns that the cache did not hold; the 13718 (13717-13719) new tokens of the files are read once); first turn (nothing cached) first token: 1.12 s (0.99-1.12).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 551.8 | 1.124 | 124.6 |
| 1 | 3744 | 624 | 3120 | 1352.5 | 2.333 | 131.1 |
| 2 | 5277 | 3808 | 1469 | 1106.0 | 1.351 | 168.0 |
| 3 | 8710 | 5270 | 3440 | 1406.6 | 2.471 | 131.8 |
| 4 | 11636 | 8774 | 2862 | 1508.1 | 1.927 | 141.1 |
| 5 | 14587 | 11690 | 2897 | 1766.5 | 1.667 | 167.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 62 s.
