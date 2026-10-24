# Community benchmark on NVIDIA L40S-48Q

Measured on 2026-10-19 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA L40S-48Q, 48 GiB VRAM
- CPU: INTEL(R) XEON(R) PLATINUM 8580 (32 cores, 32 threads); installed RAM: 251.7 GiB; storage: not measured
- OS: Linux 6.8.0-138-generic; GPU driver: 550.127.05; CUDA: 12.4
- Strata: project 0.1.45, commit 477ac13c; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-teste-020-ud-q5_k_xl-256k
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 86.9 (51.4-110.9) | 0.32 (0.27-0.65) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1249.2 (1143.2-1250.3) | not measured | 3.24 (3.24-3.54) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2288.9 (2224.4-2301.1) | not measured | 7.05 (7.01-7.25) |
| agent session, follow-up turns | 14577 (14526-14596) at the last turn | see below | up to 64 | 3 | see below | 98.4 (65.2-109.0) | 3.25 (3.06-4.23) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.8); earlier-turn tokens read again per session: 4 (2-74) (earlier turns that the cache did not hold; the 13717 (13717-13719) new tokens of the files are read once); first turn (nothing cached) first token: 1.66 s (1.64-1.78).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 372.7 | 1.658 | 67.1 |
| 1 | 3741 | 621 | 3120 | 743.0 | 4.229 | 86.6 |
| 2 | 5274 | 3804 | 1470 | 427.9 | 3.465 | 95.5 |
| 3 | 8707 | 5337 | 3370 | 834.6 | 4.067 | 98.3 |
| 4 | 11597 | 8735 | 2862 | 740.8 | 3.898 | 98.4 |
| 5 | 14526 | 11627 | 2899 | 803.1 | 3.639 | 65.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 117 s.
