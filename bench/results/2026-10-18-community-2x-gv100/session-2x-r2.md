# Community benchmark on Quadro GV100

Measured on 2026-10-18 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Quadro GV100, 32 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-1650 v4 @ 3.60GHz (6 cores, 12 threads); installed RAM: 125.7 GiB; storage: not measured
- OS: Linux 6.8.0-117-generic; GPU driver: 580.159.03; CUDA: 13.0
- Strata: project 0.1.45, commit 477ac13c; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-q4_k_xl
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata <host> (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 65.6 (53.0-89.3) | 0.45 (0.38-0.51) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1070.6 (1068.5-1071.7) | not measured | 3.78 (3.78-3.79) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1708.4 (1702.4-1710.4) | not measured | 9.44 (9.43-9.47) |
| agent session, follow-up turns | 14537 (14536-14545) at the last turn | see below | up to 64 | 3 | see below | 68.5 (54.4-81.2) | 4.29 (3.40-4.58) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.4-68.7); earlier-turn tokens read again per session: 3 (3-144) (earlier turns that the cache did not hold; the 13717 (13717-13718) new tokens of the files are read once); first turn (nothing cached) first token: 2.42 s (2.40-2.42).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 256.7 | 2.4 | 48.6 |
| 1 | 3741 | 621 | 3120 | 719.9 | 4.369 | 61.0 |
| 2 | 5274 | 3805 | 1469 | 436.0 | 3.4 | 61.4 |
| 3 | 8707 | 5337 | 3370 | 744.2 | 4.561 | 54.4 |
| 4 | 11621 | 8759 | 2862 | 672.8 | 4.287 | 81.2 |
| 5 | 14536 | 11637 | 2899 | 680.6 | 4.292 | 60.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the <host>).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the <host>, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 173 s.
