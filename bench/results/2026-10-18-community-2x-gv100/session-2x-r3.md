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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 67.1 (61.3-89.9) | 0.42 (0.38-0.47) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1070.8 (1070.2-1071.0) | not measured | 3.78 (3.78-3.78) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1709.2 (1708.1-1715.2) | not measured | 9.44 (9.40-9.44) |
| agent session, follow-up turns | 14534 (14487-14555) at the last turn | see below | up to 64 | 3 | see below | 68.8 (51.2-76.1) | 4.30 (3.40-4.54) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.5-68.7); earlier-turn tokens read again per session: 3 (3-73) (earlier turns that the cache did not hold; the 13718 (13717-13718) new tokens of the files are read once); first turn (nothing cached) first token: 2.42 s (2.40-2.43).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 256.6 | 2.4 | 42.3 |
| 1 | 3741 | 620 | 3121 | 729.4 | 4.307 | 63.4 |
| 2 | 5274 | 3805 | 1469 | 436.0 | 3.398 | 65.5 |
| 3 | 8707 | 5337 | 3370 | 748.1 | 4.535 | 71.4 |
| 4 | 11611 | 8749 | 2862 | 677.8 | 4.254 | 69.2 |
| 5 | 14534 | 11635 | 2899 | 681.1 | 4.288 | 54.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the <host>).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the <host>, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 144 s.
