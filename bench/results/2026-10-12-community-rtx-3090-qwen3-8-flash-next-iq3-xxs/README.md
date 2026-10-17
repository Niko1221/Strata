# Community benchmark on NVIDIA GeForce RTX 3090

Measured on 2026-10-12 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3090, 24 GiB VRAM
- CPU: Intel(R) Core(TM) i9-10900KF CPU @ 3.70GHz (10 cores, 20 threads); installed RAM: 62.6 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 595.91.07; CUDA: 13.2
- Strata: project 0.1.43, commit 7695859a; engine 0.1.43
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 131072 --kv int8 --vision --vram-reserve-mib 700 --spec-min-p 0.5
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 81.6 (70.0-100.3) | 0.37 (0.31-0.45) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 764.6 (764.4-765.0) | not measured | 5.27 (5.27-5.27) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1476.9 (1473.4-1477.5) | not measured | 10.89 (10.89-10.92) |
| agent session, follow-up turns | 14479 (14477-14513) at the last turn | see below | up to 64 | 3 | see below | 99.0 (61.0-120.5) | 5.15 (4.93-5.24) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.6-68.9); earlier-turn tokens read again per session: 4 (3-73) (earlier turns that the cache did not hold; the 13635 (13634-13635) new tokens of the files are read once); first turn (nothing cached) first token: 1.86 s (1.83-1.89).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 335.1 | 1.834 | 78.2 |
| 1 | 3719 | 637 | 3082 | 593.1 | 5.217 | 85.6 |
| 2 | 5252 | 3782 | 1470 | 299.2 | 4.939 | 86.5 |
| 3 | 8696 | 5316 | 3380 | 651.9 | 5.206 | 89.2 |
| 4 | 11619 | 8759 | 2860 | 559.3 | 5.136 | 99.0 |
| 5 | 14513 | 11667 | 2846 | 553.0 | 5.17 | 93.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 160 s.
