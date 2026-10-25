# Community benchmark on AMD Radeon RX 7900 XT

Measured on 2026-10-25 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XT, 20 GiB VRAM
- CPU: AMD Ryzen 7 5800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 60.7 GiB; storage: not measured
- OS: Linux 7.0.0-34-generic; GPU driver: 6.19.14.31400100; CUDA: not read
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --vision --vram-reserve-mib 700
```

Environment of the config: `STRATA_HIPBLASLT_TUNING=gfx1100-hipblaslt-100202.txt`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 45.6 (35.1-51.7) | 0.72 (0.60-0.80) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1363.2 (1343.8-1366.7) | not measured | 2.98 (2.97-3.02) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1548.7 (1546.9-1548.7) | not measured | 10.40 (10.40-10.41) |
| agent session, follow-up turns | 14537 (14525-14554) at the last turn | see below | up to 64 | 3 | see below | 53.9 (39.5-63.3) | 2.75 (2.17-3.01) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.8); earlier-turn tokens read again per session: 73 (3-74) (earlier turns that the cache did not hold; the 13728 (13728-13729) new tokens of the files are read once); first turn (nothing cached) first token: 1.95 s (1.89-1.95).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 327.8 | 1.885 | 38.4 |
| 1 | 3752 | 631 | 3121 | 1136.4 | 2.775 | 39.5 |
| 2 | 5285 | 3816 | 1469 | 686.4 | 2.169 | 45.7 |
| 3 | 8729 | 5348 | 3381 | 1136.3 | 3.008 | 48.1 |
| 4 | 11636 | 8774 | 2862 | 1041.4 | 2.781 | 63.3 |
| 5 | 14554 | 11655 | 2899 | 1058.8 | 2.771 | 46.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 136 s.
