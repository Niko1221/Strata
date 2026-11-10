# Community benchmark on AMD Radeon

Measured on 2026-11-10 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 12 GiB VRAM
- CPU: AMD Ryzen 5 3600 6-Core Processor (6 cores, 10 threads); installed RAM: 56.0 GiB; storage: nvme0n1 INTEL SSDPEKNW010T8                      nvme
- OS: Linux 7.0.14-26-pve; GPU driver: not read; CUDA (driver supports up to): not read
- Strata: project 0.1.59, commit d2f09647; engine 0.1.59
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq2_xs
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
component:strata-engine --pack iq2_xs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 27.3 (24.9-29.7) | 1.41 (1.24-1.73) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 331.1 (326.8-331.3) | not measured | 12.18 (12.17-12.34) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 351.5 (351.3-351.7) | not measured | 45.73 (45.71-45.76) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 350.1 (349.8-350.5) | not measured | 91.59 (91.49-91.68) |
| agent session, follow-up turns | 13591 (13574-13609) at the last turn | see below | up to 64 | 3 | see below | 33.2 (25.8-37.5) | 9.00 (5.54-10.69) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.8 % (69.8-69.9); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 12730 (12730-12730) new tokens of the files are read once); first turn (nothing cached) first token: 3.73 s (3.55-3.78).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 173.5 | 3.554 | 27.7 |
| 1 | 3832 | 627 | 3205 | 303.1 | 10.618 | 29.1 |
| 2 | 5364 | 3895 | 1469 | 253.3 | 5.843 | 25.8 |
| 3 | 8604 | 5422 | 3182 | 298.9 | 10.691 | 33.6 |
| 4 | 10836 | 8667 | 2169 | 276.6 | 7.886 | 35.8 |
| 5 | 13609 | 10899 | 2710 | 293.7 | 9.269 | 31.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 683 s.
