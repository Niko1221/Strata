# Community benchmark on NVIDIA GeForce RTX 5090 Laptop GPU

Measured on 2026-11-23 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090 Laptop GPU, 24 GiB VRAM
- CPU: AMD Ryzen 9 9955HX3D 16-Core Processor (16 cores, 32 threads); installed RAM: 61.7 GiB; storage: Samsung SSD 990 PRO 1TB SSD/NVMe; WD Blue SN5100 500GB SSD/NVMe
- OS: Windows 11 (10.0.26300); GPU driver: 610.88; CUDA (driver supports up to): 13.3
- Strata: project 0.1.65, commit not a git checkout; engine strata-inkling-0.1.65
- GPU detail (after the run, near idle): power_limit_w=[N/A], power_draw_w=149.52, pcie=gen4 x8, pstate=P2, temp_c=82
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: inkling-2-mini-sq-20
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-inkling.exe --native Inkling-2-Mini-SQ-20.strata --max-context 131072 --vision Inkling-2-Mini-Vision.strata --vram-experts-gb 15.68 --ram-experts-gb 0.00
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
| short chat | 44 (41-48) | 0 | 100 (100-100) | 2 | not measured | 27.7 (27.0-28.4) | 3.51 (3.20-3.82) |
| 4K prompt, fresh | not measured | not measured | 24 cap | 0 | not measured | not measured | not measured |
| 16K prompt, fresh | not measured | not measured | 24 cap | 0 | not measured | not measured | not measured |
| 32K prompt, fresh | not measured | not measured | 24 cap | 0 | not measured | not measured | not measured |
| agent session, follow-up turns | not measured at the last turn | see below | up to 64 | 3 | see below | not measured | not measured |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: not measured; earlier-turn tokens read again per session: 0 (0-0) (earlier turns that the cache did not hold; the 0 (0-0) new tokens of the files are read once); first turn (nothing cached) first token: 2.04 s (2.03-2.14).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 553 | 0 | 553 | 2000.7 | 2.031 | 35.1 |

### Skipped and failed

- failed chat: no token came back
- failed chat: no token came back
- failed chat: no token came back
- failed chat: no token came back
- failed chat: no token came back
- failed chat: no token came back
- failed chat: no token came back
- failed 4K: no token came back
- failed 4K: no token came back
- failed 4K: no token came back
- failed 16K: no token came back
- failed 16K: no token came back
- failed 16K: no token came back
- failed agent: no token came back
- failed agent: no token came back
- failed agent: no token came back
- failed 32K: no token came back
- failed 32K: no token came back
- failed 32K: no token came back

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 162 s.
