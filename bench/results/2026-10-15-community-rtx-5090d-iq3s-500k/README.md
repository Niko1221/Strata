# Community benchmark on NVIDIA GeForce RTX 5090 D

Measured on 2026-10-15 by feigo313 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run as part of the suite.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090 D, 32 GiB VRAM
- CPU: AMD Ryzen 9 9950X 16-Core Processor (16 cores, 32 threads); installed RAM: 95.6 GiB (2x 48 GB DDR5-6000); storage: the model files sit on an NVMe SSD
- OS: Windows 11 (10.0.26300); GPU driver: 616.92; CUDA: not read
- Strata: project 0.1.44.1, commit 4c51f0a3; engine: the official Windows release 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S
- Context: 500000 tokens; parallel slots: 2
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack sc117_iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp strata-rt --kv-resident 32768 --adapt-async 1 --batch 2 --max-context 500000 --vision --vram-reserve-mib 1500 --rope-scaling yarn --rope-scale 2 --resident-experts --kv int8 --spec-min-p 0.5 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

Environment of the config: `STRATA_PIN_BY_NEED=1`

## Method

Run by `python tools/strata_bench.py --clients 3 --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 140.2 (115.4-157.2) | 0.25 (0.21-0.34) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 4021.8 (3901.9-4116.1) | not measured | 1.04 (0.99-1.06) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 5667.8 (5596.5-5686.1) | not measured | 2.87 (2.86-2.92) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 5650.0 (5637.0-5655.5) | not measured | 5.72 (5.72-5.73) |
| agent session, follow-up turns | 14579 (14579-14593) at the last turn | see below | up to 64 | 3 | see below | 162.3 (102.2-203.4) | 0.93 (0.74-1.01) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.4-68.6); earlier-turn tokens read again per session: 73 (73-145) (earlier turns that the cache did not hold; the 13730 (13727-13731) new tokens of the files are read once); first turn (nothing cached) first token: 0.69 s (0.66-0.71).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 951.7 | 0.657 | 76.1 |
| 1 | 3744 | 624 | 3120 | 3285.9 | 1.002 | 102.2 |
| 2 | 5277 | 3807 | 1470 | 1930.4 | 0.812 | 119.2 |
| 3 | 8721 | 5270 | 3451 | 3522.1 | 1.01 | 149.3 |
| 4 | 11631 | 8769 | 2862 | 3102.1 | 0.961 | 169.6 |
| 5 | 14579 | 11682 | 2897 | 3188.8 | 0.954 | 152.7 |

### Concurrency: 3 clients

Total 106.5 tok/s (90.3-112.4); first token p50 0.651 s, p95 2.936 s; request latency p50 2.918 s, p95 3.994 s; errors 0. Server parallel slots: 2 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured by the suite: this is a speed suite. Correctness is covered separately on this install: a 336K-token needle retrieval (the access code hidden at the very start of a 335,402-token document) returns the exact code, and greedy-completion comparisons against the pre-calibration configuration show the code prompt byte-identical with only synonym-level drift in one factual prompt (the same engine session reproduces that drift on its own, so it is run-to-run nondeterminism, not a calibration effect). The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 75 s.
