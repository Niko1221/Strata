# Community benchmark on NVIDIA GeForce RTX 4080 SUPER

Measured on 2026-10-14 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4080 SUPER, 16 GiB VRAM
- CPU: AMD Ryzen 7 7800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 63.2 GiB; storage: not measured
- OS: Windows 11 (10.0.26300); GPU driver: 616.92; CUDA: not read
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 196608 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 196608 --kv q4_0 --kv-grow --vision --vram-reserve-mib 350 --pcie-frac 0.35 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 88.6 (83.5-115.5) | 0.34 (0.30-0.39) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 2164.5 (2136.2-2165.6) | not measured | 1.90 (1.90-1.90) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3455.9 (3455.4-3456.2) | not measured | 4.68 (4.67-4.69) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 3489.4 (3488.6-3492.4) | not measured | 9.24 (9.22-9.24) |
| agent session, follow-up turns | 14563 (14553-14566) at the last turn | see below | up to 64 | 3 | see below | 98.1 (82.8-119.9) | 1.85 (1.78-1.88) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.6-68.8); earlier-turn tokens read again per session: 4 (3-74) (earlier turns that the cache did not hold; the 13729 (13729-13729) new tokens of the files are read once); first turn (nothing cached) first token: 1.02 s (1.01-1.03).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 625.5 | 1.015 | 67.2 |
| 1 | 3750 | 629 | 3121 | 1681.2 | 1.876 | 83.6 |
| 2 | 5283 | 3814 | 1469 | 833.6 | 1.782 | 98.0 |
| 3 | 8727 | 5346 | 3381 | 1823.1 | 1.88 | 85.6 |
| 4 | 11638 | 8776 | 2862 | 1566.5 | 1.851 | 119.9 |
| 5 | 14553 | 11654 | 2899 | 1591.7 | 1.845 | 85.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 105 s.

## Notes added by the submitter

- Storage for `Strata-data`: Lexar NM790 4 TB (NVMe SSD)
- GPU PCIe link: PCIe 4.0 x16 (gen 4 / width 16, current and max, read under load)
- CUDA: 13.4 (driver 616.92)
- GPU under load (`nvidia-smi` during generation): 99-100% utilization, 190-240 W of 320 W, 60-70 °C, not power or thermal limited
- Background load: none (headless machine, session locked, accessed over SSH; Sunshine idle). No power limit set.
- Calibration: `START-HERE.bat --calibrate` kept `--pcie-frac 0.35 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97`.
- Vision: the image encoder runs on the CPU (`"vision": {"gpu": false}`). This is the main difference from the previous runs, see below.

### How this configuration was reached (same machine, same day, all runs `BENCH.bat --long`, medians)

| Run | Change | Chat decode tok/s | Agent decode tok/s | 4K prompt tok/s | 16K prompt tok/s | 32K prompt tok/s | Agent TTFT s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | install defaults, vision on GPU | 63.0 | 63.5 | 1590 | 2128 | 2152 | 2.36 |
| B | A + calibration | 65.7 | 68.1 | 1590 | 2125 | 2150 | 2.37 |
| C | B + `--adapt-async 1` | 67.1 | 66.8 | 1591 | 2127 | 2149 | 2.37 |
| D | B + `STRATA_PIN_BY_NEED=1` | 69.8 | 69.7 | 1416 | 1897 | 1904 | 2.53 |
| E | B + both opt-ins | 74.9 | 70.5 | 1412 | 1898 | 1903 | 2.54 |
| **F (this report)** | **B + vision on CPU** | **88.6** | **98.1** | **2165** | **3456** | **3489** | **1.85** |

Engine facts from `results.json`:

| | Vision on GPU (A/B) | Vision on CPU (F) |
| --- | ---: | ---: |
| `expert_slots` | 4023 | 4552 |
| `expert_cache_mib` | 7862 | 8888 |
| `vram_free_mib` | 61 | 87 |

Configuration F was measured twice in separate engine starts: prompt throughput within 0.3%, decode within 5%.

Real-world check after the benchmark (configuration F, agent client, ~90K tokens already in context):
3,397 generated tokens in 63 s, **97.3 tok/s**, expert cache 81.8% hit (+5.2% of routed experts over PCIe).
Images still work with the encoder on the CPU.
