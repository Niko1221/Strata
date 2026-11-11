# Community benchmark on NVIDIA GeForce RTX 4080 SUPER

Measured on 2026-11-10 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4080 SUPER, 16 GiB VRAM
- CPU: AMD Ryzen 9 7950X3D 16-Core Processor; installed RAM: 63.7 GiB; storage: KINGSTON SKC3000D2048G SSD/NVMe; Lexar SSD NM710 1TB SSD/NVMe; Lexar SSD NM790 4TB SSD/NVMe; SD Ultra 3D 2TB SSD/SATA
- OS: Windows 11; GPU driver: 621.14; CUDA (driver supports up to): 13.5
- Strata: engine 0.1.59
- GPU detail (after the run, near idle): power_limit_w=320.00, power_draw_w=176.23, pcie=gen4 x16, pstate=P2, temp_c=57.21
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: swift-1.5-iq3_xxs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers, 64 new tokens per turn, 3 sessions

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 | 0 | 100 (100-100) | 9 | not measured | 100.7 (93.1-118.8) | 0.3 (0.2-0.3) |
| 4K prompt, fresh | 4019 | 0 (0-0) | 24 cap | 3 | 2388.8 (2388.1-2400.5) | not measured | 1.7 (1.7-1.7) |
| 16K prompt, fresh | 16058 | 0 (0-0) | 24 cap | 3 | 3627.0 (3581.2-3661.0) | not measured | 4.5 (4.4-4.5) |
| agent session, follow-up turns | 13796 at the last turn | see below | up to 64 | 15 | see below | 109.3 (89.8-141.1) | 1.6 (1.1-1.7) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.0 % (69.792-70.016); earlier-turn tokens read again per session: 73 (4-144); first turn (nothing cached) first token: 1.01 s.

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 657.1 | 0.960 | 83.6 |
| 1 | 4074 | 630 | 3444 | 2133.6 | 1.654 | 100.3 |
| 2 | 5607 | 4137 | 1470 | 1302.4 | 1.167 | 97.2 |
| 3 | 8832 | 5650 | 3182 | 1972.2 | 1.657 | 117.2 |
| 4 | 11011 | 8843 | 2168 | 1642.1 | 1.357 | 130.6 |
| 5 | 13743 | 11034 | 2709 | 1750.6 | 1.588 | 102.3 |

Per-request data: `results.json`.

## Recall checks

`tools/needle_bench.py` on the same instance, lengths 32k,256k, depths 10/50/90 %: all 6 needles found. Raw runs: `needles.json`.

| Length | Depth | Found | Prompt tokens | Word |
| --- | ---: | --- | ---: | --- |
| 32k | 10% | YES | 31653 | meadow-copper-504 |
| 32k | 50% | YES | 31653 | falcon-quartz-940 |
| 32k | 90% | YES | 31654 | willow-cobalt-696 |
| 256k | 10% | YES | 249107 | falcon-saffron-138 |
| 256k | 50% | YES | 249105 | quartz-tundra-528 |
| 256k | 90% | YES | 249105 | quartz-glacier-192 |

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi` and the OS; background load is not recorded.
