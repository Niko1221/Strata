# Community benchmark on NVIDIA GeForce RTX 4080 SUPER

Measured on 2026-11-10 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4080 SUPER, 16 GiB VRAM
- CPU: AMD Ryzen 9 7950X3D 16-Core Processor; installed RAM: 63.7 GiB; storage: KINGSTON SKC3000D2048G SSD/NVMe; Lexar SSD NM710 1TB SSD/NVMe; Lexar SSD NM790 4TB SSD/NVMe; SD Ultra 3D 2TB SSD/SATA
- OS: Windows 11; GPU driver: 621.14; CUDA (driver supports up to): 13.5
- Strata: engine 0.1.59
- GPU detail (after the run, near idle): power_limit_w=320.00, power_draw_w=173.47, pcie=gen4 x16, pstate=P2, temp_c=58.42
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers, 64 new tokens per turn, 3 sessions

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 | 0 | 100 (100-100) | 9 | not measured | 92.2 (80.9-113.6) | 0.3 (0.3-0.3) |
| 4K prompt, fresh | 4019 | 0 (0-0) | 24 cap | 3 | 2066.4 (2055.1-2073.0) | not measured | 2.0 (2.0-2.0) |
| 16K prompt, fresh | 16058 | 0 (0-0) | 24 cap | 3 | 3623.2 (3617.6-3627.9) | not measured | 4.5 (4.5-4.5) |
| agent session, follow-up turns | 13793 at the last turn | see below | up to 64 | 15 | see below | 107.7 (87.7-118.2) | 1.8 (1.3-2.0) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.0 % (69.942-70.062); earlier-turn tokens read again per session: 74 (3-74); first turn (nothing cached) first token: 1.02 s.

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 638.7 | 0.990 | 85.7 |
| 1 | 4078 | 632 | 3446 | 1801.0 | 1.951 | 99.5 |
| 2 | 5611 | 4071 | 1540 | 1167.9 | 1.353 | 87.7 |
| 3 | 8857 | 5675 | 3182 | 1660.7 | 1.951 | 104.5 |
| 4 | 11060 | 8892 | 2168 | 1426.1 | 1.570 | 114.2 |
| 5 | 13793 | 11084 | 2709 | 1553.8 | 1.781 | 93.6 |

Per-request data: `results.json`.

## Recall checks

`tools/needle_bench.py` on the same instance, lengths 32k,128k, depths 10/50/90 %: all 6 needles found. Raw runs: `needles.json`.

| Length | Depth | Found | Prompt tokens | Word |
| --- | ---: | --- | ---: | --- |
| 32k | 10% | YES | 31653 | meadow-copper-504 |
| 32k | 50% | YES | 31653 | falcon-quartz-940 |
| 32k | 90% | YES | 31654 | willow-cobalt-696 |
| 128k | 10% | YES | 123410 | falcon-saffron-138 |
| 128k | 50% | YES | 123408 | quartz-tundra-528 |
| 128k | 90% | YES | 123408 | quartz-glacier-192 |

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi` and the OS; background load is not recorded.
