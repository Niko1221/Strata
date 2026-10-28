# Community benchmark on NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 5080

Measured on 2026-10-28 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 5080, 32 GiB VRAM
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 95.7 GiB; storage: HP SSD FX900 Pro 4TB SSD/NVMe; HP SSD FX900 Pro 4TB SSD/NVMe
- OS: Windows 11 (10.0.26200); GPU driver: 621.01; CUDA (driver supports up to): 13.5; engine built for CUDA 13.0
- Strata: project 0.1.52, commit 5e8a4cca; engine 0.1.52
- GPU detail (after the run, near idle): power_limit_w=150.00, power_draw_w=32.47, pcie=gen5 x8, pstate=P1, temp_c=48.22
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=91.41, pcie=gen5 x8, pstate=P1, temp_c=47.70
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: swift-1.5-iq2_xs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack swift-iq2_xs --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto:4096 --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 2048 --vram-reserve-later-mib 1500 --spec-min-p 0.5
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 109.8 (104.5-153.8) | 0.29 (0.27-0.32) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1430.2 (1423.9-1448.3) | not measured | 2.84 (2.81-2.85) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2266.1 (2253.5-2280.1) | not measured | 7.12 (7.08-7.17) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2647.4 (2634.5-2665.3) | not measured | 12.15 (12.07-12.21) |
| agent session, follow-up turns | 12819 (12799-12820) at the last turn | see below | up to 64 | 3 | see below | 135.0 (119.6-168.3) | 2.21 (1.64-2.66) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.3-69.5); earlier-turn tokens read again per session: 71 (1-73) (earlier turns that the cache did not hold; the 11946 (11946-11949) new tokens of the files are read once); first turn (nothing cached) first token: 1.27 s (1.18-1.30).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 532.2 | 1.179 | 96.3 |
| 1 | 3298 | 623 | 2675 | 1228.0 | 2.213 | 119.6 |
| 2 | 4827 | 3360 | 1467 | 904.1 | 1.658 | 123.7 |
| 3 | 8071 | 4891 | 3180 | 1281.1 | 2.55 | 133.7 |
| 4 | 10099 | 8134 | 1965 | 1053.7 | 1.906 | 123.0 |
| 5 | 12819 | 10159 | 2660 | 1215.8 | 2.231 | 153.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 127 s.
