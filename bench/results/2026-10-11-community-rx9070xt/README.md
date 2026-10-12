# Community benchmark on RX 9070 XT

Measured on 2026-10-11 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 9070 XT (gfx1201), 16 GiB VRAM; desktop driven by the iGPU, so the dGPU is free
- CPU: AMD Ryzen 7 9700X 8-Core Processor (8 cores, 8 threads); installed RAM: 121.4 GiB; storage: NVMe (PCIe probe at start: 54.9 GB/s host->device)
- OS: CachyOS (Linux 7.2.8-1-cachyos); GPU driver: amdgpu (in-kernel), ROCm/HIP 7.2.4
- Strata: project 0.1.43, commit 7695859a; engine 0.1.43
- Background workloads and power limits: background app. 10% CPU on 20G RAM beyond the engine; no power-limit changes

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's configuration with two deliberate changes: --vram-reserve-mib 500 (the desktop runs on the iGPU, so the dGPU is otherwise free) and --ple-io direct (default; an A/B against --ple-io ram on this machine showed no gain, see the bench issue). --pcie-frac 0.75, --spec-min-p 0.70 and --pool-workers 4 are the measured values from `setup.sh --calibrate` on this PC, not hand-set. Vision is CPU-side (Q8_0 mmproj, 1024 image tokens) and was not exercised by this suite.

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 65.2 (58.3-77.6) | 0.41 (0.37-0.48) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1396.5 (1343.9-1400.3) | not measured | 2.90 (2.90-3.02) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2375.4 (2287.7-2383.4) | not measured | 6.79 (6.77-7.05) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2725.6 (2723.5-2726.9) | not measured | 11.79 (11.79-11.80) |
| agent session, follow-up turns | 14437 (14411-14481) at the last turn | see below | up to 64 | 3 | see below | 66.0 (51.4-87.9) | 2.56 (1.43-3.01) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.7-68.8); earlier-turn tokens read again per session: 3 (2-5) (earlier turns that the cache did not hold; the 13635 (13634-13635) new tokens of the files are read once); first turn (nothing cached) first token: 2.24 s (1.12-2.33).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 264.1 | 2.329 | 46.6 |
| 1 | 3707 | 624 | 3083 | 1074.5 | 2.894 | 75.3 |
| 2 | 5240 | 3770 | 1470 | 638.0 | 2.329 | 63.3 |
| 3 | 8684 | 5303 | 3381 | 1135.2 | 3.006 | 57.7 |
| 4 | 11563 | 8703 | 2860 | 1003.3 | 2.879 | 51.4 |
| 5 | 14411 | 11565 | 2846 | 1039.7 | 2.765 | 54.0 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 147 s.
