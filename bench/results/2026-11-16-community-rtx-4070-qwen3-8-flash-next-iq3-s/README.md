# Community benchmark on NVIDIA GeForce RTX 4070

Measured on 2026-11-16 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4070, 12 GiB VRAM
- CPU: AMD Ryzen 7 5800XT 8-Core Processor (8 cores, 16 threads); installed RAM: 60.7 GiB (4 x 16 GB DDR4, two Corsair CMK32GX4M2E3200C16 kits, running at 2133 MT/s: rated 3200 MT/s, the BIOS memory profile (A-XMP) was off; read with `dmidecode` on 2026-11-16, BIOS unchanged since the runs); storage: nvme0n1 Samsung SSD 990 PRO 2TB nvme
- OS: Linux 7.0.0-43-generic; GPU driver: 580.178.04; CUDA (driver supports up to): 13.0; engine built for CUDA 13.0
- Strata: project 0.1.59.1, commit 13eedf55; engine 0.1.59
- GPU detail (after the run, near idle): power_limit_w=200.00, power_draw_w=110.62, pcie=gen4 x16, pstate=P2, temp_c=53
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 131072 --kv int8 --spec-min-p 0.5 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 43.4 (41.6-56.8) | 0.80 (0.69-0.88) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1545.9 (1543.7-1558.5) | not measured | 2.63 (2.61-2.64) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1701.2 (1694.0-1712.6) | not measured | 9.47 (9.41-9.51) |
| agent session, follow-up turns | 13847 (13802-13865) at the last turn | see below | up to 64 | 3 | see below | 48.7 (41.1-56.3) | 2.40 (2.06-2.84) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.0 % (70.0-70.1); earlier-turn tokens read again per session: 3 (3-74) (earlier turns that the cache did not hold; the 12988 (12988-12989) new tokens of the files are read once); first turn (nothing cached) first token: 2.23 s (2.07-2.31).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 298.4 | 2.071 | 35.7 |
| 1 | 4078 | 636 | 3442 | 1406.2 | 2.48 | 41.1 |
| 2 | 5611 | 4142 | 1469 | 649.4 | 2.295 | 42.1 |
| 3 | 8857 | 5674 | 3183 | 1303.0 | 2.481 | 47.7 |
| 4 | 11108 | 8919 | 2189 | 849.6 | 2.614 | 54.1 |
| 5 | 13865 | 11156 | 2709 | 966.9 | 2.838 | 52.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The memory ran at 2133 MT/s, below its 3200 MT/s rating; most of the model is read from system RAM, so these numbers are likely lower than the same machine with the memory profile on (not measured). The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 133 s.
