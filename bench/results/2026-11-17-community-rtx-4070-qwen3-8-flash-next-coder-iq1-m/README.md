# Community benchmark on NVIDIA GeForce RTX 4070

Measured on 2026-11-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4070, 12 GiB VRAM
- CPU: AMD Ryzen 7 5800XT 8-Core Processor (8 cores, 16 threads); installed RAM: 60.7 GiB (4 x 16 GB DDR4, two Corsair CMK32GX4M2E3200C16 kits, running at 3000 MT/s with the BIOS memory profile (A-XMP) on and the speed set by hand: 3200 MT/s was not stable; read with `dmidecode`); storage: nvme0n1 Samsung SSD 990 PRO 2TB nvme
- OS: Linux 7.0.0-43-generic; GPU driver: 580.178.04; CUDA (driver supports up to): 13.0; engine built for CUDA 13.0
- Strata: project 0.1.59.1, commit 13eedf55; engine 0.1.59
- GPU detail (after the run, near idle): power_limit_w=200.00, power_draw_w=150.16, pcie=gen4 x16, pstate=P2, temp_c=54
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-coder-iq1_m
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack coder-iq1_m --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-coder.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --vision --pcie-frac 0.55 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 57.7 (50.6-79.6) | 0.66 (0.59-0.73) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1986.8 (1951.7-1991.7) | not measured | 2.05 (2.04-2.09) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2262.4 (2255.5-2269.5) | not measured | 7.12 (7.10-7.15) |
| agent session, follow-up turns | 13782 (13754-13850) at the last turn | see below | up to 64 | 3 | see below | 64.4 (52.9-73.9) | 1.73 (1.49-2.06) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.8 % (69.8-69.9); earlier-turn tokens read again per session: 75 (70-144) (earlier turns that the cache did not hold; the 12989 (12989-12989) new tokens of the files are read once); first turn (nothing cached) first token: 1.32 s (1.18-1.41).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 528.0 | 1.177 | 48.7 |
| 1 | 4074 | 631 | 3443 | 1861.6 | 1.877 | 55.5 |
| 2 | 5606 | 4067 | 1539 | 977.3 | 1.603 | 67.0 |
| 3 | 8852 | 5599 | 3253 | 1598.4 | 2.064 | 61.2 |
| 4 | 11092 | 8903 | 2189 | 1319.4 | 1.688 | 71.2 |
| 5 | 13850 | 11141 | 2709 | 1456.3 | 1.889 | 52.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. This is a rerun of the 2133 MT/s report with only the memory speed changed (same engine 0.1.59, commit 13eedf55 and configuration); `python tools/strata_bench.py --compare` gives the differences. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 95 s.
