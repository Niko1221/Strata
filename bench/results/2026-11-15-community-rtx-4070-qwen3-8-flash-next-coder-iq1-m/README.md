# Community benchmark on NVIDIA GeForce RTX 4070

Measured on 2026-11-15 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4070, 12 GiB VRAM
- CPU: AMD Ryzen 7 5800XT 8-Core Processor (8 cores, 16 threads); installed RAM: 60.7 GiB (4 x 16 GB DDR4, two Corsair CMK32GX4M2E3200C16 kits, running at 2133 MT/s: rated 3200 MT/s, the BIOS memory profile (A-XMP) was off; read with `dmidecode` on 2026-11-16, BIOS unchanged since the runs); storage: nvme0n1 Samsung SSD 990 PRO 2TB nvme
- OS: Linux 7.0.0-43-generic; GPU driver: 580.178.04; CUDA (driver supports up to): 13.0; engine built for CUDA 13.0
- Strata: project 0.1.59.1, commit 13eedf55; engine 0.1.59
- GPU detail (after the run, near idle): power_limit_w=200.00, power_draw_w=157.68, pcie=gen4 x16, pstate=P2, temp_c=56
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 50.8 (48.4-71.1) | 0.75 (0.66-0.81) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1962.2 (1913.7-1966.0) | not measured | 2.07 (2.07-2.13) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2220.5 (2194.2-2220.8) | not measured | 7.26 (7.26-7.35) |
| agent session, follow-up turns | 13709 (13708-13729) at the last turn | see below | up to 64 | 3 | see below | 62.1 (39.8-81.9) | 1.77 (1.44-1.96) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.9 % (69.9-69.9); earlier-turn tokens read again per session: 4 (4-4) (earlier turns that the cache did not hold; the 12989 (12987-12989) new tokens of the files are read once); first turn (nothing cached) first token: 1.48 s (1.32-1.55).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 469.1 | 1.323 | 46.4 |
| 1 | 4074 | 631 | 3443 | 1835.5 | 1.905 | 52.4 |
| 2 | 5607 | 4137 | 1470 | 988.6 | 1.514 | 65.3 |
| 3 | 8797 | 5615 | 3182 | 1766.1 | 1.829 | 62.1 |
| 4 | 10998 | 8809 | 2189 | 1251.0 | 1.775 | 44.8 |
| 5 | 13709 | 11000 | 2709 | 1402.9 | 1.956 | 39.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The memory ran at 2133 MT/s, below its 3200 MT/s rating; most of the model is read from system RAM, so these numbers are likely lower than the same machine with the memory profile on (not measured). The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 95 s.

## Agentic coding trial

The same model and install were also tested on real coding work. Six completed Jira tickets were replayed through an agent pipeline built on Hermes, with thinking on, and the results were compared with the models that originally completed them: qwen3.8-27b, gpt-5.6-sol and Claude Opus 5.5. It passed 3 of the 6 tickets, and all three failures came on tickets whose tests needed real infrastructure. That report also covers per-request throughput and energy measured during the runs.

See [CODER_ST_REPLAY_REPORT.md](CODER_ST_REPLAY_REPORT.md).
