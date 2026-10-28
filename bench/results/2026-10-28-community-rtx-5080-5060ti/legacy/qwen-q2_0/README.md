# Community benchmark on NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 5080

Measured on 2026-10-28 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 5080, 32 GiB VRAM
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 95.7 GiB; storage: HP SSD FX900 Pro 4TB SSD/NVMe; HP SSD FX900 Pro 4TB SSD/NVMe
- OS: Windows 11 (10.0.26200); GPU driver: 621.01; CUDA (driver supports up to): 13.5; engine built for CUDA 13.0
- Strata: project 0.1.52, commit 5e8a4cca; engine 0.1.52
- GPU detail (after the run, near idle): power_limit_w=150.00, power_draw_w=32.86, pcie=gen5 x8, pstate=P1, temp_c=48.36
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=95.35, pcie=gen5 x8, pstate=P1, temp_c=48.23
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-q2_0
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack q2_0 --native Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vision --prefill auto:4096 --vram-reserve-mib 2048 --vram-reserve-later-mib 1500 --spec-min-p 0.5
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 116.1 (101.2-160.8) | 0.33 (0.29-0.35) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1571.7 (1544.0-1578.0) | not measured | 2.59 (2.58-2.63) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2398.9 (2368.1-2410.4) | not measured | 6.75 (6.70-6.82) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2747.5 (2738.6-2794.7) | not measured | 11.71 (11.51-11.73) |
| agent session, follow-up turns | 12792 (12728-12810) at the last turn | see below | up to 64 | 3 | see below | 123.4 (87.3-145.7) | 1.91 (1.40-2.33) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.1-69.4); earlier-turn tokens read again per session: 3 (3-142) (earlier turns that the cache did not hold; the 11950 (11949-11951) new tokens of the files are read once); first turn (nothing cached) first token: 1.22 s (1.08-1.28).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 572.3 | 1.08 | 99.2 |
| 1 | 3305 | 628 | 2677 | 1399.6 | 1.97 | 118.9 |
| 2 | 4837 | 3369 | 1468 | 1072.9 | 1.4 | 107.2 |
| 3 | 8081 | 4900 | 3181 | 1527.4 | 2.107 | 115.4 |
| 4 | 10104 | 8140 | 1964 | 1207.9 | 1.662 | 123.4 |
| 5 | 12792 | 10129 | 2663 | 1407.4 | 1.914 | 101.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 119 s.
