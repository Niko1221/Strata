# Community benchmark on NVIDIA GeForce RTX 4090

Measured on 2026-11-01 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4090, 48 GiB VRAM
- CPU: Intel(R) Core(TM) i7-14700KF (20 cores, 28 threads); installed RAM: 95.8 GiB; storage: Samsung SSD 990 PRO 2TB SSD/NVMe
- OS: Windows 11 (10.0.26220); GPU driver: 610.88; CUDA (driver supports up to): 13.3; engine built for CUDA 13.0
- Strata: project 0.1.55, commit d6e617a3; engine 0.1.50
- GPU detail (after the run, near idle): power_limit_w=450.00, power_draw_w=368.51, pcie=gen4 x16, pstate=P0, temp_c=58
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-gyro-gyro-m
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack gyro-gyro-m --native Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf --ple-gguf Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --spec-min-p 0.5
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 165.5 (146.5-221.9) | 0.24 (0.22-0.25) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 3796.5 (3790.8-3808.4) | not measured | 1.07 (1.07-1.08) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 4873.1 (4761.3-4879.5) | not measured | 3.32 (3.31-3.40) |
| agent session, follow-up turns | 12929 (12927-12934) at the last turn | see below | up to 64 | 3 | see below | 168.2 (146.0-195.9) | 0.85 (0.61-2.89) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (55.3-69.4); earlier-turn tokens read again per session: 72 (3-5588) (earlier turns that the cache did not hold; the 12064 (12063-12064) new tokens of the files are read once); first turn (nothing cached) first token: 0.61 s (0.53-0.65).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 987.7 | 0.649 | 77.1 |
| 1 | 3359 | 0 | 3359 | 3074.3 | 1.997 | 158.8 |
| 2 | 4892 | 3423 | 1469 | 2445.5 | 0.617 | 168.2 |
| 3 | 8136 | 0 | 8136 | 4317.8 | 2.886 | 148.1 |
| 4 | 10201 | 8196 | 2005 | 2777.0 | 0.74 | 178.8 |
| 5 | 12934 | 10251 | 2683 | 3102.5 | 0.896 | 180.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 50 s.
