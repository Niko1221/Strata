# Community benchmark on NVIDIA GeForce RTX 5090 Laptop GPU

Measured on 2026-11-23 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090 Laptop GPU, 24 GiB VRAM
- CPU: AMD Ryzen 9 9955HX3D 16-Core Processor (16 cores, 32 threads); installed RAM: 61.7 GiB; storage: Samsung SSD 990 PRO 1TB SSD/NVMe; WD Blue SN5100 500GB SSD/NVMe
- OS: Windows 11 (10.0.26300); GPU driver: 610.88; CUDA (driver supports up to): 13.3
- Strata: project 0.1.65, commit not a git checkout; engine 0.1.65
- GPU detail (after the run, near idle): power_limit_w=[N/A], power_draw_w=37.79, pcie=gen4 x8, pstate=P0, temp_c=80
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-q2_0
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack q2_0 --native Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.7 --mtp rt --mtp-draft-vocab draft_vocab_de.bin --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 2048 --mmap-experts --pool-workers 10 --pcie-frac 0 --conversation-cache-mib 16384 --conversation-cache-slots 4 --control-vector-scaled Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer
```

Environment of the config: `STRATA_ROUTE_TAIL_SKIP=7`

## Method

Run by `python tools/strata_bench.py --clients 2 --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 100.2 (68.5-159.1) | 0.25 (0.20-0.42) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1829.7 (706.6-1837.8) | not measured | 2.22 (2.22-5.73) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3010.6 (2941.0-3064.7) | not measured | 5.37 (5.28-5.50) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 3184.8 (3150.9-3219.4) | not measured | 10.11 (10.00-10.22) |
| agent session, follow-up turns | 9007 (8932-9010) at the last turn | see below | up to 64 | 3 | see below | 138.7 (98.9-167.0) | 1.75 (1.74-1.79) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.9 % (70.7-70.9); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 8207 (8206-8207) new tokens of the files are read once); first turn (nothing cached) first token: 2.23 s (2.16-2.28).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 285.5 | 2.16 | 82.4 |
| 1 | 2260 | 618 | 1642 | 947.2 | 1.762 | 157.0 |
| 2 | 3966 | 2323 | 1643 | 957.5 | 1.745 | 98.9 |
| 3 | 5624 | 3982 | 1642 | 956.4 | 1.746 | 113.0 |
| 4 | 7282 | 5640 | 1642 | 955.3 | 1.75 | 167.0 |
| 5 | 8932 | 7290 | 1642 | 955.6 | 1.752 | 132.5 |

### Concurrency: 2 clients

Total 96.6 tok/s (90.2-101.9); first token p50 0.984 s, p95 1.804 s; request latency p50 2.056 s, p95 2.791 s; errors 0. Server parallel slots: 1 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 126 s.
