# Community benchmark on NVIDIA GeForce RTX 4060 Ti

Measured on 2026-10-13 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4060 Ti, 16 GiB VRAM
- CPU: Intel(R) Core(TM) i7-14700K (20 cores, 28 threads); installed RAM: 31.7 GiB; storage: Samsung SSD 990 EVO Plus 2TB SSD/NVMe
- OS: Windows 11 (10.0.26200); GPU driver: 616.92; CUDA (driver supports up to): 13.4; engine built for CUDA 13.0
- Strata: project 0.1.44, commit ed4773eb; engine 0.1.43
- GPU detail: power_limit_w=165.00, power_draw_w=75.40, pcie=gen4 x8, pstate=P2, temp_c=55
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-coder-iq1_m
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack coder-iq1_m --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-coder.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 65536 --kv int8 --resident-experts --spec-min-p 0.5 --pool-workers 10 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 33.0 (16.0-40.0) | 0.89 (0.61-1.94) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1104.9 (980.0-1146.3) | not measured | 3.69 (3.56-4.13) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1343.9 (1334.9-1363.1) | not measured | 12.01 (11.84-12.10) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1265.0 (1043.4-1369.5) | not measured | 25.41 (23.50-30.76) |
| agent session, follow-up turns | 14571 (14570-14580) at the last turn | see below | up to 64 | 3 | see below | 44.1 (21.0-59.4) | 2.95 (1.89-3.44) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.8); earlier-turn tokens read again per session: 74 (4-75) (earlier turns that the cache did not hold; the 13730 (13730-13730) new tokens of the files are read once); first turn (nothing cached) first token: 2.08 s (1.99-2.18).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 286.8 | 2.177 | 21.4 |
| 1 | 3755 | 632 | 3123 | 926.5 | 3.438 | 21.0 |
| 2 | 5288 | 3748 | 1540 | 630.6 | 2.482 | 32.8 |
| 3 | 8732 | 5351 | 3381 | 1007.4 | 3.399 | 33.2 |
| 4 | 11640 | 8778 | 2862 | 912.2 | 3.205 | 40.1 |
| 5 | 14571 | 11672 | 2899 | 998.8 | 2.98 | 47.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 275 s.
