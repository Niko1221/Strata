# Community benchmark on NVIDIA GeForce RTX 5080

Measured on 2026-11-11 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5080, 16 GiB VRAM
- CPU: AMD Ryzen 7 7800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 127.4 GiB; storage: Samsung SSD 980 PRO 2TB SSD/NVMe; Msft Virtual Disk SSD/File Backed Virtual; Samsung SSD 870 QVO 2TB SSD/SATA; Intenso Premium Line Unspecified/USB
- OS: Windows 11 (10.0.26300); GPU driver: 621.01; CUDA (driver supports up to): 13.5; engine built for CUDA 13.0
- Strata: project 0.1.59, commit d2f09647; engine ds4-0.1
- GPU detail (after the run, near idle): power_limit_w=360.00, power_draw_w=87.56, pcie=gen5 x16, pstate=P1, temp_c=50.85
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: deepseek-v4.1-flash
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking on for the suite; temperature 0
- Experts per word: **5 (Faster)**, the automatic choice on a 16 GiB card (the engine writes "experts per word: 5
  (faster mode; full quality: --experts 6)"; the config file passes no `--experts` flag). The sibling run
  [`.../deepseek-coder-exp6`](../2026-11-11-community-rtx-5080-deepseek-coder-exp6/) measures the same set with
  `--experts 6` (full quality), so the two differ only in that one option.

```text
strata-dsv41.exe --pack deepseek-v4.1-flash-ista-das-coder --max-context 131072 --eos 1 --vision on --dspark auto --dspark-conf 0.65 --reserve-mib 900
```

## Method

Run by `python tools/strata_bench.py --long --think` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 58 (55-64) | 0 | 100 (100-100) | 9 | not measured | 9.8 (8.9-11.2) | 2.65 (2.35-3.02) |
| 4K prompt, fresh | 4151 (4151-4151) | 0 (0-0) | 24 cap | 3 | 736.6 (597.3-737.9) | not measured | 5.66 (5.65-6.97) |
| 16K prompt, fresh | 16588 (16588-16588) | 0 (0-0) | 24 cap | 3 | 1396.6 (1278.0-1396.7) | not measured | 11.92 (11.91-13.02) |
| 32K prompt, fresh | 33152 (33152-33152) | 0 (0-0) | 24 cap | 3 | 1621.8 (1619.1-1627.5) | not measured | 20.49 (20.42-20.53) |
| agent session, follow-up turns | 13287 (13265-13297) at the last turn | see below | up to 64 | 3 | see below | 10.4 (9.2-11.4) | 4.75 (4.32-6.40) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.8 % (69.8-69.9); earlier-turn tokens read again per session: 132 (132-132) (earlier turns that the cache did not hold; the 12348 (12347-12350) new tokens of the files are read once); first turn (nothing cached) first token: 3.84 s (3.82-4.13).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 627 | 0 | 627 | 128.1 | 4.127 | 9.6 |
| 1 | 3789 | 626 | 3163 | 503.4 | 6.317 | 9.2 |
| 2 | 5287 | 3789 | 1498 | 302.7 | 5.084 | 9.7 |
| 3 | 8401 | 5350 | 3051 | 490.6 | 6.397 | 9.5 |
| 4 | 10646 | 8464 | 2182 | 409.4 | 5.484 | 9.9 |
| 5 | 13297 | 10709 | 2588 | 463.1 | 5.755 | 9.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 442 s.
