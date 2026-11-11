# Community benchmark on NVIDIA GeForce RTX 5080

Measured on 2026-11-11 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5080, 16 GiB VRAM
- CPU: AMD Ryzen 7 7800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 127.4 GiB; storage: Samsung SSD 980 PRO 2TB SSD/NVMe; Msft Virtual Disk SSD/File Backed Virtual; Samsung SSD 870 QVO 2TB SSD/SATA; Intenso Premium Line Unspecified/USB
- OS: Windows 11 (10.0.26300); GPU driver: 621.01; CUDA (driver supports up to): 13.5; engine built for CUDA 13.0
- Strata: project 0.1.59, commit d2f09647; engine ds4-0.1
- GPU detail (after the run, near idle): power_limit_w=360.00, power_draw_w=89.23, pcie=gen5 x16, pstate=P1, temp_c=52.79
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: deepseek-v4.1-flash
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking on for the suite; temperature 0
- Experts per word: **6 (full quality)**, set with `--experts 6` in the run config (the engine confirms "experts per
  word: 6 (full quality)"). Without it this 16 GiB card picks Faster (5 of 6 experts); the sibling run
  [`.../deepseek-coder-exp5`](../2026-11-11-community-rtx-5080-deepseek-coder-exp5/) measures the same set that way,
  so the two differ only in that one option.

```text
strata-dsv41.exe --pack deepseek-v4.1-flash-ista-das-coder --max-context 131072 --eos 1 --vision on --experts 6 --dspark auto --dspark-conf 0.65 --reserve-mib 900
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
| short chat | 58 (55-64) | 0 | 100 (100-100) | 9 | not measured | 8.4 (7.9-8.8) | 3.06 (2.79-3.60) |
| 4K prompt, fresh | 4151 (4151-4151) | 0 (0-0) | 24 cap | 3 | 685.7 (567.2-699.0) | not measured | 6.08 (5.97-7.34) |
| 16K prompt, fresh | 16588 (16588-16588) | 0 (0-0) | 24 cap | 3 | 1347.4 (1224.6-1368.1) | not measured | 12.35 (12.16-13.60) |
| 32K prompt, fresh | 33152 (33152-33152) | 0 (0-0) | 24 cap | 3 | 1524.4 (1522.5-1538.8) | not measured | 21.81 (21.60-21.84) |
| agent session, follow-up turns | 13297 (13228-13303) at the last turn | see below | up to 64 | 3 | see below | 8.4 (7.4-10.0) | 5.02 (4.44-6.55) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.0 % (69.8-70.0); earlier-turn tokens read again per session: 69 (69-133) (earlier turns that the cache did not hold; the 12367 (12319-12368) new tokens of the files are read once); first turn (nothing cached) first token: 4.19 s (4.09-4.39).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 627 | 0 | 627 | 118.5 | 4.39 | 8.7 |
| 1 | 3789 | 626 | 3163 | 495.4 | 6.531 | 8.4 |
| 2 | 5307 | 3852 | 1455 | 293.8 | 5.096 | 8.5 |
| 3 | 8410 | 5360 | 3050 | 487.6 | 6.549 | 7.4 |
| 4 | 10646 | 8465 | 2181 | 405.0 | 5.553 | 8.2 |
| 5 | 13297 | 10709 | 2588 | 458.8 | 5.8 | 8.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 490 s.
