# Community benchmark on NVIDIA GeForce RTX 5080

Measured on 2026-11-11 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5080, 16 GiB VRAM
- CPU: AMD Ryzen 7 7800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 127.4 GiB; storage: Samsung SSD 980 PRO 2TB SSD/NVMe; Msft Virtual Disk SSD/File Backed Virtual; Samsung SSD 870 QVO 2TB SSD/SATA; Intenso Premium Line Unspecified/USB
- OS: Windows 11 (10.0.26300); GPU driver: 621.01; CUDA (driver supports up to): 13.5; engine built for CUDA 13.0
- Strata: project 0.1.59, commit d2f09647; engine ds4-0.1
- GPU detail (after the run, near idle): power_limit_w=360.00, power_draw_w=87.43, pcie=gen5 x16, pstate=P1, temp_c=45.82
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: deepseek-v4.1-flash, ISTA-DASLab IST-DAS 1.75 bpw (GSQ-RCO **IQ1_M**, four shards,
  39.36 / 39.18 / 40.00 / 2.19 GB; HF repo `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF`, revision
  `cb02422c900df489cea5833e8a92171224059b3c` pinned in `setup.py`). The 111 GB n-gram table and the 8 GB DSpark
  drafter sit in the shared folder `deepseek-v4.1-flash-shared/` and are reused by every ISTA-DAS set.
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking on for the suite; temperature 0
- Experts per word: **5 (Faster)**, the automatic choice on a 16 GiB card (the engine writes "experts per word: 5
  (faster mode; full quality: --experts 6)"); the config passes no `--experts` flag.
- DSpark (drafter): **not loaded**. `--dspark auto` decides against it because the routed experts (115 GB) are far
  more than the card holds (15 GB), so its verify rounds read 3.3x a token's bytes and gain nothing; `--dspark-conf
  0.65` is therefore without effect here.
- Expert placement: 0 of the 15,360 routed experts sit in VRAM. The engine keeps a **96.8 GB RAM store** (11,120
  slots of 8.7 MB, policy 0, 8 reader threads) and reads the rest in place from the model files.
- Sibling run on the same machine and protocol: the ISTA-DAS **Coder** IQ1_M
  ([`.../2026-11-11-community-rtx-5080-deepseek-coder-exp5`](../2026-11-11-community-rtx-5080-deepseek-coder-exp5/)).

```text
strata-dsv41.exe --pack deepseek-v4.1-flash-ist-das-1.75bpw --max-context 131072 --eos 1 --vision on --dspark auto --dspark-conf 0.65 --reserve-mib 900
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
| short chat | 58 (55-64) | 0 | 100 (100-100) | 9 | not measured | 9.4 (9.0-10.0) | 3.79 (3.52-4.36) |
| 4K prompt, fresh | 4151 (4151-4151) | 0 (0-0) | 24 cap | 3 | 160.8 (160.8-161.8) | not measured | 25.84 (25.68-25.85) |
| 16K prompt, fresh | 16588 (16588-16588) | 0 (0-0) | 24 cap | 3 | 550.6 (538.2-554.9) | not measured | 30.16 (29.95-30.87) |
| 32K prompt, fresh | 33152 (33152-33152) | 0 (0-0) | 24 cap | 3 | 875.1 (846.5-940.2) | not measured | 37.94 (35.32-39.23) |
| agent session, follow-up turns | 13274 (13270-13284) at the last turn | see below | up to 64 | 3 | see below | 9.6 (9.2-10.9) | 18.62 (16.52-21.74) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.8 % (69.7-69.9); earlier-turn tokens read again per session: 132 (132-195) (earlier turns that the cache did not hold; the 12363 (12350-12363) new tokens of the files are read once); first turn (nothing cached) first token: 13.07 s (13.05-13.12).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 627 | 0 | 627 | 43.8 | 13.124 | 9.4 |
| 1 | 3801 | 626 | 3175 | 169.2 | 18.802 | 9.2 |
| 2 | 5303 | 3801 | 1502 | 91.8 | 16.525 | 9.5 |
| 3 | 8400 | 5350 | 3050 | 158.9 | 21.739 | 9.2 |
| 4 | 10631 | 8400 | 2231 | 126.5 | 17.769 | 9.4 |
| 5 | 13274 | 10687 | 2587 | 140.2 | 18.635 | 9.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 877 s.

Two things a reader should keep in mind with these numbers:

- **The prompt throughput is not monotonic.** 4K reads at 160.8 tok/s, 16K at 550.6 and 32K at 875.1. The 4K case is
  the first test after the warm-up, and with 115 GB of experts behind a 96.8 GB RAM store a short prompt touches many
  experts that are not yet resident, so it pays the model-file reads that the longer prompts amortise. The 16K and 32K
  figures are the ones that describe steady state; the 4K figure describes a start.
- **Decode is about as fast as the Coder's** (9.4 against 9.8 tok/s chat) although this set is twice as large, because
  decode reads only 5 of 15,360 experts per token and both sets keep them at the same 1-bit storage. What differs is
  the amount RAM cannot hold: 96.8 GB of 115 GB here against all 63 GB of the Coder.

Machine paths, user and host names were replaced before committing; the per-request data is otherwise unchanged.

