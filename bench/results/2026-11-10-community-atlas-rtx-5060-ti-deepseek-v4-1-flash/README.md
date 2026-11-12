# Community benchmark on RTX 5060 Ti 16 GB in a Ryzen 9 7900X (Atlas)

Measured on 2026-11-10 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti, 16 GiB VRAM
- CPU: AMD Ryzen 9 7900X 12-Core Processor (12 cores, 24 threads); installed RAM: 61.8 GiB; storage: zram0; nvme0n1 Corsair MP700 PRO XT nvme; nvme1n1 WD_BLACK SN770M 2TB  nvme
- OS: Linux 7.1.5-100.fc43.x86_64; GPU driver: 580.173.02; CUDA (driver supports up to): 13.0
- Strata: project 0.1.59.1, commit not a git checkout; engine ds4-0.1
- GPU detail (after the run, near idle): power_limit_w=180.00, power_draw_w=65.74, pcie=gen4 x8, pstate=P1, temp_c=54
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: deepseek-v4.1-flash
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 32 (29-38) | 0 | 100 (100-100) | 9 | not measured | 14.8 (13.6-16.9) | 1.31 (1.17-1.55) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 3 | 370.2 (349.7-374.6) | not measured | 11.15 (11.02-11.80) |
| 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 3 | 438.7 (431.9-458.5) | not measured | 37.77 (36.14-38.37) |
| 32K prompt, fresh | 33126 (33126-33126) | 0 (0-0) | 24 cap | 3 | 727.5 (553.3-728.0) | not measured | 45.56 (45.54-59.91) |
| agent session, follow-up turns | 13510 (13493-13510) at the last turn | see below | up to 64 | 3 | see below | 16.0 (14.7-16.8) | 10.30 (8.18-12.45) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.3 % (70.3-70.3); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 12630 (12629-12631) new tokens of the files are read once); first turn (nothing cached) first token: 6.01 s (6.00-6.19).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 602 | 0 | 602 | 97.4 | 6.186 | 13.1 |
| 1 | 3998 | 636 | 3362 | 270.3 | 12.454 | 14.7 |
| 2 | 5516 | 4061 | 1455 | 168.1 | 8.672 | 15.0 |
| 3 | 8615 | 5565 | 3050 | 259.7 | 11.762 | 15.7 |
| 4 | 10851 | 8670 | 2181 | 223.4 | 9.781 | 15.7 |
| 5 | 13493 | 10906 | 2587 | 242.6 | 10.684 | 15.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 638 s.

## Atlas system-specific notes

- **GPU:** NVIDIA RTX 5060 Ti 16 GB, PCIe 4.0 x8 (the card supports 5.0).
- **CPU/RAM:** AMD Ryzen 9 7900X, 12C/24T; 61.8 GiB usable DDR5.
- **Storage:** 1 TB Corsair MP700 PRO XT, PCIe Gen5 x4, mounted /mnt/mp700; the root is a 1.8 TB WD_BLACK SN770M, 99% full (24 GiB free). The model and pack are on the MP700; the Engram and the CPU-lane experts stream from it.
- **Software:** Strata app 0.2.13.1; official DeepSeek engine 0.1.59 (prebuilt CUDA 13, sm_120; the 7900X has AVX-512, so the prebuilt AVX2 package runs); driver 580.173.02; CUDA 13.
- **Configuration:** the install start script used its defaults: **131,072 context, vision on, automatic RAM store, DSpark auto**; not the 40 GiB explicit store, 64K context, or vision-off settings of the Boris report.
- **Memory state at load:** GPU 14,999 / 16,311 MiB; RAM 44 GiB used, 17 GiB available. This is a 16 GB-VRAM / 64 GB-RAM / SSD-backed configuration, not the Boris RAM-resident one.
- **Run:** 3 measured runs, thinking off, temperature 0, one request at a time, community-format tools/strata_bench.py --long (not standard-v1); **0 failed requests**.
- The Atlas embedding and vision services (llama-embed-atlas.service, qwen3vl-8b-vision.service) were stopped for the run and restored afterward.
