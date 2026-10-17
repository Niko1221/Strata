# Community benchmark on AMD Radeon RX 7900 XTX

Measured on 2026-10-16 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

*Reported by Neil Huang.*

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX, 24 GiB VRAM
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 63.7 GiB; storage: not measured
- OS: Windows 11 (10.0.26100); GPU driver: 32.0.31041.1004; CUDA: not read
- Storage: Samsung SSD 990 PRO 2 TB NVMe (model and engine on the same drive)
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: a desktop Windows session with a browser and small
  helper processes ran alongside; GPU at stock power limit; PCIe link speed/width not recorded
  (the expert cache filled to 9.0 GiB of VRAM during this run, ~14 GiB when the desktop is idle)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq2_xs
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata.exe --pack iq2_xs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto:16384 --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --vram-reserve-mib 2048
```

Environment of the config: `STRATA_HIPBLASLT_TUNING=gfx1100-hipblaslt-100500.txt`, `STRATA_PF_FUSED=1`, `STRATA_PF_GEMM=1`, `STRATA_PF_SWITCH_MIN_T=4096`, `STRATA_HIP_WMMA=1`, `STRATA_PA_FAST=1`, `STRATA_ROUTE_TAIL_SKIP=7`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 75.7 (61.1-87.2) | 0.39 (0.33-0.45) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 928.9 (927.3-940.2) | not measured | 4.36 (4.31-4.37) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1982.9 (1981.4-1987.0) | not measured | 8.14 (8.12-8.14) |
| agent session, follow-up turns | 14559 (14557-14573) at the last turn | see below | up to 64 | 3 | see below | 82.3 (51.4-114.6) | 4.02 (3.63-4.25) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.4-68.8); earlier-turn tokens read again per session: 74 (3-144) (earlier turns that the cache did not hold; the 13728 (13728-13729) new tokens of the files are read once); first turn (nothing cached) first token: 3.09 s (3.04-3.10).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 202.6 | 3.045 | 55.3 |
| 1 | 3741 | 621 | 3120 | 766.7 | 4.111 | 64.2 |
| 2 | 5274 | 3804 | 1470 | 406.6 | 3.652 | 72.0 |
| 3 | 8718 | 5267 | 3451 | 820.4 | 4.255 | 74.1 |
| 4 | 11632 | 8770 | 2862 | 714.4 | 4.034 | 89.7 |
| 5 | 14557 | 11658 | 2899 | 725.9 | 4.042 | 51.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load, storage and power notes were added by the submitter (see "Hardware and software"). Total time of the suite: 140 s.
