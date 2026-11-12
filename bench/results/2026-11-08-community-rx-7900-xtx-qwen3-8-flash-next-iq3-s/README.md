# Community benchmark on AMD Radeon RX 7900 XTX

Measured on 2026-11-08 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX, 24 GiB VRAM
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 91.7 GiB; storage: sda     CT2000MX500SSD1       sata; nvme1n1 IRP-SSDPR-P44A-4K0-80 nvme; nvme0n1 WDS200T1X0E-00AFY0    nvme
- OS: Linux 7.2.14-arch1-1; GPU driver: not read; CUDA (driver supports up to): not read
- Strata: project 0.1.58, commit 78362ebc; engine 0.1.58
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700
```

Environment of the config: `STRATA_HIPBLASLT_TUNING=gfx1100-hipblaslt-100202.txt`, `STRATA_SCORES_THREAD=1`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 69.6 (59.1-77.1) | 0.46 (0.38-0.50) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1684.5 (1682.9-1697.1) | not measured | 2.41 (2.39-2.41) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1902.7 (1898.4-1903.2) | not measured | 8.47 (8.47-8.48) |
| agent session, follow-up turns | 13581 (13527-13584) at the last turn | see below | up to 64 | 3 | see below | 84.8 (65.2-89.7) | 2.10 (1.72-2.25) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.9 % (69.8-69.9); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 12725 (12725-12725) new tokens of the files are read once); first turn (nothing cached) first token: 1.61 s (1.55-1.62).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 397.3 | 1.555 | 55.9 |
| 1 | 3835 | 630 | 3205 | 1480.5 | 2.19 | 75.7 |
| 2 | 5368 | 3898 | 1470 | 856.3 | 1.74 | 76.7 |
| 3 | 8612 | 5431 | 3181 | 1429.0 | 2.251 | 85.9 |
| 4 | 10802 | 8638 | 2164 | 1154.1 | 1.899 | 85.2 |
| 5 | 13527 | 10817 | 2710 | 1309.6 | 2.093 | 73.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 101 s.
