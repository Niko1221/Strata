# Community benchmark on AMD Radeon RX 9070 XT

Measured on 2026-11-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 9070 XT, 16 GiB VRAM
- CPU: AMD Ryzen 5 7600X3D 6-Core Processor (6 cores, 12 threads); installed RAM: 31.4 GiB; storage: Patriot P300 1TB SSD/NVMe; SKHynix_HFS512GDE9X084N SSD/NVMe
- OS: Windows 11 (10.0.26300); GPU driver: 32.0.32015.16001; CUDA (driver supports up to): not read
- Strata: desktop app 0.2.14 (control 0.1.60.1, commit `3af562d5` per the standard run), official Windows HIP `rx9070` engine 0.1.60
- Background workloads: Claude desktop app and Google Chrome open; no other GPU work

## Model and configuration

- Model and quantization: qwen3.8-flash-next-gyro-gyro-s
- Context: 131072 tokens; parallel slots: 1
- Settings: hand-tuned configuration (see the folder's top README), hip backend; thinking off for the suite; temperature 0

```text
component:strata-engine --pack gyro-gyro-s --native Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf --ple-gguf Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --resident-experts --kv-resident 32768 --vision --resident-budget-gib 17 --vram-reserve-mib 1024
```

Environment of the config: `STRATA_RESIDENT_PIN=1`, `STRATA_HIP_WMMA=1`, `STRATA_PARTIAL_PIN=1`, `STRATA_PARTIAL_PIN_GIB=8`, `STRATA_UNBUFFERED_LOAD=0`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 71.3 (60.8-89.9) | 0.35 (0.29-0.57) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1236.2 (1222.4-1248.3) | not measured | 3.27 (3.25-3.32) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1571.3 (1560.6-1575.4) | not measured | 10.26 (10.24-10.33) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1568.9 (1567.2-1570.0) | not measured | 20.47 (20.45-20.50) |
| agent session, follow-up turns | 8934 (8930-8935) at the last turn | see below | up to 64 | 3 | see below | 60.4 (38.0-100.7) | 1.93 (1.42-2.16) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.8 % (70.8-70.8); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 8206 (8206-8206) new tokens of the files are read once); first turn (nothing cached) first token: 2.03 s (2.01-2.08).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 306.6 | 2.014 | 58.9 |
| 1 | 2277 | 635 | 1642 | 1125.1 | 1.494 | 78.1 |
| 2 | 3983 | 2340 | 1643 | 988.6 | 1.703 | 63.8 |
| 3 | 5645 | 4003 | 1642 | 840.2 | 1.975 | 48.7 |
| 4 | 7291 | 5649 | 1642 | 833.2 | 2.001 | 53.3 |
| 5 | 8935 | 7293 | 1642 | 776.5 | 2.163 | 45.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 166 s.
