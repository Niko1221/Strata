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
- Settings: the configuration the desktop app wrote for this PC, unchanged (hip backend); thinking off for the suite; temperature 0

```text
component:strata-engine --pack gyro-gyro-s --native Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf --ple-gguf Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --resident-experts --kv-resident 32768 --vision --pool-workers 5 --vram-reserve-mib 1536
```

Environment of the config: `STRATA_RESIDENT_PIN=0`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 50.2 (39.4-55.7) | 0.60 (0.48-0.83) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 829.0 (769.1-832.1) | not measured | 4.89 (4.87-5.26) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1346.6 (1317.2-1354.0) | not measured | 11.96 (11.90-12.24) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1322.8 (1320.4-1329.8) | not measured | 24.28 (24.15-24.32) |
| agent session, follow-up turns | 8936 (8930-8983) at the last turn | see below | up to 64 | 3 | see below | 61.2 (19.9-76.1) | 2.23 (1.80-2.54) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.8 % (70.8-70.8); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 8206 (8206-8206) new tokens of the files are read once); first turn (nothing cached) first token: 2.44 s (2.31-2.53).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 268.2 | 2.314 | 37.7 |
| 1 | 2270 | 628 | 1642 | 918.6 | 1.806 | 65.8 |
| 2 | 3976 | 2333 | 1643 | 777.0 | 2.143 | 53.6 |
| 3 | 5649 | 4007 | 1642 | 746.9 | 2.218 | 28.4 |
| 4 | 7293 | 5651 | 1642 | 701.6 | 2.37 | 41.8 |
| 5 | 8936 | 7294 | 1642 | 655.9 | 2.544 | 19.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 208 s.
