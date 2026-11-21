# Community benchmark on AMD Radeon RX 9070 XT

Measured on 2026-11-21 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 9070 XT, 16 GiB VRAM
- CPU: AMD Ryzen 5 7600X3D 6-Core Processor (6 cores, 12 threads); installed RAM: 31.4 GiB; storage: Patriot P300 1TB SSD/NVMe; SKHynix_HFS512GDE9X084N SSD/NVMe
- OS: Windows 11 (10.0.26300); GPU driver: 32.0.32029.2010; CUDA (driver supports up to): not read
- Strata: project ?, commit not a git checkout; engine 0.1.63
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-gyro-gyro-s-Q4_0
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
component:strata-engine --pack gyro-gyro-s --native Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf --ple-gguf Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf --expert-profile expert-profile.bin --expert-cache auto --max-context 131072 --kv int8 --resident-experts --kv-resident 32768 --vision --draft dflash2 --draft-model qwen3.8-flash-next-dflash2.q4_0-selq4.pack --draft-fallback none --mtp-draft-vocab draft_vocab_latin.bin --resident-budget-gib 17 --spec 6 --draft-conf 0.6 --prefill 8192 --pcie-frac 1.0 --conversation-cache-min-free-mib 0 --conversation-cache-mib 0 --vram-reserve-mib 1024
```

Environment of the config: `STRATA_RESIDENT_PIN=1`, `STRATA_HIP_WMMA=1`, `STRATA_PARTIAL_PIN=1`, `STRATA_PARTIAL_PIN_GIB=8`, `STRATA_UNBUFFERED_LOAD=0`, `STRATA_PREFILL_COPY_EVENT_EVERY=16`, `STRATA_PREFILL_COPY_WAIT_DEDUPE=1`, `STRATA_ROUTE_TAIL_SKIP=7`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 86.9 (66.9-124.2) | 0.33 (0.29-0.36) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1744.6 (1741.6-1760.2) | not measured | 2.33 (2.32-2.34) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2405.4 (2391.3-2422.4) | not measured | 7.01 (6.76-7.16) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2451.7 (2450.7-2453.5) | not measured | 13.86 (13.41-14.04) |
| agent session, follow-up turns | 8941 (8914-8971) at the last turn | see below | up to 64 | 3 | see below | 57.1 (39.4-87.6) | 1.92 (1.45-2.12) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.8 % (70.7-70.9); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 8206 (8206-8207) new tokens of the files are read once); first turn (nothing cached) first token: 2.24 s (2.21-2.26).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 322.8 | 2.264 | 72.7 |
| 1 | 2273 | 631 | 1642 | 1117.8 | 1.499 | 68.9 |
| 2 | 3979 | 2336 | 1643 | 987.1 | 1.693 | 70.8 |
| 3 | 5654 | 4012 | 1642 | 867.2 | 1.917 | 40.9 |
| 4 | 7298 | 5656 | 1642 | 838.9 | 1.989 | 42.8 |
| 5 | 8941 | 7299 | 1642 | 811.9 | 2.043 | 39.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 129 s.
