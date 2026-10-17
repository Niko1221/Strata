# Community benchmark on AMD Radeon RX 7900 XTX

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only in this folder; the separate correctness check for this model is in [`../humaneval/`](../humaneval/README.md).

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX, 24 GiB VRAM
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor (16 cores, 32 threads); installed RAM: 61.4 GiB; storage: not measured
- OS: Windows 11 (10.0.26300); GPU driver: 32.0.32015.2008; CUDA: not read
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: no game or browser running during this run; Balanced power plan, no power limits set

## Model and configuration

- Model and quantization: qwen3.8-flash-next-coder-iq1_m
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata.exe --pack coder-iq1_m --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-coder.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768
```

Environment of the config: `STRATA_HIPBLASLT_TUNING=gfx1100-hipblaslt-100500.txt`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 65.6 (61.9-82.3) | 0.41 (0.39-0.45) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1568.3 (1564.5-1568.9) | not measured | 2.59 (2.58-2.61) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1631.0 (1630.4-1633.3) | not measured | 9.89 (9.86-9.91) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1567.0 (1530.9-1587.5) | not measured | 20.49 (20.23-20.97) |
| agent session, follow-up turns | 14512 (14481-14596) at the last turn | see below | up to 64 | 3 | see below | 83.4 (62.9-92.0) | 2.18 (1.46-2.45) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.6); earlier-turn tokens read again per session: 5 (4-74) (earlier turns that the cache did not hold; the 13729 (13729-13730) new tokens of the files are read once); first turn (nothing cached) first token: 1.49 s (1.49-1.50).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 421.0 | 1.493 | 65.9 |
| 1 | 3755 | 632 | 3123 | 1449.3 | 2.182 | 70.7 |
| 2 | 5288 | 3818 | 1470 | 1022.0 | 1.464 | 62.9 |
| 3 | 8689 | 5309 | 3380 | 1419.3 | 2.422 | 68.5 |
| 4 | 11569 | 8707 | 2862 | 1344.2 | 2.177 | 78.4 |
| 5 | 14481 | 11582 | 2899 | 1347.7 | 2.199 | 77.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 168 s.
