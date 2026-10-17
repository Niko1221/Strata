# Community benchmark on AMD Radeon RX 7900 XTX

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only in this folder; the separate correctness check for this model is in [`../humaneval/`](../humaneval/README.md).

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX, 24 GiB VRAM
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor (16 cores, 32 threads); installed RAM: 61.4 GiB; storage: not measured
- OS: Windows 11 (10.0.26300); GPU driver: 32.0.32015.2008; CUDA: not read
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: no game or browser running during this run; Balanced power plan, no power limits set

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq2_xs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata.exe --pack iq2_xs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 80.2 (70.4-94.0) | 0.34 (0.30-0.41) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1198.3 (1184.9-1221.4) | not measured | 3.38 (3.32-3.41) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1366.9 (1335.7-1367.4) | not measured | 11.78 (11.78-12.06) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1279.3 (1276.5-1282.8) | not measured | 25.09 (25.02-25.14) |
| agent session, follow-up turns | 14567 (14559-14583) at the last turn | see below | up to 64 | 3 | see below | 90.2 (73.1-109.4) | 3.06 (2.80-3.25) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.8); earlier-turn tokens read again per session: 3 (3-74) (earlier turns that the cache did not hold; the 13729 (13729-13729) new tokens of the files are read once); first turn (nothing cached) first token: 2.41 s (2.40-2.47).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 258.4 | 2.401 | 66.0 |
| 1 | 3741 | 620 | 3121 | 1036.6 | 3.06 | 77.4 |
| 2 | 5274 | 3734 | 1540 | 555.0 | 2.817 | 77.6 |
| 3 | 8718 | 5337 | 3381 | 1059.2 | 3.242 | 77.3 |
| 4 | 11635 | 8773 | 2862 | 950.2 | 3.057 | 92.4 |
| 5 | 14583 | 11684 | 2899 | 944.7 | 3.111 | 91.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 208 s.
