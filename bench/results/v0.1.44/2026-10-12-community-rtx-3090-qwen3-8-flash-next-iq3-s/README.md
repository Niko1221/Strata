# Community benchmark on NVIDIA GeForce RTX 3090

Measured on 2026-10-12 with `tools/strata_bench.py` (suite 1, 5 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3090, 24 GiB VRAM
- CPU: AMD Ryzen 9 3900X 12-Core Processor (12 cores, 24 threads); installed RAM: 62.7 GiB; storage: not measured
- OS: Linux 6.8.0-146-generic; GPU driver: 595.99.02; <host>: 13.2
- Strata: project 0.1.43.1, commit a0116b27; engine 0.1.43.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 131072 --kv int8 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

## Method

Run by `python tools/strata_bench.py --long --runs 5` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 5 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 5 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 5 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 15 | not measured | 89.1 (83.0-99.7) | 0.36 (0.34-0.42) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 5 | 2014.8 (1766.4-2024.8) | not measured | 2.01 (2.00-2.29) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 5 | 2378.1 (2304.0-2387.8) | not measured | 6.78 (6.75-7.00) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 5 | 2350.0 (2349.3-2353.6) | not measured | 13.68 (13.66-13.68) |
| agent session, follow-up turns | 14453 (14446-14460) at the last turn | see below | up to 64 | 5 | see below | 103.5 (74.8-144.4) | 1.85 (1.61-2.49) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.6-68.8); earlier-turn tokens read again per session: 4 (3-73) (earlier turns that the cache did not hold; the 13633 (13633-13634) new tokens of the files are read once); first turn (nothing cached) first token: 1.16 s (1.12-1.26).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 489.5 | 1.263 | 65.4 |
| 1 | 3703 | 623 | 3080 | 1251.2 | 2.484 | 100.1 |
| 2 | 5236 | 3766 | 1470 | 779.8 | 1.908 | 89.3 |
| 3 | 8680 | 5299 | 3381 | 1373.6 | 2.488 | 95.2 |
| 4 | 11580 | 8720 | 2860 | 1259.4 | 2.299 | 103.5 |
| 5 | 14453 | 11607 | 2846 | 1363.5 | 2.115 | 74.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 208 s.
