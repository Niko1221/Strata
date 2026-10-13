# Community benchmark on AMD Radeon

Measured on 2026-10-13 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 16 GiB VRAM
- CPU: Intel(R) Core(TM) i5-9400F CPU @ 2.90GHz (6 cores, 6 threads); installed RAM: 15.5 GiB; storage: not measured
- OS: Linux 7.2.0-1-cachyos; GPU driver: 7.2.0-1-cachyos; CUDA: not read
- Strata: project 0.1.43.1, commit a0116b27; engine 0.1.43.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-coder-iq1_m
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata --pack coder-iq1_m --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-agent.bin --expert-cache auto --prefill auto --spec 5 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --mmap-experts --adapt-every 100000 --pcie-frac 0.6 --kv-grow
```

Environment of the config: `STRATA_HIPBLASLT_TUNING=gfx1200-hipblaslt-100202.txt`, `STRATA_IO_PREFETCH=1`, `STRATA_VERIFY_TIMEOUT_S=300`, `STRATA_WATCHDOG_S=180`, `STRATA_PROFILE_ROUTING_ONLY=1`, `STRATA_ROUTE_TAIL_SKIP=7`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 8.9 (7.8-12.7) | 4.92 (4.00-6.02) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 178.4 (178.1-191.9) | not measured | 22.63 (21.17-22.69) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 322.7 (313.0-327.2) | not measured | 49.91 (49.25-51.49) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 373.2 (364.6-374.9) | not measured | 86.12 (85.73-88.08) |
| agent session, follow-up turns | 14439 (14426-14513) at the last turn | see below | up to 64 | 3 | see below | 5.7 (3.5-8.0) | 18.84 (15.62-21.78) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.7-68.9); earlier-turn tokens read again per session: 4 (3-5) (earlier turns that the cache did not hold; the 13634 (13632-13636) new tokens of the files are read once); first turn (nothing cached) first token: 15.35 s (14.68-18.63).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 32.9 | 18.627 | 4.1 |
| 1 | 3711 | 631 | 3080 | 142.2 | 21.783 | 4.9 |
| 2 | 5244 | 3774 | 1470 | 89.6 | 16.532 | 5.1 |
| 3 | 8663 | 5283 | 3380 | 166.7 | 20.413 | 5.1 |
| 4 | 11555 | 8695 | 2860 | 153.1 | 18.829 | 6.3 |
| 5 | 14426 | 11580 | 2846 | 151.1 | 18.968 | 3.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 1192 s.
