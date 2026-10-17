# Community benchmark on NVIDIA GeForce RTX 4090

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4090, 24 GiB VRAM
- CPU: 13th Gen Intel(R) Core(TM) i7-13700K (16 cores, 24 threads); installed RAM: 127.8 GiB; storage: not measured
- OS: Windows 10 (10.0.26200); GPU driver: 610.88; CUDA: not read
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --pcie-frac 0.20 --spec-min-p 0.70 --pool-workers 7 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

Environment of the config: `STRATA_ROUTE_TAIL_SKIP=0`

## Method

Run by `python tools/strata_bench.py --clients 4 --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 92.2 (85.9-106.2) | 0.32 (0.30-0.37) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 2158.7 (2131.1-2161.8) | not measured | 1.88 (1.88-1.92) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 4161.7 (4159.0-4171.2) | not measured | 3.88 (3.87-3.91) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 4228.6 (4227.3-4234.6) | not measured | 7.62 (7.61-7.63) |
| agent session, follow-up turns | 14555 (14530-14591) at the last turn | see below | up to 64 | 3 | see below | 119.8 (84.6-146.1) | 1.86 (1.78-1.90) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.6-68.8); earlier-turn tokens read again per session: 5 (3-74) (earlier turns that the cache did not hold; the 13730 (13728-13730) new tokens of the files are read once); first turn (nothing cached) first token: 0.98 s (0.97-0.98).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 646.8 | 0.977 | 89.0 |
| 1 | 3759 | 637 | 3122 | 1687.2 | 1.872 | 86.0 |
| 2 | 5292 | 3752 | 1540 | 864.5 | 1.801 | 107.9 |
| 3 | 8736 | 5355 | 3381 | 1820.5 | 1.902 | 94.3 |
| 4 | 11616 | 8754 | 2862 | 1556.8 | 1.87 | 146.1 |
| 5 | 14530 | 11631 | 2899 | 1575.4 | 1.864 | 84.6 |

### Concurrency: 4 clients

Total 74.8 tok/s (69.8-77.5); first token p50 3.285 s, p95 5.931 s; request latency p50 4.473 s, p95 7.064 s; errors 0. Server parallel slots: 1 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 116 s.
