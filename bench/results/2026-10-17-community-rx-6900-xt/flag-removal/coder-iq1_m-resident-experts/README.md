# Community benchmark on AMD Radeon RX 6900 XT

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 6900 XT, 16 GiB VRAM
- CPU: Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz (6 cores, 12 threads); installed RAM: 127.9 GiB; storage: not measured
- OS: Windows 11 (10.0.22631); GPU driver: 32.0.21045.5002; CUDA: not read
- Strata: project 0.1.45, commit 477ac13c (v0.1.45) with the BENCH fix #2919; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-coder-iq1_m
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata.exe --pack coder-iq1_m --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-coder.bin --expert-cache auto --prefill 12288 --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --resident-experts --vram-reserve-mib 1024 --kv-resident 20480 --pool-tasks 36
```

Environment of the config: `STRATA_RESIDENT_PIN=0`, `STRATA_HIP_PROMPT_F16=1`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 29.4 (24.7-34.7) | 1.06 (0.92-1.17) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 786.3 (713.7-786.6) | not measured | 5.15 (5.15-5.68) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 961.6 (925.6-963.5) | not measured | 16.75 (16.72-17.40) |
| agent session, follow-up turns | 14569 (14528-14602) at the last turn | see below | up to 64 | 3 | see below | 38.0 (25.9-41.9) | 4.22 (3.14-5.40) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.5-68.8); earlier-turn tokens read again per session: 74 (2-75) (earlier turns that the cache did not hold; the 13718 (13717-13719) new tokens of the files are read once); first turn (nothing cached) first token: 3.29 s (3.25-3.43).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 179.5 | 3.432 | 19.5 |
| 1 | 3751 | 628 | 3123 | 616.3 | 5.117 | 25.9 |
| 2 | 5284 | 3744 | 1540 | 421.9 | 3.687 | 41.0 |
| 3 | 8668 | 5299 | 3369 | 628.8 | 5.401 | 32.4 |
| 4 | 11575 | 8713 | 2862 | 578.8 | 4.993 | 35.2 |
| 5 | 14528 | 11629 | 2899 | 620.8 | 4.718 | 41.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 222 s.
