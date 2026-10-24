# Community benchmark on AMD Radeon RX 9060 XT

Measured on 2026-10-24 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 9060 XT, 16 GiB VRAM
- CPU: AMD Ryzen 5 9600X 6-Core Processor (6 cores, 12 threads); installed RAM: 31.4 GiB; storage: not measured
- OS: Windows 11 (10.0.26300); GPU driver: 32.0.32015.7029; CUDA: not read
- Strata: project 0.1.47, commit 011c9e60; engine 0.1.47
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-gyro-gyro-m
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata.exe --pack gyro-gyro-m --native Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf --ple-gguf Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 65536 --kv int8 --mmap-experts --vision
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 33.9 (24.9-42.3) | 0.98 (0.68-1.28) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 201.9 (182.6-216.0) | not measured | 19.96 (18.67-22.06) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 204.0 (202.1-224.3) | not measured | 78.79 (71.67-79.53) |
| agent session, follow-up turns | 14578 (14578-14587) at the last turn | see below | up to 64 | 3 | see below | 25.2 (11.1-29.8) | 19.13 (13.75-22.83) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.5 % (69.3-69.6); earlier-turn tokens read again per session: 74 (3-144) (earlier turns that the cache did not hold; the 13731 (13729-13731) new tokens of the files are read once); first turn (nothing cached) first token: 9.77 s (7.66-10.02).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 61.4 | 10.021 | 25.1 |
| 1 | 4093 | 624 | 3469 | 254.9 | 13.748 | 23.6 |
| 2 | 5626 | 4086 | 1540 | 76.8 | 20.147 | 24.9 |
| 3 | 9059 | 5619 | 3440 | 151.4 | 22.829 | 25.2 |
| 4 | 11831 | 9122 | 2709 | 122.9 | 22.145 | 29.8 |
| 5 | 14587 | 11870 | 2717 | 160.0 | 17.086 | 22.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 694 s.
