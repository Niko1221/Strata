# Community benchmark on NVIDIA GeForce RTX 5070 Ti Laptop GPU

Measured on 2026-10-22 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12 GiB VRAM
- CPU: Intel(R) Core(TM) Ultra 9 285H (16 cores, 16 threads); installed RAM: 30.8 GiB; storage: not measured
- OS: Linux 7.2.8-cachyos; GPU driver: 615.71.09; CUDA: not read
- Strata: project 0.1.47, commit 011c9e60; engine 0.1.47
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-coder-iq1_m
- Context: 204800 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack coder-iq1_m --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-coder.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 204800 --kv int8 --resident-experts --vision --spec-min-p 0.5 --pool-workers 2 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 27.0 (19.4-33.6) | 1.36 (1.11-2.10) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 289.9 (239.8-297.9) | not measured | 13.95 (13.57-16.84) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 399.4 (359.8-401.1) | not measured | 40.31 (40.16-44.72) |
| agent session, follow-up turns | 14585 (14517-14603) at the last turn | see below | up to 64 | 3 | see below | 25.7 (21.0-31.2) | 13.74 (8.92-15.38) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.5 % (69.5-69.6); earlier-turn tokens read again per session: 74 (5-74) (earlier turns that the cache did not hold; the 13731 (13730-13731) new tokens of the files are read once); first turn (nothing cached) first token: 8.55 s (7.76-11.30).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 54.3 | 11.298 | 20.6 |
| 1 | 4120 | 649 | 3471 | 235.9 | 15.027 | 28.8 |
| 2 | 5653 | 4183 | 1470 | 134.1 | 11.082 | 23.7 |
| 3 | 9037 | 5668 | 3369 | 231.3 | 14.687 | 21.0 |
| 4 | 11786 | 9078 | 2708 | 195.5 | 13.981 | 22.4 |
| 5 | 14517 | 11800 | 2717 | 185.9 | 14.731 | 23.0 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 510 s.
