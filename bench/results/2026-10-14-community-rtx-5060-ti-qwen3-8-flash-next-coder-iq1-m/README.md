# Community benchmark on NVIDIA GeForce RTX 5060 Ti

Measured on 2026-10-14 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti, 16 GiB VRAM
- CPU: AMD Ryzen 7 2700X Eight-Core Processor (8 cores, 16 threads); installed RAM: 64.0 GiB; storage: not measured
- OS: Windows 10 (10.0.19045); GPU driver: 610.62; CUDA: not read
- Strata: project 0.1.44.1, commit not a git checkout; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-coder-iq1_m
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack coder-iq1_m --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-coder.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 65536 --kv int8 --kv-resident 32768 --spec-min-p 0.5 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 44.9 (42.8-53.9) | 0.66 (0.59-0.76) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1235.3 (1228.5-1249.3) | not measured | 3.28 (3.25-3.31) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1666.6 (1663.4-1667.2) | not measured | 9.68 (9.67-9.70) |
| agent session, follow-up turns | 14512 (14510-14547) at the last turn | see below | up to 64 | 3 | see below | 57.8 (40.3-70.8) | 3.52 (3.17-3.66) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.5-68.7); earlier-turn tokens read again per session: 5 (4-74) (earlier turns that the cache did not hold; the 13729 (13728-13729) new tokens of the files are read once); first turn (nothing cached) first token: 1.97 s (1.95-1.99).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 321.3 | 1.948 | 34.6 |
| 1 | 3755 | 632 | 3123 | 895.5 | 3.522 | 44.1 |
| 2 | 5288 | 3819 | 1469 | 466.3 | 3.187 | 50.9 |
| 3 | 8695 | 5315 | 3380 | 938.7 | 3.662 | 40.3 |
| 4 | 11584 | 8722 | 2862 | 818.3 | 3.545 | 53.8 |
| 5 | 14510 | 11611 | 2899 | 827.5 | 3.552 | 65.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 143 s.
