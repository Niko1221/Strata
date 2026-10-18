# Community benchmark on AMD Radeon

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 12 GiB VRAM
- CPU: AMD Ryzen 5 3600 6-Core Processor (6 cores, 12 threads); installed RAM: 31.3 GiB; storage: not measured
- OS: Linux 7.0.14-24-pve; GPU driver: not read; CUDA: not read
- Strata: project 0.1.45, commit not a git checkout; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.6-35b-a3b-ud-q4_k_xl
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata --native Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --max-context 65536 --prefill auto --mtp 2
```

## Method

Run by `python tools/strata_bench.py --clients 2 --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 24.6 (24.2-24.9) | 0.86 (0.84-0.98) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 813.4 (811.3-813.9) | not measured | 4.95 (4.94-4.96) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 677.8 (677.2-678.0) | not measured | 23.71 (23.70-23.73) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 543.5 (543.2-543.6) | not measured | 59.00 (58.99-59.03) |
| agent session, follow-up turns | 14529 (14513-14536) at the last turn | see below | up to 64 | 3 | see below | 27.7 (24.6-28.8) | 11.79 (4.65-21.73) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 0.0 % (0.0-0.0); earlier-turn tokens read again per session: 30126 (30047-30138) (earlier turns that the cache did not hold; the 13697 (13697-13697) new tokens of the files are read once); first turn (nothing cached) first token: 0.86 s (0.86-0.86).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 707.9 | 0.864 | 20.8 |
| 1 | 3753 | 0 | 3753 | 805.0 | 4.67 | 25.3 |
| 2 | 5269 | 0 | 5269 | 780.3 | 6.759 | 27.8 |
| 3 | 8687 | 0 | 8687 | 737.0 | 11.798 | 28.0 |
| 4 | 11590 | 0 | 11590 | 710.2 | 16.328 | 27.4 |
| 5 | 14536 | 0 | 14536 | 669.3 | 21.728 | 24.9 |

### Concurrency: 2 clients

Total 22.7 tok/s (21.3-24.0); first token p50 3.667 s, p95 7.054 s; request latency p50 8.441 s, p95 11.846 s; errors 0. Server parallel slots: 1 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 574 s.
