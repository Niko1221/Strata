# Community benchmark on AMD Radeon

Measured on 2026-10-18 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 12 GiB VRAM
- CPU: AMD Ryzen 5 3600 6-Core Processor (6 cores, 12 threads); installed RAM: 31.3 GiB; storage: not measured
- OS: Linux 7.0.14-24-pve; GPU driver: not read; CUDA: not read
- Strata: project 0.1.45, commit not a git checkout; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen36-ud-q4_k_m-mtp
- Context: 81920 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata --native Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --max-context 81920 --prefill auto --mtp 2
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 24.8 (24.5-25.2) | 0.88 (0.85-0.99) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 813.3 (812.5-815.2) | not measured | 4.95 (4.94-4.95) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 678.2 (677.7-678.3) | not measured | 23.69 (23.69-23.71) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 543.5 (543.2-543.5) | not measured | 59.00 (59.00-59.03) |
| agent session, follow-up turns | 14518 (14513-14527) at the last turn | see below | up to 64 | 3 | see below | 26.8 (24.9-28.7) | 11.72 (4.64-21.71) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 0.0 % (0.0-0.0); earlier-turn tokens read again per session: 30062 (30044-30065) (earlier turns that the cache did not hold; the 13697 (13697-13697) new tokens of the files are read once); first turn (nothing cached) first token: 0.86 s (0.86-0.86).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 710.0 | 0.861 | 26.1 |
| 1 | 3738 | 0 | 3738 | 805.6 | 4.647 | 24.9 |
| 2 | 5253 | 0 | 5253 | 779.8 | 6.742 | 28.0 |
| 3 | 8672 | 0 | 8672 | 740.4 | 11.723 | 27.0 |
| 4 | 11578 | 0 | 11578 | 711.8 | 16.276 | 26.6 |
| 5 | 14518 | 0 | 14518 | 669.1 | 21.709 | 25.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 539 s.
