# Community benchmark on AMD Radeon

Measured on 2026-11-10 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 12 GiB VRAM
- CPU: AMD Ryzen 5 3600 6-Core Processor (6 cores, 10 threads); installed RAM: 56.0 GiB; storage: nvme0n1 INTEL SSDPEKNW010T8                      nvme
- OS: Linux 7.0.14-26-pve; GPU driver: not read; CUDA (driver supports up to): not read
- Strata: project 0.1.59, commit d2f09647; engine 0.1.59
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.6-35b-a3b-ud-q4_k_xl
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
component:strata-engine --native Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --max-context 65536 --prefill auto --mtp 2
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 33.3 (31.8-34.0) | 0.51 (0.44-0.59) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 821.2 (818.8-821.7) | not measured | 4.90 (4.90-4.91) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 682.0 (681.7-683.3) | not measured | 23.56 (23.52-23.57) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 546.8 (546.7-546.8) | not measured | 58.64 (58.64-58.65) |
| agent session, follow-up turns | 13554 (13551-13572) at the last turn | see below | up to 64 | 3 | see below | 28.7 (24.4-33.5) | 11.57 (4.78-19.56) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 0.0 % (0.0-0.0); earlier-turn tokens read again per session: 29448 (29419-29484) (earlier turns that the cache did not hold; the 12708 (12708-12708) new tokens of the files are read once); first turn (nothing cached) first token: 0.85 s (0.85-0.86).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 716.3 | 0.855 | 32.7 |
| 1 | 3845 | 0 | 3845 | 804.7 | 4.788 | 32.0 |
| 2 | 5360 | 0 | 5360 | 783.6 | 6.847 | 31.0 |
| 3 | 8599 | 0 | 8599 | 743.5 | 11.575 | 28.5 |
| 4 | 10816 | 0 | 10816 | 720.8 | 15.017 | 26.6 |
| 5 | 13572 | 0 | 13572 | 694.3 | 19.557 | 24.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 515 s.
