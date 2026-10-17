# Community benchmark on NVIDIA GeForce RTX 4090

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4090, 24 GiB VRAM
- CPU: 13th Gen Intel(R) Core(TM) i7-13700K (16 cores, 24 threads); installed RAM: 127.8 GiB; storage: not measured
- OS: Windows 10 (10.0.26200); GPU driver: 610.88; CUDA: not read
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-q4_k_xl
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack unsloth-ud-q4_k_xl --native Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --pcie-frac 0.20 --spec-min-p 0.70 --pool-workers 7 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

Environment of the config: `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=0`

## Method

Run by `python tools/strata_bench.py --clients 4 --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 62.8 (54.4-78.2) | 0.51 (0.45-0.62) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1310.9 (1301.1-1313.5) | not measured | 3.10 (3.09-3.12) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2510.6 (2509.7-2511.1) | not measured | 6.42 (6.42-6.45) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2498.1 (2493.5-2501.4) | not measured | 12.89 (12.88-12.91) |
| agent session, follow-up turns | 14567 (14557-14592) at the last turn | see below | up to 64 | 3 | see below | 80.7 (54.8-109.7) | 3.17 (3.04-3.20) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.7-68.8); earlier-turn tokens read again per session: 3 (3-3) (earlier turns that the cache did not hold; the 13728 (13728-13729) new tokens of the files are read once); first turn (nothing cached) first token: 1.63 s (1.62-1.63).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 383.1 | 1.629 | 42.1 |
| 1 | 3741 | 621 | 3120 | 993.4 | 3.174 | 56.1 |
| 2 | 5274 | 3804 | 1470 | 486.7 | 3.046 | 67.3 |
| 3 | 8718 | 5337 | 3381 | 1067.4 | 3.194 | 81.9 |
| 4 | 11644 | 8782 | 2862 | 911.1 | 3.182 | 88.0 |
| 5 | 14592 | 11693 | 2899 | 921.6 | 3.196 | 57.8 |

### Concurrency: 4 clients

Total 47.8 tok/s (46.1-48.7); first token p50 5.033 s, p95 9.095 s; request latency p50 6.922 s, p95 10.877 s; errors 0. Server parallel slots: 1 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 188 s.
