# Community benchmark on NVIDIA GeForce RTX 4090

Measured on 2026-10-21 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4090, 24 GiB VRAM
- CPU: AMD Ryzen 9 7950X3D 16-Core Processor (16 cores, 32 threads); installed RAM: 93.5 GiB; storage: not measured
- OS: Linux 7.2.11-arch1-1; GPU driver: 615.78.08; CUDA: not read
- Strata: project 0.1.46.3, commit 04443aeb; engine 0.1.46.3
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-signal-ap-q4_k_xl
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack signal-ap-q4_k_xl --native Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf --embd-gguf signal-ap-q4_k_xl-embd-bf16.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --resident-budget-gib 67 --pcie-frac 0.35 --spec-min-p 0.70 --pool-workers 8
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 60.6 (54.5-74.3) | 0.42 (0.34-0.45) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1692.1 (1333.0-1715.1) | not measured | 2.39 (2.36-3.03) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2770.7 (2523.5-2786.7) | not measured | 5.82 (5.78-6.38) |
| agent session, follow-up turns | 14711 (14701-14714) at the last turn | see below | up to 64 | 3 | see below | 72.5 (56.1-93.3) | 2.37 (2.22-3.65) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.2 % (69.2-69.5); earlier-turn tokens read again per session: 144 (3-145) (earlier turns that the cache did not hold; the 13828 (13828-13829) new tokens of the files are read once); first turn (nothing cached) first token: 1.31 s (1.22-1.49).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 415.1 | 1.494 | 64.2 |
| 1 | 4102 | 637 | 3465 | 960.6 | 3.646 | 66.5 |
| 2 | 5635 | 4165 | 1470 | 527.6 | 2.811 | 68.3 |
| 3 | 9068 | 5628 | 3440 | 983.5 | 3.521 | 56.1 |
| 4 | 11818 | 9110 | 2708 | 829.0 | 3.288 | 71.5 |
| 5 | 14701 | 11811 | 2890 | 949.2 | 3.066 | 60.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 107 s.
