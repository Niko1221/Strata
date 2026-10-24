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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 62.7 (52.3-70.6) | 0.46 (0.37-0.48) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1645.6 (1314.3-1683.5) | not measured | 2.46 (2.40-3.08) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2743.3 (2491.1-2747.0) | not measured | 5.88 (5.87-6.47) |
| agent session, follow-up turns | 14708 (14704-14716) at the last turn | see below | up to 64 | 3 | see below | 77.4 (58.8-94.0) | 2.40 (2.30-3.65) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.3-69.3); earlier-turn tokens read again per session: 73 (72-74) (earlier turns that the cache did not hold; the 13828 (13828-13828) new tokens of the files are read once); first turn (nothing cached) first token: 1.32 s (1.27-1.55).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 397.9 | 1.552 | 45.2 |
| 1 | 4090 | 626 | 3464 | 954.4 | 3.653 | 67.7 |
| 2 | 5623 | 4154 | 1469 | 522.6 | 2.837 | 67.0 |
| 3 | 9056 | 5686 | 3370 | 970.9 | 3.5 | 70.6 |
| 4 | 11828 | 9049 | 2779 | 865.5 | 3.234 | 94.0 |
| 5 | 14708 | 11889 | 2819 | 930.0 | 3.057 | 83.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 108 s.
