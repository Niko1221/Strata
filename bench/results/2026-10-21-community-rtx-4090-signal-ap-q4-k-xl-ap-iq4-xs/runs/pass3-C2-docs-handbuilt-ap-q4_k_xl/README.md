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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 65.4 (58.1-73.6) | 0.38 (0.30-0.45) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1724.2 (1369.3-1728.5) | not measured | 2.35 (2.34-2.96) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2765.1 (2530.3-2770.4) | not measured | 5.83 (5.82-6.37) |
| agent session, follow-up turns | 14702 (14689-14724) at the last turn | see below | up to 64 | 3 | see below | 83.2 (59.3-98.7) | 2.35 (2.22-3.73) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.3-69.4); earlier-turn tokens read again per session: 73 (73-75) (earlier turns that the cache did not hold; the 13828 (13827-13829) new tokens of the files are read once); first turn (nothing cached) first token: 1.24 s (1.23-1.54).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 403.5 | 1.54 | 64.7 |
| 1 | 4103 | 638 | 3465 | 966.7 | 3.623 | 59.3 |
| 2 | 5636 | 4166 | 1470 | 520.3 | 2.849 | 77.6 |
| 3 | 9069 | 5699 | 3370 | 975.7 | 3.485 | 69.8 |
| 4 | 11841 | 9062 | 2779 | 750.1 | 3.727 | 70.0 |
| 5 | 14724 | 11904 | 2820 | 936.4 | 3.037 | 74.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 105 s.
