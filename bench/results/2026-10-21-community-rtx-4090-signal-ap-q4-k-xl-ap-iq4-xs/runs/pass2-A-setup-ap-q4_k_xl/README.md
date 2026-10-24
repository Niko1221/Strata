# Community benchmark on NVIDIA GeForce RTX 4090

Measured on 2026-10-21 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4090, 24 GiB VRAM
- CPU: AMD Ryzen 9 7950X3D 16-Core Processor (16 cores, 32 threads); installed RAM: 93.5 GiB; storage: not measured
- OS: Linux 7.2.11-arch1-1; GPU driver: 615.78.08; CUDA: not read
- Strata: project 0.1.46.3, commit 04443aeb; engine 0.1.46.3
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: signal-3.8-flash-next-ap-q4_k_xl
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack signal-ap-q4_k_xl --native Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf --ple-gguf Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf --embd-gguf token-embd-bf16.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --ple-io ram --kv-resident 32768 --vision --vram-reserve-mib 700
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 71.5 (46.4-77.8) | 0.48 (0.36-0.56) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1635.8 (984.2-1680.9) | not measured | 2.48 (2.41-4.11) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3080.0 (2714.9-3128.6) | not measured | 5.24 (5.16-5.95) |
| agent session, follow-up turns | 14691 (14670-14718) at the last turn | see below | up to 64 | 3 | see below | 89.4 (69.6-107.3) | 2.50 (2.36-5.12) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.3-69.4); earlier-turn tokens read again per session: 72 (2-75) (earlier turns that the cache did not hold; the 13827 (13827-13827) new tokens of the files are read once); first turn (nothing cached) first token: 1.47 s (1.42-1.94).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 321.0 | 1.944 | 58.3 |
| 1 | 4106 | 641 | 3465 | 830.7 | 4.213 | 71.2 |
| 2 | 5639 | 4099 | 1540 | 404.2 | 3.837 | 69.6 |
| 3 | 9072 | 5702 | 3370 | 870.3 | 3.899 | 86.3 |
| 4 | 11832 | 9124 | 2708 | 532.0 | 5.116 | 84.5 |
| 5 | 14691 | 11872 | 2819 | 793.5 | 3.58 | 79.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 113 s.
