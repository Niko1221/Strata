# Community benchmark on NVIDIA GeForce RTX 5090

Measured on 2026-10-23 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090, 32 GiB VRAM
- CPU: AMD Ryzen 9 5950X 16-Core Processor (16 cores, 32 threads); installed RAM: 121.5 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 595.99.02; CUDA: 13.2
- Strata: project 0.1.48.1, commit 7ddc17a3; engine 0.1.48.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-nvfp4-nvfp4-BF16
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack pack --native qwen-nvfp4-gptq-dense.gguf --native-dense-gguf qwen-nvfp4-gptq-dense.gguf --ple-gguf ple-fp8.gguf --embd-gguf token-embd-bf16.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --pcie-frac 0.20 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 64.3 (56.9-70.1) | 0.71 (0.69-0.78) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1727.9 (1699.7-1731.3) | not measured | 2.34 (2.34-2.38) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3399.0 (3392.4-3400.5) | not measured | 4.75 (4.74-4.76) |
| agent session, follow-up turns | 14595 (14566-14608) at the last turn | see below | up to 64 | 3 | see below | 101.9 (68.2-120.2) | 2.20 (2.08-2.62) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.6 % (69.3-69.6); earlier-turn tokens read again per session: 1 (1-143) (earlier turns that the cache did not hold; the 13730 (13729-13731) new tokens of the files are read once); first turn (nothing cached) first token: 2.02 s (1.88-2.08).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 328.0 | 1.877 | 55.6 |
| 1 | 4109 | 641 | 3468 | 1476.1 | 2.371 | 68.2 |
| 2 | 5642 | 4102 | 1540 | 661.7 | 2.346 | 76.8 |
| 3 | 9075 | 5706 | 3369 | 1466.7 | 2.317 | 95.3 |
| 4 | 11847 | 9068 | 2779 | 1067.6 | 2.624 | 114.5 |
| 5 | 14566 | 11849 | 2717 | 1226.6 | 2.232 | 84.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 98 s.
