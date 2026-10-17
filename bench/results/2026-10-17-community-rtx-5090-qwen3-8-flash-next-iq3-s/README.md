# Community benchmark on NVIDIA GeForce RTX 5090

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090, 32 GiB VRAM
- CPU: AMD Ryzen 7 7700X 8-Core Processor (8 cores, 16 threads); installed RAM: 62.0 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 595.99.02; source build with CUDA 13.2.2 (nvcc 13.2.86), targeting sm_120; matching CUDA runtime and cuBLAS libraries.
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- PCIe link: Gen5 x8 verified under load. GPU power limit: 500 W; background workloads were not formally recorded.

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 262144 tokens; parallel slots: 1
- Settings: official calibration was run before this measurement; the benchmark then used the saved configuration unchanged. CPU expert workers: 7 (default). Thinking off; temperature 0. No experimental speed projection was enabled.

```text
strata-cuda13.2.2 --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-grow --pcie-frac 0.20 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

Environment of the config: `STRATA_IQ_MT_MIN=1`, `STRATA_PREFILL_MMQ=1`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 154.9 (135.9-189.4) | 0.19 (0.17-0.23) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 3387.0 (2777.9-3391.8) | not measured | 1.20 (1.20-1.46) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 6351.8 (5786.5-6366.7) | not measured | 2.54 (2.54-2.79) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 6482.7 (6445.2-6487.4) | not measured | 4.97 (4.96-5.00) |
| agent session, follow-up turns | 14547 (14533-14574) at the last turn | see below | up to 64 | 3 | see below | 178.5 (141.3-223.6) | 1.17 (1.11-1.78) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.8); earlier-turn tokens read again per session: 74 (4-74) (earlier turns that the cache did not hold; the 13729 (13728-13729) new tokens of the files are read once); first turn (nothing cached) first token: 0.59 s (0.58-0.66).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 942.4 | 0.658 | 127.2 |
| 1 | 3754 | 634 | 3120 | 1762.2 | 1.785 | 143.3 |
| 2 | 5287 | 3817 | 1470 | 1067.3 | 1.391 | 158.7 |
| 3 | 8731 | 5350 | 3381 | 2071.7 | 1.647 | 157.7 |
| 4 | 11640 | 8778 | 2862 | 1813.9 | 1.594 | 190.2 |
| 5 | 14574 | 11675 | 2899 | 2032.1 | 1.445 | 154.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 63 s.

## Build provenance and scope

Engine SHA-256: `7084fecf1ac1bd17ffb2c74b97967784a4e35c03951ad5c9d824e811ff0ddba1`. Full source commit: `296b59d2f73c8f04a805df29c2fa97655adace78`. This was not a measurement of current main. The context capacity was 262,144 tokens, but the largest measured input was 32,049 tokens. No full-256K, concurrent-serving, or answer-quality claim is made.

Calibration selected PCIe share 0.20, draft threshold 0.70, and adaptive tier 1/160/0.97; it retained the default worker count. This report records the selected configuration, not proof that it is globally optimal.

Privacy: machine-specific library paths and the bind address were removed from the generated JSON; numerical measurements are unchanged. No API credentials or private conversation data are included.
