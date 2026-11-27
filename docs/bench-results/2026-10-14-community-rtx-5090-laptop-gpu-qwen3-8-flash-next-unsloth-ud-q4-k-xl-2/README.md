# Community benchmark on NVIDIA GeForce RTX 5090 Laptop GPU

Measured on 2026-10-14 by **jie3000** with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090 Laptop GPU, 24 GiB VRAM
- CPU: Intel(R) Core(TM) Ultra 9 275HX (24 cores, 24 threads); installed RAM: 255.7 GiB; storage: not measured
- PCIe link: Gen 5 x16 (nvidia-smi `pcie.link.gen.current` = 5, `pcie.link.width.current` = 16)
- OS: Windows 11 (10.0.26300); GPU driver: 617.42; CUDA: not read
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads: not recorded; GPU power limit 150 W (default) / 175 W (max), laptop Dynamic Boost

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-q4_k_xl
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack unsloth-ud-q4_k_xl --native Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 8 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --kv-resident 49152 --resident-experts --vision --vram-reserve-mib 500 --pool-workers 20
```

Environment of the config: `STRATA_PREFILL_CPU_SHARE=auto`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 38.1 (22.1-45.4) | 0.79 (0.67-0.83) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1520.2 (1041.8-1534.3) | not measured | 2.68 (2.66-3.89) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2070.8 (1829.6-2087.8) | not measured | 7.80 (7.76-8.82) |
| agent session, follow-up turns | 14558 (14541-14577) at the last turn | see below | up to 64 | 3 | see below | 34.0 (25.1-44.0) | 2.97 (2.14-5.46) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.4 % (68.4-68.4); earlier-turn tokens read again per session: 144 (143-144) (earlier turns that the cache did not hold; the 13728 (13727-13729) new tokens of the files are read once); first turn (nothing cached) first token: 2.02 s (2.00-2.38).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 259.7 | 2.382 | 24.5 |
| 1 | 3741 | 621 | 3120 | 576.3 | 5.465 | 26.0 |
| 2 | 5274 | 3734 | 1540 | 470.2 | 3.338 | 31.1 |
| 3 | 8718 | 5267 | 3451 | 680.0 | 5.117 | 38.8 |
| 4 | 11598 | 8736 | 2862 | 642.0 | 4.503 | 34.0 |
| 5 | 14541 | 11642 | 2899 | 747.6 | 3.92 | 25.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 158 s.
