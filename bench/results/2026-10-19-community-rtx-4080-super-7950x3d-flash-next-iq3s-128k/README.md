# Community benchmark on NVIDIA GeForce RTX 4080 SUPER

Measured on 2026-10-19 by [c-OOL-io](https://github.com/c-OOL-io) with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP; context recall via `tools/needle_bench.py`.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4080 SUPER, 16 GiB VRAM
- CPU: AMD Ryzen 9 7950X3D 16-Core Processor (16 cores, 32 threads); installed RAM: 63.7 GiB; storage: not measured
- OS: Windows 11 (10.0.26200); GPU driver: 617.42; CUDA: not read
- Strata: project 0.1.45, commit bd5c88b0; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (calibration-tuned: `--pcie-frac 0.20 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97`); thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf ple-fp8.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 131072 --kv int8 --vision --vram-reserve-mib 700 --pcie-frac 0.20 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 84.2 (79.6-115.6) | 0.35 (0.31-0.40) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 2070.0 (2038.2-2074.2) | not measured | 1.98 (1.97-2.01) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3377.1 (1999.3-3461.5) | not measured | 4.80 (4.68-8.08) |
| agent session, follow-up turns | 14556 (14542-14574) at the last turn | see below | up to 64 | 3 | see below | 95.8 (69.1-116.0) | 1.95 (1.88-2.35) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.7); earlier-turn tokens read again per session: 74 (4-74) (earlier turns that the cache did not hold; the 13717 (13717-13717) new tokens of the files are read once); first turn (nothing cached) first token: 1.02 s (0.99-1.08).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 587.9 | 1.082 | 51.5 |
| 1 | 3743 | 623 | 3120 | 1349.1 | 2.353 | 88.7 |
| 2 | 5276 | 3806 | 1470 | 782.1 | 1.919 | 89.0 |
| 3 | 8709 | 5339 | 3370 | 1684.2 | 2.046 | 95.8 |
| 4 | 11627 | 8765 | 2862 | 1442.4 | 2.017 | 116.0 |
| 5 | 14542 | 11643 | 2899 | 1487.4 | 1.992 | 69.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Recall checks

Needle-in-a-haystack via `tools/needle_bench.py`, thinking off, greedy: six checks (two context lengths × depths 10/50/90) with a unique code word hidden in a haystack of the engine's own source and docs, each answered through the normal API. **6/6 found.**

| Context | Depth | Prompt tokens | Time (s) | Code word (exact) |
| --- | ---: | ---: | ---: | --- |
| 32K | 10% | 32,792 | 14.6 | meadow-copper-504 |
| 32K | 50% | 32,792 | 11.2 | falcon-quartz-940 |
| 32K | 90% | 32,793 | 10.1 | willow-cobalt-696 |
| 128K | 10% | 127,535 | 49.6 | falcon-saffron-138 |
| 128K | 50% | 127,533 | 64.5 | quartz-tundra-528 |
| 128K | 90% | 127,533 | 52.6 | quartz-glacier-192 |

Raw runs: `needles.json`.

## Correctness and limitations

Speed-focused: context recall is verified in the recall section; no general correctness suite (e.g. HumanEval) was run. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 82 s.
