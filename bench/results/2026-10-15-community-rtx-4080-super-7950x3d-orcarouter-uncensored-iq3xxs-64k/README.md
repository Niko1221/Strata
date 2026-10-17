# Community benchmark on NVIDIA GeForce RTX 4080 SUPER

Measured on 2026-10-15 by [c-OOL-io](https://github.com/c-OOL-io) with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP; context recall via `tools/needle_bench.py`.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4080 SUPER, 16 GiB VRAM
- CPU: AMD Ryzen 9 7950X3D 16-Core Processor (16 cores, 32 threads); installed RAM: 63.7 GiB; storage: not measured
- OS: Windows 11 (10.0.26200); GPU driver: 617.42; CUDA: not read
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: orcarouter-qwen3.8-flash-next-uncensored-iq3_xxs
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
--pack orca-iq3_xxs --native Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 65536 --kv int8 --kv-resident 32768 --vram-reserve-mib 1000 --pcie-frac 0.20 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97 --adapt-async 1
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 95.4 (88.7-133.9) | 0.33 (0.29-0.36) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 2017.0 (2008.3-2020.7) | not measured | 2.02 (2.02-2.03) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3517.4 (3040.8-3522.4) | not measured | 4.61 (4.60-5.32) |
| agent session, follow-up turns | 14607 (14587-14611) at the last turn | see below | up to 64 | 3 | see below | 109.7 (82.0-131.5) | 1.99 (1.89-2.04) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.5-68.8); earlier-turn tokens read again per session: 2 (1-143) (earlier turns that the cache did not hold; the 13730 (13729-13730) new tokens of the files are read once); first turn (nothing cached) first token: 1.57 s (1.55-1.57).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 403.1 | 1.548 | 102.7 |
| 1 | 3748 | 625 | 3123 | 1571.8 | 2.027 | 104.4 |
| 2 | 5281 | 3812 | 1469 | 779.0 | 1.918 | 82.0 |
| 3 | 8724 | 5346 | 3378 | 1689.8 | 2.043 | 103.6 |
| 4 | 11650 | 8788 | 2862 | 1454.6 | 2.005 | 128.1 |
| 5 | 14587 | 11688 | 2899 | 1480.2 | 2.004 | 116.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Recall checks

Needle-in-a-haystack via `tools/needle_bench.py`, thinking off, greedy: six checks (two context lengths × depths 10/50/90) with a unique code word hidden in a haystack of the engine's own source and docs, each answered through the normal API. **6/6 found.**

| Context | Depth | Prompt tokens | Time (s) | Code word (exact) |
| --- | ---: | ---: | ---: | --- |
| 32K | 10% | 32,778 | 12.2 | meadow-copper-504 |
| 32K | 50% | 32,778 | 10.4 | falcon-quartz-940 |
| 32K | 90% | 32,779 | 10.4 | willow-cobalt-696 |
| 64K | 10% | 64,799 | 20.2 | falcon-saffron-138 |
| 64K | 50% | 64,797 | 20.2 | quartz-tundra-528 |
| 64K | 90% | 64,797 | 15.3 | quartz-glacier-192 |

Raw runs: `needles.json`.

## Correctness and limitations

Speed-focused: context recall is verified in the recall section; no general correctness suite (e.g. HumanEval) was run. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 81 s.