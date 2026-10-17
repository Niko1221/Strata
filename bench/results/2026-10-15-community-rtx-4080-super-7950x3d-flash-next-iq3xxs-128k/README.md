# Community benchmark on NVIDIA GeForce RTX 4080 SUPER

Measured on 2026-10-15 by [c-OOL-io](https://github.com/c-OOL-io) with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP; context recall via `tools/needle_bench.py`.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4080 SUPER, 16 GiB VRAM
- CPU: AMD Ryzen 9 7950X3D 16-Core Processor (16 cores, 32 threads); installed RAM: 63.7 GiB; storage: not measured
- OS: Windows 11 (10.0.26200); GPU driver: 617.42; CUDA: not read
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --pcie-frac 0.35 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 94.8 (86.3-121.5) | 0.32 (0.29-0.37) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 2396.0 (2357.6-2408.6) | not measured | 1.72 (1.71-1.74) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3296.9 (3291.2-3300.7) | not measured | 4.91 (4.89-4.93) |
| agent session, follow-up turns | 14569 (14545-14596) at the last turn | see below | up to 64 | 3 | see below | 104.4 (59.2-127.3) | 1.67 (1.58-1.73) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.5-68.6); earlier-turn tokens read again per session: 73 (73-145) (earlier turns that the cache did not hold; the 13729 (13728-13730) new tokens of the files are read once); first turn (nothing cached) first token: 1.37 s (1.35-1.40).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 458.5 | 1.353 | 71.3 |
| 1 | 3744 | 624 | 3120 | 1910.6 | 1.669 | 91.3 |
| 2 | 5277 | 3737 | 1540 | 994.6 | 1.585 | 104.4 |
| 3 | 8721 | 5341 | 3380 | 2026.1 | 1.706 | 98.0 |
| 4 | 11626 | 8764 | 2862 | 1750.5 | 1.681 | 127.3 |
| 5 | 14545 | 11646 | 2899 | 1769.8 | 1.674 | 59.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Recall checks

Needle-in-a-haystack via `tools/needle_bench.py`, thinking off, greedy: six checks (two context lengths × depths 10/50/90) with a unique code word hidden in a haystack of the engine's own source and docs, each answered through the normal API. **6/6 found.**

| Context | Depth | Prompt tokens | Time (s) | Code word (exact) |
| --- | ---: | ---: | ---: | --- |
| 32K | 10% | 32,778 | 10.3 | meadow-copper-504 |
| 32K | 50% | 32,778 | 9.5 | falcon-quartz-940 |
| 32K | 90% | 32,779 | 9.5 | willow-cobalt-696 |
| 128K | 10% | 125,012 | 37.4 | falcon-saffron-138 |
| 128K | 50% | 125,010 | 38.3 | quartz-tundra-528 |
| 128K | 90% | 125,010 | 32.6 | quartz-glacier-192 |

Raw runs: `needles.json`.

## Correctness and limitations

Speed-focused: context recall is verified in the recall section; no general correctness suite (e.g. HumanEval) was run. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 74 s.