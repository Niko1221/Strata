# Community benchmark on NVIDIA GeForce RTX 5060 Ti

Measured on 2026-10-14 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti, 16 GiB VRAM
- CPU: AMD Ryzen 7 2700X Eight-Core Processor (8 cores, 16 threads); installed RAM: 64.0 GiB; storage: not measured
- OS: Windows 10 (10.0.19045); GPU driver: 610.62; CUDA: not read
- Strata: project 0.1.44.1, commit not a git checkout; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: swift-1.5-iq3_xxs
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack swift-iq3_xxs --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --spec-min-p 0.5 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 46.5 (40.5-57.4) | 0.63 (0.57-0.77) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 682.1 (678.5-686.3) | not measured | 5.92 (5.91-5.95) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1063.5 (1061.5-1065.1) | not measured | 15.14 (15.12-15.19) |
| agent session, follow-up turns | 14567 (14560-14588) at the last turn | see below | up to 64 | 3 | see below | 56.6 (37.7-68.0) | 5.66 (5.28-5.84) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.6); earlier-turn tokens read again per session: 73 (73-74) (earlier turns that the cache did not hold; the 13730 (13729-13731) new tokens of the files are read once); first turn (nothing cached) first token: 4.24 s (4.18-4.25).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 146.5 | 4.183 | 33.6 |
| 1 | 3743 | 620 | 3123 | 549.4 | 5.716 | 43.2 |
| 2 | 5276 | 3806 | 1470 | 278.8 | 5.304 | 45.3 |
| 3 | 8720 | 5269 | 3451 | 594.6 | 5.841 | 53.1 |
| 4 | 11646 | 8784 | 2862 | 508.3 | 5.673 | 62.5 |
| 5 | 14567 | 11668 | 2899 | 514.8 | 5.673 | 37.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 213 s.
