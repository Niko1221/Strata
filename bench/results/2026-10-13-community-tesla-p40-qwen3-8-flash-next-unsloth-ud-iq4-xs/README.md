# Community benchmark on Tesla P40

Measured on 2026-10-13 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Tesla P40, 22 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz (16 cores, 16 threads); installed RAM: 97.7 GiB; storage: not measured
- OS: Windows 2022Server (10.0.20348); GPU driver: 582.78; CUDA: 13.0
- Strata: project 0.1.43, commit not a git checkout; engine 0.1.43
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack unsloth-ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --resident-budget-gib 55 --vision --vram-reserve-mib 700 --pcie-frac 1.00 --spec-min-p 0.50 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97
```

Environment of the config: `STRATA_WATCHDOG_S=600`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 11.9 (10.1-12.6) | 2.17 (1.93-2.57) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 274.2 (86.5-275.4) | not measured | 14.75 (14.67-46.57) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 355.8 (224.3-357.8) | not measured | 45.22 (44.97-71.70) |
| agent session, follow-up turns | 14485 (14481-14522) at the last turn | see below | up to 64 | 3 | see below | 15.8 (10.5-22.4) | 12.83 (8.67-71.49) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.7-68.8); earlier-turn tokens read again per session: 73 (3-74) (earlier turns that the cache did not hold; the 13633 (13633-13635) new tokens of the files are read once); first turn (nothing cached) first token: 6.82 s (6.34-8.41).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 73.4 | 8.408 | 7.8 |
| 1 | 3712 | 632 | 3080 | 43.2 | 71.491 | 10.5 |
| 2 | 5245 | 3775 | 1470 | 38.3 | 38.467 | 11.1 |
| 3 | 8689 | 5309 | 3380 | 68.2 | 49.649 | 14.0 |
| 4 | 11596 | 8736 | 2860 | 53.6 | 53.441 | 18.5 |
| 5 | 14481 | 11635 | 2846 | 65.9 | 43.312 | 13.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 838 s.
