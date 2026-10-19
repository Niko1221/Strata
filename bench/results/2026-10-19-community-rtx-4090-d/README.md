# Community benchmark on NVIDIA GeForce RTX 4090 D

Measured on 2026-10-19 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4090 D, 24 GiB VRAM
- CPU: AMD Ryzen 9 9900X 12-Core Processor (12 cores, 24 threads); installed RAM: 91.9 GiB; storage: not measured
- OS: Linux 7.2.10-200.fc44.x86_64; GPU driver: 615.71.09; CUDA: not read
- Strata: project 0.1.45, commit 477ac13c; engine 0.1.45
- Background workloads: none observed running during the suite (GPU idle immediately before/after each pass; verified by sampling `nvidia-smi` every 1s through the run)
- Power limit and clocks: power limit 375 W (card default 425 W) via a local `nvidia-smi -pl`/NVML tune applied at boot, with a GPC clock offset of +200 MHz and a memory clock offset of +2050 MHz (sampled peak during this run: 2925 MHz core, 11276 MHz memory)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-q4_k_xl
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-q4_k_xl --native Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 960 --pcie-frac 0.00 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

Environment of the config: `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=0`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 70.5 (65.3-83.1) | 0.42 (0.37-0.48) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1488.7 (1484.7-1496.7) | not measured | 2.71 (2.70-2.73) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2816.8 (2812.7-2819.4) | not measured | 5.72 (5.71-5.73) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2780.0 (2770.5-2780.9) | not measured | 11.55 (11.55-11.60) |
| agent session, follow-up turns | 14551 (14540-14592) at the last turn | see below | up to 64 | 3 | see below | 89.1 (68.8-108.8) | 2.78 (2.67-2.81) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.8); earlier-turn tokens read again per session: 3 (3-72) (earlier turns that the cache did not hold; the 13716 (13716-13717) new tokens of the files are read once); first turn (nothing cached) first token: 1.09 s (1.08-1.09).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 567.8 | 1.087 | 46.6 |
| 1 | 3741 | 621 | 3120 | 1129.8 | 2.776 | 68.8 |
| 2 | 5274 | 3804 | 1470 | 552.7 | 2.675 | 74.8 |
| 3 | 8707 | 5337 | 3370 | 1205.5 | 2.812 | 78.0 |
| 4 | 11587 | 8725 | 2862 | 1034.7 | 2.782 | 94.5 |
| 5 | 14540 | 11643 | 2897 | 1045.7 | 2.786 | 85.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 137 s.
