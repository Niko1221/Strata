# Community benchmark on NVIDIA GeForce RTX 5090

Measured on 2026-10-25 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090, 32 GiB VRAM
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 62.2 GiB; storage: not measured
- OS: Linux 7.3.0-5-cachyos; GPU driver: 615.78.08; CUDA: not read
- Strata: project 0.1.50, commit 41cf05a0; engine 0.1.50
- Background workloads and power limits: background load not recorded. The card runs a custom LACT profile: a **400 W cap** (the card's minimum; the stock default is 575 W), locked 300-2500 MHz core, +600 MHz VF offset at p-state 0. It was in force during this run (lactd active since this machine's boot at 09:59:07, `/etc/lact/config.yaml` unchanged since 2026-10-19) and re-checked afterwards with `nvidia-smi --query-gpu=power.limit` -> 400.00 W. These are a 400 W card's numbers, not a stock 5090's.

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --vision --vram-reserve-mib 832 --kv-grow --pcie-frac 0.35 --spec-min-p 0.55 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97
```

Environment of the config: `STRATA_DEQUANT_WIDE=1`, `STRATA_GDN_CHUNKED=1`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 183.9 (165.7-227.0) | 0.15 (0.13-0.17) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 5987.8 (5483.7-5993.1) | not measured | 0.68 (0.68-0.74) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 7007.9 (6969.6-7022.7) | not measured | 2.31 (2.30-2.32) |
| agent session, follow-up turns | 12784 (12782-12785) at the last turn | see below | up to 64 | 3 | see below | 223.0 (185.3-249.2) | 0.65 (0.59-0.74) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.9 % (68.9-69.1); earlier-turn tokens read again per session: 143 (72-144) (earlier turns that the cache did not hold; the 11896 (11896-11897) new tokens of the files are read once); first turn (nothing cached) first token: 0.45 s (0.45-0.46).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 1398.4 | 0.448 | 151.6 |
| 1 | 3211 | 633 | 2578 | 3605.6 | 0.73 | 185.3 |
| 2 | 4744 | 3275 | 1469 | 2314.5 | 0.648 | 211.1 |
| 3 | 7988 | 4808 | 3180 | 4405.7 | 0.736 | 214.0 |
| 4 | 10016 | 7981 | 2035 | 2826.0 | 0.733 | 223.0 |
| 5 | 12785 | 10079 | 2706 | 4035.2 | 0.685 | 205.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded, and the power limit in the section above was read by hand after the run, not by the suite. Total time of the suite: 33 s.
