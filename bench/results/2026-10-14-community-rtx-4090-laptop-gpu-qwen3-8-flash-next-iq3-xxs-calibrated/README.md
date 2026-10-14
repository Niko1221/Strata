# Community benchmark on NVIDIA GeForce RTX 4090 Laptop GPU

Measured on 2026-10-14 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4090 Laptop GPU, 16 GiB VRAM
- CPU: 13th Gen Intel(R) Core(TM) i7-13700HX (16 cores, 16 threads); installed RAM: 61.2 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 595.99.02; CUDA: 13.2
- Strata: project 0.1.43.1, commit not a git checkout; engine 0.1.43.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Context: 65536 tokens; parallel slots: 1
- Settings: after `./setup.sh --calibrate`, which kept `--pcie-frac 0.35 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97` (defaults before: `--pcie-frac` 0.55, `--spec-min-p` 0.5); thinking off for the suite; temperature 0
- Storage: NVMe SSD (ext4); power limits and background load not recorded

```text
strata --pack iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 65536 --kv int8 --pcie-frac 0.35 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 58.5 (53.6-67.3) | 0.53 (0.49-0.60) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 900.4 (890.4-904.7) | not measured | 4.49 (4.47-4.54) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1713.7 (1711.7-1724.9) | not measured | 9.41 (9.34-9.41) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1745.8 (1735.2-1746.5) | not measured | 18.41 (18.39-18.52) |
| agent session, follow-up turns | 14477 (14476-14518) at the last turn | see below | up to 64 | 3 | see below | 70.8 (37.8-87.9) | 4.39 (4.21-4.54) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.8-68.8); earlier-turn tokens read again per session: 4 (3-4) (earlier turns that the cache did not hold; the 13634 (13634-13635) new tokens of the files are read once); first turn (nothing cached) first token: 1.92 s (1.92-1.97).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 320.3 | 1.924 | 41.5 |
| 1 | 3704 | 624 | 3080 | 684.8 | 4.535 | 59.7 |
| 2 | 5237 | 3767 | 1470 | 345.1 | 4.292 | 64.8 |
| 3 | 8681 | 5300 | 3381 | 755.5 | 4.51 | 69.2 |
| 4 | 11605 | 8745 | 2860 | 645.2 | 4.465 | 87.5 |
| 5 | 14477 | 11631 | 2846 | 647.3 | 4.429 | 37.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 209 s.

## Before and after calibration

Same machine, same suite (`--long`), engine 0.1.43.1. Before = default settings (2026-10-13), after = calibrated (this run).

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| short chat decode tok/s | 46.0 | 58.5 | +27.2% |
| short chat first token (s) | 0.72 | 0.53 | -25.8% |
| agent decode tok/s | 63.3 | 70.8 | +11.8% |
| 4K prompt tok/s | 893.3 | 900.4 | +0.8% |
| 16K prompt tok/s | 1695.6 | 1713.7 | +1.1% |
| 32K prompt tok/s | 1730.6 | 1745.8 | +0.9% |

Each side is a single suite run, so the prompt-speed differences of about 1% are within run-to-run variation.
