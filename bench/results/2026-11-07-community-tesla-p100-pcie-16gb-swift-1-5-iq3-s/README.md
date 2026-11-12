# Community benchmark on Tesla P100-PCIE-16GB

Measured on 2026-11-07 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Tesla P100-PCIE-16GB, 16 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-1650 v3 @ 3.50GHz (6 cores, 12 threads); installed RAM: 78.4 GiB; storage: not measured
- OS: Linux 7.2.8-2-cachyos; GPU driver: 580.178.04; CUDA: 13.0
- Strata: project 0.1.48, commit 5c8dd322; engine 0.1.48
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: swift-1.5-iq3_s
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack swift-iq3_s --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 131072 --kv int8 --vision --spec-min-p 0.5 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 31.5 (27.2-38.3) | 1.02 (0.92-1.11) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 318.1 (314.0-318.4) | not measured | 12.68 (12.66-12.85) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 351.3 (351.3-351.6) | not measured | 45.76 (45.72-45.77) |
| agent session, follow-up turns | 14611 (14569-14616) at the last turn | see below | up to 64 | 3 | see below | 35.8 (28.7-42.3) | 10.49 (6.97-12.59) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.5 % (69.5-69.6); earlier-turn tokens read again per session: 72 (4-74) (earlier turns that the cache did not hold; the 13730 (13729-13732) new tokens of the files are read once); first turn (nothing cached) first token: 3.67 s (3.62-3.68).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 167.5 | 3.68 | 21.2 |
| 1 | 4089 | 621 | 3468 | 276.4 | 12.594 | 34.9 |
| 2 | 5622 | 4152 | 1470 | 210.0 | 7.045 | 31.6 |
| 3 | 9055 | 5685 | 3370 | 273.3 | 12.379 | 28.7 |
| 4 | 11819 | 9111 | 2708 | 256.3 | 10.614 | 35.8 |
| 5 | 14569 | 11852 | 2717 | 260.1 | 10.494 | 35.0 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 423 s.
