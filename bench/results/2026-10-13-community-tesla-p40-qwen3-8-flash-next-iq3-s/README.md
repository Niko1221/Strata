# Community benchmark on Tesla P40

Measured on 2026-10-13 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Tesla P40, 22 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz (16 cores, 16 threads); installed RAM: 97.7 GiB; storage: not measured
- OS: Windows 2022Server (10.0.20348); GPU driver: 582.78; CUDA: 13.0
- Strata: project 0.1.43, commit not a git checkout; engine 0.1.43
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --pcie-frac 1.00 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 21.5 (19.7-23.6) | 1.52 (1.42-1.71) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 344.7 (342.2-345.3) | not measured | 11.72 (11.70-11.82) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 396.4 (395.1-396.4) | not measured | 40.60 (40.60-40.73) |
| agent session, follow-up turns | 14437 (14430-14476) at the last turn | see below | up to 64 | 3 | see below | 26.2 (21.2-38.6) | 9.91 (6.53-11.13) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.8-68.8); earlier-turn tokens read again per session: 3 (3-3) (earlier turns that the cache did not hold; the 13634 (13633-13634) new tokens of the files are read once); first turn (nothing cached) first token: 4.98 s (4.67-5.25).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 117.7 | 5.249 | 16.5 |
| 1 | 3712 | 631 | 3081 | 299.9 | 10.353 | 21.2 |
| 2 | 5245 | 3776 | 1469 | 225.9 | 6.584 | 22.4 |
| 3 | 8689 | 5308 | 3381 | 306.4 | 11.127 | 26.8 |
| 4 | 11573 | 8713 | 2860 | 289.8 | 9.983 | 31.5 |
| 5 | 14437 | 11591 | 2846 | 289.6 | 9.912 | 23.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 420 s.
