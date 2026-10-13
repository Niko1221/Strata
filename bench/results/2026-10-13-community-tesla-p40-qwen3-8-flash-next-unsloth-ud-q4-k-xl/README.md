# Community benchmark on Tesla P40

Measured on 2026-10-13 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Tesla P40, 22 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz (16 cores, 16 threads); installed RAM: 97.7 GiB; storage: not measured
- OS: Windows 2022Server (10.0.20348); GPU driver: 582.78; CUDA: 13.0
- Strata: project 0.1.43, commit not a git checkout; engine 0.1.43
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-q4_k_xl
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack unsloth-ud-q4_k_xl --native Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --resident-budget-gib 70 --vision --vram-reserve-mib 700 --pcie-frac 1.00 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 9.4 (7.9-10.1) | 3.07 (2.72-3.47) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 244.5 (84.2-245.4) | not measured | 16.51 (16.46-47.89) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 333.1 (195.9-333.7) | not measured | 48.30 (48.21-82.08) |
| agent session, follow-up turns | 14463 (14418-14474) at the last turn | see below | up to 64 | 3 | see below | 11.4 (7.5-16.0) | 14.44 (11.28-68.84) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.5-68.8); earlier-turn tokens read again per session: 4 (3-144) (earlier turns that the cache did not hold; the 13633 (13632-13634) new tokens of the files are read once); first turn (nothing cached) first token: 8.42 s (8.20-9.76).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 63.7 | 9.764 | 5.2 |
| 1 | 3701 | 621 | 3080 | 45.0 | 68.837 | 7.5 |
| 2 | 5234 | 3764 | 1470 | 38.2 | 38.698 | 10.0 |
| 3 | 8678 | 5297 | 3381 | 53.0 | 64.034 | 8.6 |
| 4 | 11554 | 8694 | 2860 | 49.1 | 58.48 | 13.0 |
| 5 | 14418 | 11572 | 2846 | 65.4 | 43.667 | 11.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 949 s.
