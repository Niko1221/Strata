# Community benchmark on AMD Radeon RX 6900 XT

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 6900 XT, 16 GiB VRAM
- CPU: Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz (6 cores, 12 threads); installed RAM: 127.9 GiB; storage: not measured
- OS: Windows 11 (10.0.22631); GPU driver: 32.0.21045.5002; CUDA: not read
- Strata: project 0.1.45, commit 477ac13c (v0.1.45) with the BENCH fix #2919; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: swift-1.5-iq2_xs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata.exe --pack swift-iq2_xs --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill 12288 --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --mmap-experts --vram-reserve-mib 1024 --kv-resident 20480 --pool-tasks 36
```

Environment of the config: `STRATA_HIP_PROMPT_F16=1`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 35.4 (29.0-40.1) | 0.88 (0.71-1.00) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 485.1 (482.1-489.9) | not measured | 8.32 (8.24-8.37) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 682.3 (672.1-684.4) | not measured | 23.59 (23.52-23.95) |
| agent session, follow-up turns | 14534 (14531-14538) at the last turn | see below | up to 64 | 3 | see below | 49.4 (24.9-72.6) | 7.06 (5.70-7.66) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.7); earlier-turn tokens read again per session: 4 (3-75) (earlier turns that the cache did not hold; the 13718 (13717-13719) new tokens of the files are read once); first turn (nothing cached) first token: 4.71 s (4.68-4.76).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 131.2 | 4.678 | 31.6 |
| 1 | 3744 | 623 | 3121 | 437.5 | 7.18 | 24.9 |
| 2 | 5277 | 3807 | 1470 | 251.8 | 5.872 | 37.4 |
| 3 | 8710 | 5341 | 3369 | 442.3 | 7.659 | 54.7 |
| 4 | 11601 | 8739 | 2862 | 398.6 | 7.221 | 50.3 |
| 5 | 14538 | 11639 | 2899 | 410.1 | 7.113 | 36.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 280 s.
