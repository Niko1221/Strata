# Community benchmark on AMD Radeon RX 6900 XT

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 6900 XT, 16 GiB VRAM
- CPU: Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz (6 cores, 12 threads); installed RAM: 127.9 GiB; storage: not measured
- OS: Windows 11 (10.0.22631); GPU driver: 32.0.21045.5002; CUDA: not read
- Strata: project 0.1.45, commit 477ac13c (v0.1.45) with the BENCH fix #2919; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --control-vector-scaled Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --vram-reserve-mib 1024 --prefill 12288 --kv-resident 20480 --pool-tasks 36
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 33.9 (29.4-35.9) | 0.94 (0.81-1.05) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 598.9 (592.8-599.1) | not measured | 6.76 (6.75-6.83) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 891.8 (885.6-893.2) | not measured | 18.06 (18.04-18.18) |
| agent session, follow-up turns | 14526 (14495-14549) at the last turn | see below | up to 64 | 3 | see below | 39.3 (30.7-52.3) | 6.62 (6.55-6.82) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.8); earlier-turn tokens read again per session: 4 (3-74) (earlier turns that the cache did not hold; the 13719 (13719-13719) new tokens of the files are read once); first turn (nothing cached) first token: 5.49 s (5.46-5.51).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 112.4 | 5.455 | 26.9 |
| 1 | 3748 | 626 | 3122 | 461.1 | 6.821 | 39.4 |
| 2 | 5281 | 3811 | 1470 | 223.5 | 6.622 | 34.1 |
| 3 | 8714 | 5345 | 3369 | 501.3 | 6.769 | 44.7 |
| 4 | 11623 | 8761 | 2862 | 428.8 | 6.722 | 35.5 |
| 5 | 14549 | 11650 | 2899 | 440.3 | 6.631 | 35.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 261 s.
