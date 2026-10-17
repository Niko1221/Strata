# Community benchmark on AMD Radeon RX 6900 XT

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 6900 XT, 16 GiB VRAM
- CPU: Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz (6 cores, 12 threads); installed RAM: 127.9 GiB; storage: not measured
- OS: Windows 11 (10.0.22631); GPU driver: 32.0.21045.5002; CUDA: not read
- Strata: project 0.1.45, commit 477ac13c (v0.1.45) with the BENCH fix #2919; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: swift-1.5-iq3_xxs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata.exe --pack swift-iq3_xxs --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill 12288 --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --kv-resident 20480 --vram-reserve-mib 1024 --pool-tasks 36
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 39.2 (31.4-41.2) | 0.88 (0.73-0.97) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 683.4 (673.2-683.5) | not measured | 5.92 (5.92-6.01) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 945.9 (939.1-947.3) | not measured | 17.03 (17.00-17.15) |
| agent session, follow-up turns | 14542 (14518-14580) at the last turn | see below | up to 64 | 3 | see below | 43.3 (33.5-51.2) | 5.79 (5.72-6.02) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.5 % (68.4-68.7); earlier-turn tokens read again per session: 74 (3-144) (earlier turns that the cache did not hold; the 13717 (13717-13720) new tokens of the files are read once); first turn (nothing cached) first token: 4.84 s (4.79-4.85).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 128.0 | 4.794 | 33.0 |
| 1 | 3743 | 620 | 3123 | 522.5 | 6.024 | 35.9 |
| 2 | 5276 | 3807 | 1469 | 256.3 | 5.773 | 37.1 |
| 3 | 8709 | 5269 | 3440 | 578.4 | 5.992 | 46.9 |
| 4 | 11617 | 8755 | 2862 | 488.0 | 5.913 | 51.2 |
| 5 | 14580 | 11610 | 2970 | 512.7 | 5.838 | 39.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 236 s.
