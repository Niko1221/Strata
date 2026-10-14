# Community benchmark on AMD Radeon RX 9070 XT

Measured on 2026-10-14 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

**This is the A/B arm of `../2026-10-14-community-rx-9070-xt-qwen3-8-flash-next-iq3-xxs/`**, not a recommended setting: the same machine, day and configuration with `--kv-resident 32768` (KV streaming) removed. Prompt throughput 1.9-2.5% lower and first-token time 2.0-2.5% higher than with it; the decode medians are lower too (chat -4.4%, agent -6.8%) but with overlapping run-to-run ranges. The streaming run also skipped 22% fewer expert contributions (`route tail skip`), so it is the better of the two on both counts.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 9070 XT, 16 GiB VRAM
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 46.8 GiB; storage: not measured
- OS: Windows 11 (10.0.26200); GPU driver: 32.0.31036.15; CUDA: not read
- Strata: project 0.1.44.1, commit 0f2ed512; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv k8v4 --resident-experts --vram-reserve-mib 256
```

Environment of the config: `STRATA_HIPBLASLT_TUNING=gfx1201-hipblaslt-100500.txt`, `STRATA_RESIDENT_PIN=0`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 62.8 (55.5-74.5) | 0.42 (0.38-0.47) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 780.4 (779.3-784.3) | not measured | 5.17 (5.15-5.18) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 903.2 (900.6-904.9) | not measured | 17.80 (17.77-17.86) |
| agent session, follow-up turns | 14538 (14510-14563) at the last turn | see below | up to 64 | 3 | see below | 64.1 (41.1-82.1) | 4.81 (4.04-5.07) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.5-68.7); earlier-turn tokens read again per session: 74 (4-74) (earlier turns that the cache did not hold; the 13729 (13729-13730) new tokens of the files are read once); first turn (nothing cached) first token: 3.17 s (3.16-3.18).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 193.0 | 3.176 | 50.2 |
| 1 | 3744 | 621 | 3123 | 652.0 | 4.814 | 59.9 |
| 2 | 5277 | 3808 | 1469 | 365.5 | 4.042 | 64.1 |
| 3 | 8721 | 5340 | 3381 | 670.0 | 5.07 | 74.6 |
| 4 | 11637 | 8775 | 2862 | 600.1 | 4.796 | 56.4 |
| 5 | 14538 | 11639 | 2899 | 600.1 | 4.86 | 57.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 186 s.
