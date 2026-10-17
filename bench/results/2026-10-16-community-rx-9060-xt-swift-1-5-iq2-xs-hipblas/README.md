# Community benchmark on AMD Radeon RX 9060 XT (gfx1200), plain hipBLAS

Measured on 2026-10-16 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run. This is the plain-hipBLAS arm of an A/B pair; the tuned arm (`STRATA_HIPBLASLT_TUNING=gfx1200-hipblaslt-100500.txt`) is `bench/results/2026-10-16-community-rx-9060-xt-swift-1-5-iq2-xs-hipblaslt-100500/`.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 9060 XT, 16 GiB VRAM (gfx1200)
- CPU: AMD Ryzen 9 3900X 12-Core Processor (12 cores, 24 threads); installed RAM: 63.9 GiB; storage: not measured
- OS: Windows 11 (10.0.26200); GPU driver: 32.0.31041.1004; CUDA: not read
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: not recorded

The GPU and driver lines are corrected by hand: `tools/strata_bench.py` names the adapter from Windows' display class in the registry, which still lists the owner's retired Radeon RX 470 (8 GiB, driver 31.0.21923.11000). The engine ran on the RX 9060 XT above, as its own log shows (`strata generate: GPU 0: AMD Radeon RX 9060 XT (gfx1200)`).

## Model and configuration

- Model and quantization: swift-1.5-iq2_xs
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata.exe --pack swift-iq2_xs --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS-00001-of-00002.gguf --ple-gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 65536 --kv int8 --kv-resident 32768
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 42.4 (36.6-47.8) | 0.77 (0.63-0.94) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 478.5 (472.0-479.6) | not measured | 8.43 (8.41-8.55) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 548.3 (545.5-552.8) | not measured | 29.33 (29.09-29.49) |
| agent session, follow-up turns | 14599 (14598-14608) at the last turn | see below | up to 64 | 3 | see below | 49.4 (36.0-59.1) | 7.81 (6.34-8.65) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.8-68.8); earlier-turn tokens read again per session: 3 (1-4) (earlier turns that the cache did not hold; the 13729 (13729-13730) new tokens of the files are read once); first turn (nothing cached) first token: 4.71 s (4.64-5.53).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 111.2 | 5.525 | 30.0 |
| 1 | 3739 | 616 | 3123 | 366.2 | 8.59 | 36.0 |
| 2 | 5272 | 3802 | 1470 | 204.9 | 7.231 | 37.7 |
| 3 | 8710 | 5332 | 3378 | 392.4 | 8.651 | 49.4 |
| 4 | 11636 | 8774 | 2862 | 354.2 | 8.126 | 52.4 |
| 5 | 14599 | 11699 | 2900 | 357.6 | 8.157 | 47.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 309 s.
