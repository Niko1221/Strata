# Community benchmark on AMD Radeon RX 9060 XT (gfx1200), hipBLASLt 100500 table

Measured on 2026-10-16 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run. This is the tuned arm of an A/B pair: the plain-hipBLAS arm is `bench/results/2026-10-16-community-rx-9060-xt-swift-1-5-iq2-xs-hipblas/`, and the two runs differ only in `STRATA_HIPBLASLT_TUNING`.

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

Environment of the config: `STRATA_HIPBLASLT_TUNING=gfx1200-hipblaslt-100500.txt`. The engine took the table (`prefill gemm: hipBLASLt tuning enabled (57 rows, gfx1200, version 100500)`, the install's own log) and logged no fallbacks.

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 42.5 (37.2-47.4) | 0.77 (0.65-0.93) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 506.2 (502.3-507.5) | not measured | 7.99 (7.95-8.03) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 580.6 (579.6-581.1) | not measured | 27.70 (27.68-27.75) |
| agent session, follow-up turns | 14582 (14565-14605) at the last turn | see below | up to 64 | 3 | see below | 47.2 (38.0-58.5) | 7.96 (6.84-8.53) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

Against the plain-hipBLAS arm (`bench/results/2026-10-16-community-rx-9060-xt-swift-1-5-iq2-xs-hipblas/`, same day, same configuration, only `STRATA_HIPBLASLT_TUNING` apart) the table reads prompts 5.8 % faster: 4K fresh 478.5 -> 506.2 tok/s and 16K fresh 548.3 -> 580.6 tok/s (the two arms' ranges do not overlap). First token follows the read: 4K 8.43 -> 7.99 s, 16K 29.33 -> 27.70 s. Decode is unchanged (42.4 vs 42.5 tok/s, ranges overlap).

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.6); earlier-turn tokens read again per session: 73 (72-73) (earlier turns that the cache did not hold; the 13730 (13727-13732) new tokens of the files are read once); first turn (nothing cached) first token: 4.75 s (4.61-4.81).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 133.3 | 4.605 | 28.2 |
| 1 | 3743 | 623 | 3120 | 416.2 | 7.549 | 38.0 |
| 2 | 5276 | 3807 | 1469 | 216.2 | 6.842 | 38.8 |
| 3 | 8718 | 5340 | 3378 | 398.1 | 8.527 | 53.9 |
| 4 | 11644 | 8711 | 2933 | 365.1 | 8.09 | 58.5 |
| 5 | 14565 | 11666 | 2899 | 362.4 | 8.07 | 42.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 301 s.

`--expert-cache auto` sized the cache a little smaller in this arm (5,282 experts, 7,253 MiB) than in the plain arm (5,377, 7,382 MiB), 1.8 % fewer experts, so the faster prompt reads are not from a larger expert cache.
