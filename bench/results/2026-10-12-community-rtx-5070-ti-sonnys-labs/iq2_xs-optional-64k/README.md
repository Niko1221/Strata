# Community benchmark on NVIDIA GeForce RTX 5070 Ti

Measured on 2026-10-12 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5070 Ti, 16 GiB VRAM
- CPU: AMD Ryzen 5 5600G with Radeon Graphics (6 cores, 12 threads); installed RAM: 58.8 GiB; storage: not measured
- OS: Linux 7.0.14-17-pve; GPU driver: 615.71.09; CUDA: not read
- Strata: project 0.1.43, commit 7695859a; engine 0.1.43
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq2_xs
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack iq2_xs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 65536 --kv int8 --resident-experts --vram-reserve-mib 1024 --kv-resident 32768
```

## Method

Run by `python tools/strata_bench.py --clients 2 --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 81.0 (66.5-89.8) | 0.42 (0.34-0.44) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1451.8 (1399.0-1456.8) | not measured | 2.78 (2.77-2.89) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2153.8 (2138.5-2157.2) | not measured | 7.48 (7.46-7.53) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2183.4 (2183.3-2184.3) | not measured | 14.71 (14.71-14.71) |
| agent session, follow-up turns | 14480 (14478-14501) at the last turn | see below | up to 64 | 3 | see below | 86.5 (71.5-120.9) | 2.58 (2.30-2.91) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.7); earlier-turn tokens read again per session: 74 (73-74) (earlier turns that the cache did not hold; the 13633 (13633-13633) new tokens of the files are read once); first turn (nothing cached) first token: 1.24 s (1.24-1.25).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 499.6 | 1.237 | 49.1 |
| 1 | 3701 | 621 | 3080 | 1091.9 | 2.841 | 71.6 |
| 2 | 5234 | 3764 | 1470 | 614.9 | 2.409 | 80.5 |
| 3 | 8678 | 5227 | 3451 | 1193.3 | 2.913 | 81.2 |
| 4 | 11585 | 8725 | 2860 | 1048.7 | 2.75 | 102.4 |
| 5 | 14478 | 11632 | 2846 | 1070.4 | 2.682 | 86.5 |

### Concurrency: 2 clients

Total 65.9 tok/s (54.5-69.4); first token p50 1.48 s, p95 2.988 s; request latency p50 3.177 s, p95 4.495 s; errors 0. Server parallel slots: 1 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 161 s.
