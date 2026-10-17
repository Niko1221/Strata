# Community benchmark on NVIDIA GeForce RTX 3060

Measured on 2026-10-16 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3060, 12 GiB VRAM
- CPU: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz (4 cores, 8 threads); installed RAM: 46.9 GiB; storage: not measured
- OS: Linux 6.12.111+deb13-amd64; GPU driver: 615.71.09; CUDA: not read
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq2_xs
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

This run follows setup calibration. See the [pre-calibration report and comparison](../2026-10-16-community-rtx-3060-qwen3-8-flash-next-iq2-xs/README.md).
Calibration saved `--pcie-frac 0.34 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97`.
The saved engine information reports three CPU workers and 4,387 expert-cache slots (6,030 MiB), versus 4,324 slots
(5,943 MiB) before calibration. The suites therefore compare complete configurations, not an isolated parameter.

```text
strata --pack iq2_xs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --conversation-cache-mib 4096 --conversation-cache-slots 4 --adapt-async 1 --pcie-frac 0.34 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 43.7 (41.6-59.4) | 0.85 (0.76-1.00) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 894.7 (882.3-897.7) | not measured | 4.52 (4.50-4.60) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1049.2 (1047.5-1057.3) | not measured | 15.34 (15.23-15.37) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1049.9 (1048.4-1057.6) | not measured | 30.58 (30.35-30.62) |
| agent session, follow-up turns | 14563 (14521-14581) at the last turn | see below | up to 64 | 3 | see below | 47.8 (37.2-62.1) | 4.04 (3.09-4.61) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.8); earlier-turn tokens read again per session: 3 (3-74) (earlier turns that the cache did not hold; the 13729 (13728-13729) new tokens of the files are read once); first turn (nothing cached) first token: 2.36 s (2.17-2.44).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 284.1 | 2.171 | 33.8 |
| 1 | 3750 | 630 | 3120 | 730.4 | 4.302 | 42.5 |
| 2 | 5283 | 3813 | 1470 | 471.2 | 3.149 | 50.3 |
| 3 | 8727 | 5347 | 3380 | 761.3 | 4.473 | 48.3 |
| 4 | 11631 | 8769 | 2862 | 694.4 | 4.157 | 62.1 |
| 5 | 14581 | 11682 | 2899 | 728.4 | 4.015 | 43.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

No engine-log excerpt was produced for this run: the benchmark reused the server already started after calibration
(`params.started_server` is false), so it did not capture a startup log. The launch configuration and engine information
are recorded in `results.json`. The private DNS suffix in `allowed_hosts` is additionally redacted for publication.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 272 s.
