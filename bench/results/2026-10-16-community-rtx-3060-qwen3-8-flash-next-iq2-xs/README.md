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

This is the pre-calibration IQ2_XS baseline. The [second run](../2026-10-16-community-rtx-3060-qwen3-8-flash-next-iq2-xs-2/README.md)
was measured on the same day after setup calibration. Both use a 4,096 MiB conversation cache with four slots and
`--adapt-async 1`. The saved baseline engine information reports three CPU workers, PCIe fraction 0.34, draft
threshold 0.50, and 4,324 expert-cache slots (5,943 MiB).

```text
strata --pack iq2_xs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --conversation-cache-mib 4096 --conversation-cache-slots 4 --adapt-async 1
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 45.7 (39.1-47.1) | 0.97 (0.84-1.17) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 883.7 (878.7-887.4) | not measured | 4.58 (4.56-4.60) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1045.5 (1045.0-1055.2) | not measured | 15.39 (15.25-15.40) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1042.5 (1041.8-1057.2) | not measured | 30.79 (30.36-30.81) |
| agent session, follow-up turns | 14581 (14571-14583) at the last turn | see below | up to 64 | 3 | see below | 48.0 (41.3-61.4) | 4.04 (3.19-4.79) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.4-68.8); earlier-turn tokens read again per session: 74 (3-144) (earlier turns that the cache did not hold; the 13728 (13728-13730) new tokens of the files are read once); first turn (nothing cached) first token: 2.31 s (2.27-2.52).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 266.7 | 2.313 | 32.7 |
| 1 | 3750 | 630 | 3120 | 746.1 | 4.212 | 41.3 |
| 2 | 5283 | 3813 | 1470 | 464.9 | 3.192 | 43.0 |
| 3 | 8727 | 5276 | 3451 | 725.3 | 4.79 | 46.1 |
| 4 | 11632 | 8770 | 2862 | 706.4 | 4.085 | 52.2 |
| 5 | 14583 | 11684 | 2899 | 708.9 | 4.124 | 42.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 279 s.

## Same-machine comparison

Medians from the saved reports, not a controlled model-only or single-setting A/B test:

| Metric | Coder IQ1_M, October 14 | IQ2_XS before calibration, October 16 | IQ2_XS after calibration, October 16 |
| --- | ---: | ---: | ---: |
| Short-chat decode, tok/s | 34.3 | 45.7 | 43.7 |
| Agent follow-up decode, tok/s | 36.6 | 48.0 | 47.8 |
| Fresh 4K prompt, tok/s | 974.3 | 883.7 | 894.7 |
| Fresh 16K prompt, tok/s | 1111.4 | 1045.5 | 1049.2 |
| Fresh 32K prompt, tok/s | 1110.3 | 1042.5 | 1049.9 |
| Short-chat first token, seconds | 1.10 | 0.97 | 0.85 |
| Agent follow-up first token, seconds | 3.54 | 4.04 | 4.04 |
| Agent follow-up cache hit, percent | 68.6 | 68.6 | 68.7 |

The [Coder report](../2026-10-14-community-rtx-3060-qwen3-8-flash-next-coder-iq1-m/README.md) is from two days earlier,
and its saved launch arguments do not explicitly set the conversation-cache budget or async adaptation. The models,
generated answers, and complete configurations differ. These runs do not isolate why IQ2_XS generated faster, and
they do not establish a coding-quality difference.

The same-day calibration comparison shows small prompt-throughput increases, lower short-chat first-token latency,
slightly lower decode medians, and unchanged agent follow-up first-token latency. It does not show a clear overall
speed gain on this suite. These are one suite before and one after calibration, with repetitions inside each suite;
cache state and background load were not controlled between suites. Expert-cache capacity also changed.

All three reports configure 131,072 context tokens, but the largest fresh prompt measured is 32,049 tokens. Full-128K
capacity, sustained memory pressure, answer quality, and recall were not tested. Live-session observations and separate
ad hoc coding checks are not included in these measurements. The shared JSON files additionally redact the private
DNS suffix left in `allowed_hosts` by the benchmark; numerical measurements are unchanged.
