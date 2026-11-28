# Community benchmark on Intel(R) Arc(TM) Pro B70 Graphics

Measured on 2026-10-21 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Intel(R) Arc(TM) Pro B70 Graphics, 32 GiB VRAM
- CPU: Intel(R) Core(TM) i9-9900 CPU @ 3.10GHz (8 cores, 16 threads); installed RAM: 63.8 GiB; storage: not measured
- OS: Windows 11 (10.0.26200); GPU driver: 32.0.101.8976; CUDA: not read
- Strata: project 0.1.47, commit 5fbb7cfa; engine 0.1.47
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-coder-iq1_m
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (sycl backend); thinking off for the suite; temperature 0

```text
strata.exe --pack coder-iq1_m --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-coder.bin --expert-cache auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 32768 --kv int8 --stream-experts --vram-reserve-mib 1024 --prefill 4096
```

Environment of the config: `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `STRATA_VERIFY_DEVICE_PLAN=1`, `STRATA_VERIFY_NO_HOST=1`, `STRATA_STAGER_THREADS=12`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 50.1 (34.4-67.6) | 0.45 (0.44-0.47) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1436.0 (1316.3-1438.1) | not measured | 2.85 (2.85-3.10) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1386.8 (1383.5-1410.4) | not measured | 11.64 (11.44-11.66) |
| agent session, follow-up turns | 14495 (14457-14557) at the last turn | see below | up to 64 | 3 | see below | 62.1 (42.1-78.2) | 2.53 (1.55-3.02) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.5 % (69.4-69.6); earlier-turn tokens read again per session: 4 (2-4) (earlier turns that the cache did not hold; the 13728 (13728-13728) new tokens of the files are read once); first turn (nothing cached) first token: 0.94 s (0.93-0.97).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 645.5 | 0.972 | 65.5 |
| 1 | 4114 | 645 | 3469 | 1169.1 | 3.019 | 60.0 |
| 2 | 5641 | 4174 | 1467 | 856.0 | 1.771 | 62.1 |
| 3 | 9074 | 5705 | 3369 | 1135.5 | 3.002 | 66.0 |
| 4 | 11825 | 9117 | 2708 | 1074.2 | 2.581 | 57.5 |
| 5 | 14557 | 11840 | 2717 | 1098.3 | 2.534 | 54.8 |

### Skipped and failed

- skipped 32K: the context is 32768 tokens

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 121 s.
