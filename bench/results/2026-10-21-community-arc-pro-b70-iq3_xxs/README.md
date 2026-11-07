# Community benchmark on Intel(R) Arc(TM) Pro B70 Graphics

Measured on 2026-10-21 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Intel(R) Arc(TM) Pro B70 Graphics, 32 GiB VRAM
- CPU: Intel(R) Core(TM) i9-9900 CPU @ 3.10GHz (8 cores, 16 threads); installed RAM: 63.8 GiB; storage: not measured
- OS: Windows 11 (10.0.26200); GPU driver: 32.0.101.8976; CUDA: not read
- Strata: project 0.1.46.1, commit 132f0fe5; engine 0.1.46.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (sycl backend); thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile expert-profile-sycl.bin --expert-cache auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 32768 --kv int8 --stream-experts --vram-reserve-mib 1024 --prefill 4096
```

Environment of the config: `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `STRATA_VERIFY_DEVICE_PLAN=1`, `STRATA_VERIFY_NO_HOST=1`, `STRATA_STAGER_THREADS=12`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 38.3 (30.7-42.9) | 1.06 (0.99-1.11) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 906.4 (877.8-910.6) | not measured | 4.49 (4.45-4.63) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 955.0 (946.3-956.2) | not measured | 16.87 (16.86-17.02) |
| agent session, follow-up turns | 14692 (14688-14727) at the last turn | see below | up to 64 | 3 | see below | 43.6 (34.0-57.4) | 5.73 (4.52-6.30) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.2 % (69.2-69.5); earlier-turn tokens read again per session: 143 (3-144) (earlier turns that the cache did not hold; the 13826 (13825-13828) new tokens of the files are read once); first turn (nothing cached) first token: 4.19 s (4.12-4.23).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 145.9 | 4.229 | 39.1 |
| 1 | 4106 | 643 | 3463 | 555.0 | 6.294 | 49.2 |
| 2 | 5639 | 4099 | 1540 | 328.7 | 4.747 | 45.9 |
| 3 | 9072 | 5632 | 3440 | 551.4 | 6.3 | 35.8 |
| 4 | 11844 | 9136 | 2708 | 480.9 | 5.693 | 54.0 |
| 5 | 14727 | 11907 | 2820 | 487.9 | 5.839 | 34.0 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 234 s.
