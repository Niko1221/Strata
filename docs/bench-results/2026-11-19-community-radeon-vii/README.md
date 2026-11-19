# Community benchmark on AMD Radeon VII (gfx906)

Measured by [Zuzik01](https://github.com/Zuzik01). The same machine also ran the standard
benchmark (`strata bench run --suite standard-v1`): 21 of 21 requests, 0 failed, 12 of 12 sanity
checks; its report is in [standard-v1/](standard-v1/README.md). The gfx906 build needs the two
stream-priority mappings from #6246; see the comment there.

Measured on 2026-11-19 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run for this suite; the standard-v1 suite in `standard-v1/` did run its twelve sanity checks on the same server.

## Hardware and software

- GPU and VRAM: AMD Radeon VII (gfx906, Vega 20, wave64), 16 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2690 v3 @ 2.60GHz (12 cores, 24 threads); installed RAM: 85.9 GiB; storage: 7 local SATA SSDs/HDDs plus 2 NVMe devices (model lists trimmed)
- OS: Linux 7.0.14-30-pve (Debian 13); GPU driver: the kernel amdgpu driver; HIP runtime: ROCm 7.14 (the engine image carries it, no system ROCm install)
- Strata: v0.2.16 source build, commit 64524fb9fc82222ddce554c0643b64f056ec8366; engine 0.1.62
- Background workloads: other containers on the same host were idle; the GPU served only these runs

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata --pack iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill 2048 --kv int8 --max-context 32768
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 31.6 (26.6-35.2) | 1.22 (1.15-1.57) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 348.3 (344.1-348.3) | not measured | 11.58 (11.58-11.74) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 347.8 (345.9-350.6) | not measured | 46.22 (45.86-46.47) |
| agent session, follow-up turns | 9061 (9057-9090) at the last turn | see below | up to 64 | 3 | see below | 37.1 (29.8-45.6) | 5.40 (5.08-5.61) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 71.1 % (71.0-71.1); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 8207 (8207-8208) new tokens of the files are read once); first turn (nothing cached) first token: 4.61 s (4.39-4.78).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 140.1 | 4.389 | 20.2 |
| 1 | 2273 | 631 | 1642 | 325.2 | 5.09 | 29.8 |
| 2 | 3979 | 2336 | 1643 | 311.7 | 5.313 | 38.2 |
| 3 | 5685 | 4042 | 1643 | 308.3 | 5.374 | 42.0 |
| 4 | 7379 | 5737 | 1642 | 307.7 | 5.378 | 42.3 |
| 5 | 9061 | 7419 | 1642 | 303.4 | 5.456 | 43.0 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 350 s.
