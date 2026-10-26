# Community benchmark on NVIDIA GeForce GTX 1080 Ti + NVIDIA GeForce GTX 1080 Ti + NVIDIA GeForce GTX 1080 Ti

Measured on 2026-10-26 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce GTX 1080 Ti + NVIDIA GeForce GTX 1080 Ti + NVIDIA GeForce GTX 1080 Ti, 33 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2670 0 @ 2.60GHz (8 cores, 16 threads); installed RAM: 31.3 GiB; storage: not measured
- OS: Linux 6.12.85+deb13-amd64; GPU driver: 570.86.10; CUDA: 12.8
- Strata: project 0.1.49.2, commit 7dd5cf18; engine 0.1.49.2
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-coder-iq1_m
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack coder-iq1_m --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-coder.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --resident-experts --control-vector-scaled Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 39.7 (34.4-49.9) | 0.95 (0.84-1.03) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 355.1 (343.9-357.3) | not measured | 11.36 (11.29-11.73) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 523.0 (521.9-524.2) | not measured | 30.76 (30.68-30.83) |
| agent session, follow-up turns | 15459 (15422-15477) at the last turn | see below | up to 64 | 3 | see below | 46.0 (35.3-58.4) | 7.93 (4.45-12.07) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.5 % (70.4-70.6); earlier-turn tokens read again per session: 4 (3-74) (earlier turns that the cache did not hold; the 14602 (14601-14602) new tokens of the files are read once); first turn (nothing cached) first token: 2.58 s (2.57-2.66).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 240.4 | 2.576 | 29.8 |
| 1 | 4986 | 631 | 4355 | 362.4 | 12.068 | 43.1 |
| 2 | 6519 | 5049 | 1470 | 299.8 | 4.946 | 43.8 |
| 3 | 9936 | 6567 | 3369 | 320.3 | 10.569 | 35.3 |
| 4 | 12665 | 9971 | 2694 | 358.8 | 7.557 | 48.8 |
| 5 | 15422 | 12705 | 2717 | 323.7 | 8.443 | 41.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 321 s.
