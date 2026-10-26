# Community benchmark on NVIDIA GeForce GTX 1080 Ti + NVIDIA GeForce GTX 1080 Ti + NVIDIA GeForce GTX 1080 Ti

Measured on 2026-10-25 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce GTX 1080 Ti + NVIDIA GeForce GTX 1080 Ti + NVIDIA GeForce GTX 1080 Ti, 33 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2670 0 @ 2.60GHz (8 cores, 16 threads); installed RAM: 31.3 GiB; storage: not measured
- OS: Linux 6.12.85+deb13-amd64; GPU driver: 570.86.10; CUDA: 12.8
- Strata: project 0.1.49.2, commit 7dd5cf18; engine 0.1.49.2
- Background workloads and power limits: not recorded 
    (user: minimal background, 200W PL for the 1080Tis)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq2_xs
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack iq2_xs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 131072 --kv int8 --mmap-experts --control-vector-scaled Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --pcie-frac 1.00 --spec-min-p 0.70 --pool-workers 5
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 41.7 (19.7-53.3) | 1.04 (0.85-2.03) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 196.4 (96.6-277.3) | not measured | 20.76 (14.53-41.76) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 498.5 (498.0-500.9) | not measured | 32.27 (32.12-32.31) |
| agent session, follow-up turns | 15453 (15432-15497) at the last turn | see below | up to 64 | 3 | see below | 53.9 (34.8-60.0) | 11.21 (6.74-19.26) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.3 % (70.3-70.4); earlier-turn tokens read again per session: 144 (73-144) (earlier turns that the cache did not hold; the 14602 (14600-14602) new tokens of the files are read once); first turn (nothing cached) first token: 5.22 s (5.11-7.96).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 130.7 | 5.217 | 32.1 |
| 1 | 4997 | 642 | 4355 | 280.5 | 16.286 | 36.3 |
| 2 | 6530 | 4990 | 1540 | 172.0 | 9.136 | 43.2 |
| 3 | 9963 | 6523 | 3440 | 183.0 | 19.259 | 43.8 |
| 4 | 12694 | 10000 | 2694 | 234.9 | 11.717 | 34.8 |
| 5 | 15453 | 12736 | 2717 | 245.1 | 11.207 | 40.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 464 s.
