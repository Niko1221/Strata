# Community benchmark on NVIDIA GeForce RTX 3090

Measured on 2026-10-16 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3090, 24 GiB VRAM
- CPU: Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz (10 cores, 20 threads); installed RAM: 63.9 GiB; storage: not measured
- OS: Windows 11 (10.0.22631); GPU driver: 616.64; CUDA: not read
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: bench-v-la0-IQ3_S
- Context: 524288 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.70 --mtp rt --max-context 524288 --kv k8v4 --vision --vram-reserve-mib 700 --pcie-frac 0.00 --rope-scaling yarn --rope-scale 2 --kv-grow
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 75.5 (67.7-80.3) | 0.42 (0.34-0.51) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1085.6 (1071.3-1087.3) | not measured | 3.74 (3.72-3.79) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1634.9 (1633.6-1637.1) | not measured | 9.86 (9.85-9.87) |
| agent session, follow-up turns | 14558 (14552-14598) at the last turn | see below | up to 64 | 3 | see below | 88.9 (67.1-112.9) | 3.52 (3.23-3.75) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.5-68.8); earlier-turn tokens read again per session: 73 (3-144) (earlier turns that the cache did not hold; the 13728 (13728-13730) new tokens of the files are read once); first turn (nothing cached) first token: 1.71 s (1.68-1.72).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 366.6 | 1.683 | 54.0 |
| 1 | 3743 | 623 | 3120 | 859.0 | 3.658 | 67.1 |
| 2 | 5276 | 3807 | 1469 | 454.9 | 3.268 | 73.2 |
| 3 | 8720 | 5269 | 3451 | 930.7 | 3.753 | 83.0 |
| 4 | 11617 | 8755 | 2862 | 805.7 | 3.604 | 101.5 |
| 5 | 14552 | 11653 | 2899 | 823.8 | 3.573 | 78.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 131 s.
