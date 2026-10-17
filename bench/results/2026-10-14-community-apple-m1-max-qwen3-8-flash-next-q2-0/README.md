# Community benchmark on Apple M1 Max

Measured on 2026-10-14 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Apple M1 Max, 25 GiB VRAM
- CPU: arm (10 cores, 10 threads); installed RAM: 32.0 GiB; storage: not measured
- OS: Darwin 25.6.0; GPU driver: not read; CUDA: not read
- Strata: project 0.1.44.1, commit bef1c42c; engine 0.1.44.1-metal
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-q2_0
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (metal backend); thinking off for the suite; temperature 0

```text
strata --pack q2_0 --native Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache 64 --prefill auto --spec 2 --suffix-draft 0 --max-context 32768 --mmap-experts
```

Environment of the config: `STRATA_METAL_STREAM=1`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 10.6 (8.6-12.7) | 4.09 (3.39-4.50) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 223.7 (217.8-224.1) | not measured | 18.03 (18.02-18.54) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 228.6 (224.6-230.0) | not measured | 70.33 (69.92-71.60) |
| agent session, follow-up turns | 14515 (14512-14515) at the last turn | see below | up to 64 | 3 | see below | 9.2 (8.6-12.6) | 18.10 (10.69-22.26) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.7-68.7); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 13730 (13730-13730) new tokens of the files are read once); first turn (nothing cached) first token: 6.03 s (5.97-6.09).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 101.2 | 6.091 | 9.7 |
| 1 | 3751 | 628 | 3123 | 154.6 | 20.281 | 9.0 |
| 2 | 5284 | 3814 | 1470 | 136.2 | 10.891 | 9.2 |
| 3 | 8728 | 5347 | 3381 | 152.4 | 22.264 | 9.3 |
| 4 | 11605 | 8743 | 2862 | 159.9 | 18.002 | 10.8 |
| 5 | 14512 | 11613 | 2899 | 160.0 | 18.187 | 12.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 769 s.
