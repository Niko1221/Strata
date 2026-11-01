# Community benchmark on Apple M2 Ultra

Measured on 2026-11-01 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Apple M2 Ultra, 61 GiB VRAM
- CPU: arm (24 cores, 24 threads); installed RAM: 64.0 GiB; storage: APPLE SSD AP1024Z; MS950G70PCIe4 4096G
- OS: Darwin 27.2.0; GPU driver: not read; CUDA (driver supports up to): not read
- Strata: project 0.1.55, commit c62d5e3a; engine 0.1.55-metal
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (metal backend); thinking off for the suite; temperature 0

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache 64 --prefill auto --spec 2 --suffix-draft 0 --max-context 262144 --mmap-experts --vision
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 26.3 (23.0-27.1) | 1.31 (1.25-1.42) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 370.3 (343.5-378.7) | not measured | 10.89 (10.65-11.74) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 351.7 (347.4-361.2) | not measured | 45.70 (44.50-46.27) |
| agent session, follow-up turns | 12937 (12907-12944) at the last turn | see below | up to 64 | 3 | see below | 28.0 (25.9-30.3) | 11.63 (6.73-15.01) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.4 % (69.4-69.5); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 12065 (12064-12065) new tokens of the files are read once); first turn (nothing cached) first token: 4.74 s (4.67-5.12).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 120.0 | 5.12 | 19.9 |
| 1 | 3350 | 620 | 2730 | 205.1 | 13.35 | 25.9 |
| 2 | 4883 | 3413 | 1470 | 200.6 | 7.37 | 26.0 |
| 3 | 8127 | 4946 | 3181 | 212.5 | 15.009 | 27.6 |
| 4 | 10187 | 8182 | 2005 | 242.2 | 8.318 | 28.0 |
| 5 | 12907 | 10224 | 2683 | 221.4 | 12.158 | 27.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 452 s.
