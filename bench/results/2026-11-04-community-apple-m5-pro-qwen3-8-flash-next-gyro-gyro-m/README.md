# Community benchmark on Apple M5 Pro

Measured on 2026-11-04 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Apple M5 Pro, 52 GiB VRAM
- CPU: arm (18 cores, 18 threads); installed RAM: 64.0 GiB; storage: not measured
- OS: Darwin 27.0.0; GPU driver: not read; CUDA: not read
- Strata: project 0.1.51, commit 2d7722aa; engine 0.1.50-metal
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-gyro-gyro-m
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (metal backend); thinking off for the suite; temperature 0

```text
strata --pack gyro-gyro-m --native Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf --ple-gguf Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf --expert-profile expert-profile.bin --expert-cache 64 --prefill auto --spec 2 --suffix-draft 0 --max-context 32768 --mmap-experts --vision
```

Environment of the config: `STRATA_METAL_STREAM=1`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 25.7 (25.0-26.2) | 0.90 (0.81-0.99) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 743.4 (720.9-769.6) | not measured | 5.45 (5.26-5.62) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 698.7 (621.6-701.4) | not measured | 23.03 (22.94-25.89) |
| agent session, follow-up turns | 12749 (12732-12749) at the last turn | see below | up to 64 | 3 | see below | 22.5 (21.3-23.5) | 5.81 (3.54-7.90) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.2 % (69.2-69.2); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 11897 (11897-11897) new tokens of the files are read once); first turn (nothing cached) first token: 1.35 s (1.34-1.43).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 471.1 | 1.345 | 23.2 |
| 1 | 3210 | 629 | 2581 | 437.9 | 5.953 | 21.5 |
| 2 | 4743 | 3273 | 1470 | 422.0 | 3.544 | 22.4 |
| 3 | 7987 | 4806 | 3181 | 466.6 | 6.872 | 22.9 |
| 4 | 9988 | 8024 | 1964 | 430.6 | 4.618 | 21.3 |
| 5 | 12732 | 10026 | 2706 | 427.7 | 6.388 | 21.5 |

### Skipped and failed

- skipped 32K: the context is 32768 tokens

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 281 s.
