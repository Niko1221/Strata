# Community benchmark on NVIDIA GeForce RTX 2080 Ti

Measured on 2026-11-11 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 2080 Ti, 22 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2697 v2 @ 2.70GHz (12 cores, 24 threads); installed RAM: 128.0 GiB; storage: SSD (Windows)
- OS: Windows 11 Pro; GPU driver: not read; CUDA (driver supports up to): not read
- Strata: project 0.1.59.1, commit 6c642df7; engine 0.1.59.1
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs-v31
- Context: 204800 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 71.8 (64.5-95.5) | 0.56 (0.54-0.65) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 793.2 (791.7-798.3) | not measured | 5.11 (5.08-5.12) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1442.7 (1424.6-1449.1) | not measured | 11.21 (11.16-11.35) |
| agent session, follow-up turns | 13821 (13821-13835) at the last turn | see below | up to 64 | 3 | see below | 70.6 (59.9-82.8) | 4.51 (3.53-5.36) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.0 % (70.0-70.1); earlier-turn tokens read again per session: 74 (2-74) (earlier turns that the cache did not hold; the 12990 (12990-12990) new tokens of the files are read once); first turn (nothing cached) first token: 3.42 s (3.29-3.44).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 187.0 | 3.292 | 67.8 |
| 1 | 4091 | 648 | 3443 | 655.4 | 5.356 | 76.6 |
| 2 | 5624 | 4155 | 1469 | 422.6 | 3.53 | 70.9 |
| 3 | 8870 | 5688 | 3182 | 610.6 | 5.265 | 59.9 |
| 4 | 11083 | 8894 | 2189 | 541.6 | 4.094 | 70.2 |
| 5 | 13835 | 11126 | 2709 | 611.1 | 4.488 | 82.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 165 s.
