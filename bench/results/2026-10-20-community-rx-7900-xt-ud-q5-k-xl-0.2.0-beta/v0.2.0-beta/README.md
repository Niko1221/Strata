# Strata v0.2.0-beta, RX 7900 XT 20 GB

Measured on 2026-10-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XT (gfx1100), 20 GiB VRAM
- CPU: AMD EPYC 7K62 48-Core Processor (48 cores, 48 threads); installed RAM: 128.0 GiB; storage: not measured
- OS: Linux 7.0.14-19-pve; GPU driver: not read; CUDA: not read
- Strata: project 0.1.45.3, commit not a git checkout; engine 0.1.45.3
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-q5_k_xl
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 18.6 (14.4-22.3) | 1.79 (1.51-2.61) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 827.7 (798.6-834.0) | not measured | 4.92 (4.89-5.09) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1152.6 (1094.2-1158.0) | not measured | 13.98 (13.92-14.75) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1050.2 (1045.2-1119.5) | not measured | 30.59 (28.70-30.75) |
| agent session, follow-up turns | 14602 (14567-14621) at the last turn | see below | up to 64 | 3 | see below | 21.2 (12.3-27.2) | 5.67 (4.35-6.55) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.3-69.4); earlier-turn tokens read again per session: 74 (3-74) (earlier turns that the cache did not hold; the 13745 (13744-13746) new tokens of the files are read once); first turn (nothing cached) first token: 3.88 s (3.41-4.00).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 181.2 | 3.41 | 19.0 |
| 1 | 4042 | 624 | 3418 | 567.3 | 6.073 | 18.9 |
| 2 | 5575 | 4035 | 1540 | 354.8 | 4.394 | 21.2 |
| 3 | 9008 | 5638 | 3370 | 567.8 | 5.984 | 25.2 |
| 4 | 11755 | 9071 | 2684 | 413.5 | 6.555 | 27.2 |
| 5 | 14602 | 11795 | 2807 | 489.6 | 5.798 | 12.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 373 s.
