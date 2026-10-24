# Community benchmark on NVIDIA GeForce RTX 4090

Measured on 2026-10-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4090, 22 GiB VRAM
- CPU: AMD Ryzen 9 7950X 16-Core Processor (16 cores, 32 threads); installed RAM: 45.4 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 615.71.09; CUDA: not read
- Strata: project 0.1.45.3, commit e88ee39d; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Context: 204800 tokens; parallel slots: 1
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 103.2 (75.8-133.7) | 0.28 (0.24-0.38) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1482.4 (747.4-2037.2) | not measured | 2.73 (1.99-5.39) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3245.5 (3192.9-3295.5) | not measured | 4.97 (4.89-5.05) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 3303.8 (3288.4-3304.4) | not measured | 9.72 (9.72-9.77) |
| agent session, follow-up turns | 14700 (14695-14707) at the last turn | see below | up to 64 | 3 | see below | 126.1 (106.6-172.6) | 1.79 (1.55-1.97) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.4 % (69.4-69.6); earlier-turn tokens read again per session: 73 (3-74) (earlier turns that the cache did not hold; the 13813 (13811-13813) new tokens of the files are read once); first turn (nothing cached) first token: 1.21 s (1.15-1.44).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 535.2 | 1.152 | 87.7 |
| 1 | 4119 | 633 | 3486 | 1785.3 | 1.971 | 106.6 |
| 2 | 5652 | 4182 | 1470 | 893.7 | 1.66 | 121.1 |
| 3 | 9085 | 5716 | 3369 | 1742.5 | 1.955 | 118.8 |
| 4 | 11832 | 9148 | 2684 | 1498.2 | 1.808 | 152.1 |
| 5 | 14700 | 11893 | 2807 | 1565.0 | 1.809 | 133.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 111 s.
