# Community benchmark on Quadro GV100

Measured on 2026-10-18 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Quadro GV100, 32 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-1650 v4 @ 3.60GHz (6 cores, 12 threads); installed RAM: 125.7 GiB; storage: not measured
- OS: Linux 6.8.0-117-generic; GPU driver: 580.159.03; CUDA: 13.0
- Strata: project 0.1.45, commit 477ac13c; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-q4_k_xl
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata <host> (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 42.3 (34.7-48.3) | 0.77 (0.59-1.05) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 799.5 (797.7-800.2) | not measured | 5.06 (5.05-5.07) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1357.6 (1353.1-1357.9) | not measured | 11.87 (11.87-11.91) |
| agent session, follow-up turns | 14578 (14565-14596) at the last turn | see below | up to 64 | 3 | see below | 55.0 (34.2-71.6) | 5.39 (5.06-5.51) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.5-68.8); earlier-turn tokens read again per session: 73 (1-143) (earlier turns that the cache did not hold; the 13718 (13714-13719) new tokens of the files are read once); first turn (nothing cached) first token: 2.23 s (2.22-2.25).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 276.8 | 2.229 | 31.3 |
| 1 | 3741 | 621 | 3120 | 583.3 | 5.381 | 34.2 |
| 2 | 5274 | 3734 | 1540 | 303.7 | 5.101 | 42.7 |
| 3 | 8707 | 5267 | 3440 | 627.7 | 5.514 | 55.0 |
| 4 | 11633 | 8771 | 2862 | 533.8 | 5.395 | 58.3 |
| 5 | 14596 | 11696 | 2900 | 538.5 | 5.424 | 42.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the <host>).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the <host>, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 197 s.
