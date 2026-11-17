# Community benchmark on Intel(R) Arc(TM) Pro B70 Graphics

Measured on 2026-10-25 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Intel(R) Arc(TM) Pro B70 Graphics, 32 GiB VRAM
- CPU: Intel(R) Core(TM) i9-9900 CPU @ 3.10GHz (8 cores, 16 threads); installed RAM: 63.8 GiB; storage: not measured
- OS: Windows 11 (10.0.26200); GPU driver: 32.0.101.8976; CUDA: not read
- Strata: project 0.1.51, commit b0837aa5; engine 0.1.51
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-coder-iq1_m
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 62.6 (59.3-94.2) | 0.47 (0.44-0.53) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1085.9 (1056.8-1098.5) | not measured | 3.76 (3.70-3.85) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1192.3 (1190.3-1197.6) | not measured | 13.52 (13.47-13.56) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1174.9 (1161.4-1179.2) | not measured | 27.34 (27.25-27.66) |
| agent session, follow-up turns | 12718 (12708-12728) at the last turn | see below | up to 64 | 3 | see below | 69.2 (58.3-84.0) | 3.02 (1.97-3.73) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.2 % (69.0-69.2); earlier-turn tokens read again per session: 5 (5-75) (earlier turns that the cache did not hold; the 11896 (11896-11897) new tokens of the files are read once); first turn (nothing cached) first token: 1.89 s (1.58-2.01).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 398.5 | 1.583 | 64.9 |
| 1 | 3213 | 632 | 2581 | 821.5 | 3.19 | 68.3 |
| 2 | 4746 | 3276 | 1470 | 695.4 | 2.157 | 71.9 |
| 3 | 7990 | 4809 | 3181 | 865.8 | 3.73 | 68.7 |
| 4 | 9989 | 8025 | 1964 | 800.0 | 2.515 | 68.1 |
| 5 | 12718 | 10012 | 2706 | 802.9 | 3.406 | 62.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 226 s.
