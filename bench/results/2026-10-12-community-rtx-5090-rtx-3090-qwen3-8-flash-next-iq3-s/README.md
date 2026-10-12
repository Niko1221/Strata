# Community benchmark on NVIDIA GeForce RTX 5090 + NVIDIA GeForce RTX 3090

Measured on 2026-10-12 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090 + NVIDIA GeForce RTX 3090, 56 GiB VRAM
- CPU: AMD Ryzen 9 7900X 12-Core Processor (12 cores, 24 threads); installed RAM: 191.1 GiB; storage: not measured
- OS: Windows 11 (10.0.26200); GPU driver: 610.88; CUDA: not read
- Strata: project 0.1.43, commit 7695859a; engine 0.1.43
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 79.1 (67.2-101.8) | 0.38 (0.31-0.43) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1944.7 (1284.2-2008.5) | not measured | 2.08 (2.04-3.16) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2282.1 (2276.4-2294.6) | not measured | 7.08 (7.04-7.10) |
| agent session, follow-up turns | 14457 (14432-14472) at the last turn | see below | up to 64 | 15 follow-up requests (3 sessions x 5 turns) | see below | 115.8 (94.8-169.2) | 1.97 (1.71-3.98) |

Ranges in the agent row are over the 15 individual follow-up requests, not the 3 sessions.

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.6-68.8); earlier-turn tokens read again per session: 4 (3-74) (earlier turns that the cache did not hold; the 13635 (13633-13635) new tokens of the files are read once); first turn (nothing cached) first token: 1.30 s (1.27-1.74).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 356.2 | 1.744 | 74.7 |
| 1 | 3715 | 633 | 3082 | 780.0 | 3.98 | 115.6 |
| 2 | 5248 | 3778 | 1470 | 526.5 | 2.832 | 103.4 |
| 3 | 8692 | 5312 | 3380 | 976.6 | 3.496 | 94.8 |
| 4 | 11599 | 8739 | 2860 | 859.6 | 3.362 | 115.8 |
| 5 | 14472 | 11626 | 2846 | 958.4 | 3.003 | 106.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Long-context recall: the repository's unchanged `tools/needle_bench.py` found **6 of 6 needles** (code word at depths 10%, 50%, 90% at 32k and 128k; actual prompt lengths 32,804-32,805 and 124,840-124,842 tokens; per-request data in `needles.json`). All answers matched the expected code words. The needle test measures recall on these six inputs, not general model quality.

Not otherwise measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the speed suite: 96 s.
