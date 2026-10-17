# Community benchmark on NVIDIA GeForce RTX 3080 Laptop GPU

Measured on 2026-10-11 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3080 Laptop GPU, 16 GiB VRAM
- CPU: AMD Ryzen 9 5900HX with Radeon Graphics (8 cores, 16 threads); installed RAM: 63.9 GiB; storage: not measured
- OS: Windows 11 (10.0.26200); GPU driver: 617.14; CUDA: not read
- Strata: project 0.1.43, commit 7695859a; engine 0.1.43
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-q2_0
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack q2_0 --native Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 65536 --kv int8 --kv-resident 32768 --spec-min-p 0.5
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 65.4 (60.1-81.1) | 0.49 (0.42-0.59) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 878.6 (876.2-878.7) | not measured | 4.61 (4.60-4.61) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1379.7 (1369.1-1379.7) | not measured | 11.68 (11.68-11.77) |
| agent session, follow-up turns | 14412 (14398-14415) at the last turn | see below | up to 64 | 3 | see below | 73.8 (31.2-88.9) | 4.78 (4.37-4.92) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.5 % (68.4-68.6); earlier-turn tokens read again per session: 74 (73-144) (earlier turns that the cache did not hold; the 13633 (13633-13635) new tokens of the files are read once); first turn (nothing cached) first token: 1.92 s (1.90-1.99).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 309.6 | 1.99 | 49.3 |
| 1 | 3710 | 628 | 3082 | 647.3 | 4.807 | 74.3 |
| 2 | 5243 | 3774 | 1469 | 337.3 | 4.398 | 75.2 |
| 3 | 8687 | 5236 | 3451 | 707.7 | 4.92 | 54.7 |
| 4 | 11559 | 8699 | 2860 | 601.8 | 4.804 | 85.6 |
| 5 | 14415 | 11569 | 2846 | 598.4 | 4.792 | 47.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 159 s.
