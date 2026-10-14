# Community benchmark on NVIDIA GeForce RTX 3090

Measured on 2026-10-14 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3090, 24 GiB VRAM
- CPU: Intel(R) Xeon(R) W-2235 CPU @ 3.80GHz (6 cores, 12 threads); installed RAM: 30.9 GiB; storage: not measured
- OS: Linux 7.1.13-200.fc44.x86_64; GPU driver: 610.57.04; CUDA: not read
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-q2_0
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack q2_0 --native Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --resident-experts --kv-resident 20480
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 91.1 (70.2-118.1) | 0.31 (0.29-0.40) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1667.1 (1587.0-1668.9) | not measured | 2.43 (2.43-2.55) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2160.7 (2137.7-2167.7) | not measured | 7.46 (7.44-7.54) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2198.7 (2194.5-2204.2) | not measured | 14.62 (14.59-14.65) |
| agent session, follow-up turns | 14527 (14491-14534) at the last turn | see below | up to 64 | 3 | see below | 100.1 (65.8-136.3) | 2.13 (1.72-2.35) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.3-68.6); earlier-turn tokens read again per session: 74 (73-144) (earlier turns that the cache did not hold; the 13730 (13728-13730) new tokens of the files are read once); first turn (nothing cached) first token: 1.09 s (1.09-1.12).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 553.0 | 1.122 | 58.7 |
| 1 | 3756 | 634 | 3122 | 1342.9 | 2.348 | 71.7 |
| 2 | 5289 | 3819 | 1470 | 823.1 | 1.811 | 98.2 |
| 3 | 8733 | 5282 | 3451 | 1485.9 | 2.348 | 81.9 |
| 4 | 11617 | 8755 | 2862 | 1267.9 | 2.286 | 115.4 |
| 5 | 14534 | 11635 | 2899 | 1367.6 | 2.149 | 70.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 167 s.
