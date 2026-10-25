# Community benchmark on NVIDIA GeForce RTX 3090

Measured on 2026-10-25 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3090, 24 GiB VRAM
- CPU: Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz (10 cores, 20 threads); installed RAM: 63.9 GiB; storage: not measured
- OS: Windows 11 (10.0.22631); GPU driver: 616.64; CUDA: not read
- Strata: project 0.1.50, commit 41cf05a0; engine 0.1.50
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: bench-v-base-IQ3_S
- Context: 524288 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.70 --mtp rt --max-context 524288 --kv k8v4 --vision --vram-reserve-mib 700 --rope-scaling yarn --rope-scale 2 --kv-grow
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
| short chat | 200 (197-205) | 0 | 100 (100-100) | 9 | not measured | 99.0 (67.9-114.3) | 0.73 (0.71-0.74) |
| 4K prompt, fresh | 4024 (4024-4024) | 0 (0-0) | 24 cap | 3 | 1102.0 (1094.7-1108.8) | not measured | 3.67 (3.67-3.71) |
| 16K prompt, fresh | 16056 (16056-16056) | 0 (0-0) | 24 cap | 3 | 1679.0 (1677.3-1681.3) | not measured | 9.60 (9.58-9.62) |
| agent session, follow-up turns | 12968 (12968-12975) at the last turn | see below | up to 64 | 3 | see below | 123.0 (98.1-141.4) | 3.40 (3.23-3.64) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.8 % (69.8-70.0); earlier-turn tokens read again per session: 72 (3-74) (earlier turns that the cache did not hold; the 11896 (11896-11897) new tokens of the files are read once); first turn (nothing cached) first token: 1.85 s (1.84-1.87).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 811 | 0 | 811 | 452.4 | 1.838 | 60.9 |
| 1 | 3401 | 822 | 2579 | 748.9 | 3.47 | 99.9 |
| 2 | 4934 | 3394 | 1540 | 472.8 | 3.296 | 98.1 |
| 3 | 8178 | 4997 | 3181 | 886.3 | 3.643 | 114.1 |
| 4 | 10206 | 8241 | 1965 | 585.9 | 3.379 | 119.8 |
| 5 | 12975 | 10269 | 2706 | 783.1 | 3.48 | 107.0 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 130 s.
