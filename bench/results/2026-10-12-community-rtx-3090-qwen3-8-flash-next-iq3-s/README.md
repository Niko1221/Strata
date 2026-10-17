# Community benchmark on NVIDIA GeForce RTX 3090

Measured on 2026-10-12 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3090, 24 GiB VRAM
- CPU: Intel(R) Core(TM) i9-10900KF CPU @ 3.70GHz (10 cores, 20 threads); installed RAM: 62.6 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 595.91.07; CUDA: 13.2
- Strata: project 0.1.43, commit 7695859a; engine 0.1.43
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 131072 --kv int8 --vision --vram-reserve-mib 700 --spec-min-p 0.5 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 62.1 (50.8-66.2) | 0.43 (0.41-0.48) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 615.8 (614.4-616.7) | not measured | 6.54 (6.53-6.56) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1200.7 (1199.7-1201.8) | not measured | 13.40 (13.38-13.41) |
| agent session, follow-up turns | 14458 (14412-14474) at the last turn | see below | up to 64 | 3 | see below | 84.7 (60.9-148.7) | 6.40 (6.17-6.48) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.8); earlier-turn tokens read again per session: 4 (2-74) (earlier turns that the cache did not hold; the 13633 (13633-13634) new tokens of the files are read once); first turn (nothing cached) first token: 2.35 s (2.33-2.39).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 263.4 | 2.33 | 42.3 |
| 1 | 3712 | 631 | 3081 | 480.2 | 6.436 | 63.0 |
| 2 | 5245 | 3776 | 1469 | 238.5 | 6.187 | 69.3 |
| 3 | 8689 | 5309 | 3380 | 523.8 | 6.475 | 83.0 |
| 4 | 11593 | 8733 | 2860 | 449.4 | 6.387 | 148.7 |
| 5 | 14458 | 11612 | 2846 | 447.5 | 6.384 | 68.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 197 s.
