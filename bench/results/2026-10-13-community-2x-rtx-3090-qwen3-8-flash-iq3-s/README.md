# Community benchmark on NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090

Measured on 2026-10-13 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090, 48 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2696 v3 @ 2.30GHz (18 cores, 36 threads); installed RAM: 125.6 GiB; storage: not measured
- OS: Linux 6.8.0-124-generic; GPU driver: 580.159.03; CUDA: 13.0
- Strata: project 0.1.43.1, commit a0116b27; engine 0.1.43.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-IQ3_S
- Context: 204800 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack swift-iq3_s --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 204800 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --remote-expert-opt --spec-min-p 0.5
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 118.6 (93.5-151.9) | 0.32 (0.28-0.34) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1458.7 (1202.2-1459.4) | not measured | 2.78 (2.77-3.36) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2392.2 (2218.6-2392.5) | not measured | 6.74 (6.74-7.27) |
| agent session, follow-up turns | 14485 (14474-14489) at the last turn | see below | up to 64 | 3 | see below | 123.9 (92.6-140.6) | 2.74 (2.24-3.85) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.8); earlier-turn tokens read again per session: 73 (3-73) (earlier turns that the cache did not hold; the 13635 (13635-13636) new tokens of the files are read once); first turn (nothing cached) first token: 1.45 s (1.44-1.51).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 408.6 | 1.508 | 99.3 |
| 1 | 3715 | 633 | 3082 | 806.9 | 3.848 | 124.8 |
| 2 | 5248 | 3778 | 1470 | 534.5 | 2.775 | 108.5 |
| 3 | 8692 | 5312 | 3380 | 904.5 | 3.766 | 104.2 |
| 4 | 11615 | 8755 | 2860 | 803.0 | 3.601 | 125.3 |
| 5 | 14485 | 11639 | 2846 | 871.2 | 3.3 | 114.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 102 s.
