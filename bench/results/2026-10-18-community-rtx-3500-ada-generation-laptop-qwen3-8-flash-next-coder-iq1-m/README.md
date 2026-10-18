# Community benchmark on NVIDIA RTX 3500 Ada Generation Laptop GPU

Measured on 2026-10-18 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA RTX 3500 Ada Generation Laptop GPU, 12 GiB VRAM
- CPU: 13th Gen Intel(R) Core(TM) i9-13950HX (24 cores, 32 threads); installed RAM: 31.7 GiB; storage: not measured
- OS: Windows 11 (10.0.26200); GPU driver: 616.92; CUDA: not read
- Strata: project 0.1.45, commit 477ac13c; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-coder-iq1_m
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack coder-iq1_m --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-coder.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 32768 --kv int8 --resident-experts --pcie-frac 0.00 --spec-min-p 0.50 --pool-workers 15
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 25.1 (17.5-34.5) | 1.30 (0.97-5.97) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 547.0 (520.6-728.2) | not measured | 7.42 (5.69-7.79) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 779.1 (745.4-835.9) | not measured | 20.70 (19.31-21.61) |
| agent session, follow-up turns | 14580 (14553-14600) at the last turn | see below | up to 64 | 3 | see below | 23.7 (18.1-36.6) | 4.39 (2.46-7.35) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.8); earlier-turn tokens read again per session: 74 (4-75) (earlier turns that the cache did not hold; the 13718 (13718-13719) new tokens of the files are read once); first turn (nothing cached) first token: 3.89 s (2.21-4.47).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 139.5 | 4.466 | 19.9 |
| 1 | 3758 | 635 | 3123 | 463.6 | 6.927 | 19.5 |
| 2 | 5291 | 3751 | 1540 | 401.7 | 3.94 | 27.4 |
| 3 | 8724 | 5354 | 3370 | 796.3 | 4.355 | 23.0 |
| 4 | 11635 | 8773 | 2862 | 590.0 | 4.97 | 23.7 |
| 5 | 14580 | 11681 | 2899 | 563.3 | 5.231 | 26.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 281 s.
