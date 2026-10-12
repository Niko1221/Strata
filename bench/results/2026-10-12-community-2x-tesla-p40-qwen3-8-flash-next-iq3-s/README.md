# Community benchmark on Tesla P40 + Tesla P40

Measured on 2026-10-12 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Tesla P40 + Tesla P40, 48 GiB VRAM; PCIe link: Gen 3 x8 each
- CPU: Intel(R) Core(TM) i5-6500 CPU @ 3.20GHz (4 cores, 4 threads); installed RAM: 39.0 GiB; storage: USB SSD
- OS: Linux 6.12.111+deb13-amd64; GPU driver: 580.178.04 (supports CUDA up to 13.0); CUDA: 12.9 (engine-cuda12)
- Strata: project 0.1.43, commit 7695859a; engine 0.1.43
- Background workloads and power limits: nothing significant running; 250 W per card (default)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 131072 --kv int8 --resident-experts --vision --vram-reserve-mib 700 --pcie-frac 1.00 --spec-min-p 0.70
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 36.0 (28.8-44.7) | 0.89 (0.75-1.04) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 344.1 (326.4-345.5) | not measured | 11.72 (11.68-12.36) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 509.0 (506.8-509.5) | not measured | 31.60 (31.56-31.73) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 634.8 (633.4-634.9) | not measured | 50.55 (50.54-50.66) |
| agent session, follow-up turns | 14446 (14432-14472) at the last turn | see below | up to 64 | 3 | see below | 37.4 (29.1-46.2) | 9.68 (5.92-11.12) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.8); earlier-turn tokens read again per session: 3 (3-74) (earlier turns that the cache did not hold; the 13633 (13633-13634) new tokens of the files are read once); first turn (nothing cached) first token: 3.70 s (3.69-3.91).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 163.9 | 3.913 | 23.8 |
| 1 | 3706 | 626 | 3080 | 301.3 | 10.6 | 33.9 |
| 2 | 5239 | 3770 | 1469 | 241.7 | 6.244 | 39.7 |
| 3 | 8683 | 5302 | 3381 | 307.7 | 11.121 | 31.9 |
| 4 | 11566 | 8706 | 2860 | 293.4 | 9.867 | 39.2 |
| 5 | 14432 | 11586 | 2846 | 296.1 | 9.68 | 34.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; storage, PCIe link, background load and power limits were added by hand. Total time of the suite: 504 s.
