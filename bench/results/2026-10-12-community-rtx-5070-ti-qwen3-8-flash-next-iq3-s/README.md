# Community benchmark on NVIDIA GeForce RTX 5070 Ti

Measured on 2026-10-12 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5070 Ti, 16 GiB VRAM
- CPU: AMD Ryzen Threadripper 7960X 24-Cores (24 cores, 48 threads); installed RAM: 63.5 GiB; storage: model files on a Crucial P310 1 TB NVMe SSD (CT1000P310SSD8)
- OS: Windows 11 (10.0.26220); GPU driver: 617.42; CUDA: 13.0 (the official ready-made Windows engine, not a source build)
- Strata: project 0.1.43.1, commit a0116b27; engine 0.1.43.1
- Background workloads and power limits: an ordinary desktop session stayed open (VS Code, PowerToys, WhatsApp, Windows Defender), with no game or other GPU compute; Windows Balanced power plan; GPU power limit 300 W as reported by `nvidia-smi`; PCIe 5.0 x16 as the maximum link `nvidia-smi` reports (the link under load was not measured); about 51 GB of RAM free before the runs. IQ2_XS, IQ3_XXS and IQ3_S were run back to back on the same day, each in its own fresh server process started by the benchmark

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged by the benchmark. It is the one setup wrote, plus `--vram-reserve-mib 1125` added by hand after engine 0.1.38 reported 87 MiB of VRAM free with IQ2_XS loaded; thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 65536 --kv int8 --vram-reserve-mib 1125
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 96.3 (83.8-106.2) | 0.34 (0.33-0.42) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 2865.4 (2701.1-2891.4) | not measured | 1.44 (1.42-1.52) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3448.1 (3423.7-3452.7) | not measured | 4.69 (4.69-4.73) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 3458.2 (3454.0-3473.1) | not measured | 9.32 (9.27-9.33) |
| agent session, follow-up turns | 14465 (14431-14472) at the last turn | see below | up to 64 | 3 | see below | 102.2 (64.0-138.8) | 1.29 (0.95-1.50) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.7-68.8); earlier-turn tokens read again per session: 3 (3-74) (earlier turns that the cache did not hold; the 13634 (13633-13636) new tokens of the files are read once); first turn (nothing cached) first token: 0.99 s (0.99-1.02).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 635.4 | 0.99 | 75.1 |
| 1 | 3712 | 631 | 3081 | 2142.0 | 1.459 | 103.4 |
| 2 | 5245 | 3775 | 1470 | 1474.4 | 1.033 | 84.9 |
| 3 | 8689 | 5309 | 3380 | 2300.7 | 1.505 | 78.8 |
| 4 | 11571 | 8711 | 2860 | 2124.8 | 1.387 | 138.8 |
| 5 | 14431 | 11585 | 2846 | 2176.7 | 1.346 | 64.0 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; storage, CUDA version, background load and power limit were added by hand after the run. Total time of the suite: 94 s.
