# Community benchmark on NVIDIA GeForce RTX 5070 Ti

Measured on 2026-10-12 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5070 Ti, 16 GiB VRAM
- CPU: AMD Ryzen Threadripper 7960X 24-Cores (24 cores, 48 threads); installed RAM: 63.5 GiB; storage: model files on a Crucial P310 1 TB NVMe SSD (CT1000P310SSD8)
- OS: Windows 11 (10.0.26220); GPU driver: 617.42; CUDA: 13.0 (the official ready-made Windows engine, not a source build)
- Strata: project 0.1.43.1, commit a0116b27; engine 0.1.43.1
- Background workloads and power limits: an ordinary desktop session stayed open (VS Code, PowerToys, WhatsApp, Windows Defender), with no game or other GPU compute; Windows Balanced power plan; GPU power limit 300 W as reported by `nvidia-smi`; PCIe 5.0 x16 as the maximum link `nvidia-smi` reports (the link under load was not measured); about 51 GB of RAM free before the runs. IQ2_XS, IQ3_XXS and IQ3_S were run back to back on the same day, each in its own fresh server process started by the benchmark

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq2_xs
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged by the benchmark. It is the one setup wrote, plus `--vram-reserve-mib 1125` added by hand after engine 0.1.38 reported 87 MiB of VRAM free with IQ2_XS loaded; thinking off for the suite; temperature 0

```text
strata.exe --pack iq2_xs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 65536 --kv int8 --kv-resident 32768 --vram-reserve-mib 1125
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 123.6 (107.9-141.6) | 0.28 (0.23-0.30) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 2795.2 (2700.6-2804.0) | not measured | 1.47 (1.45-1.51) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3287.3 (3249.0-3297.7) | not measured | 4.91 (4.90-4.97) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 3319.9 (3306.5-3323.0) | not measured | 9.70 (9.69-9.73) |
| agent session, follow-up turns | 14482 (14453-14523) at the last turn | see below | up to 64 | 3 | see below | 130.0 (108.3-164.6) | 1.32 (0.90-1.54) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.4-68.8); earlier-turn tokens read again per session: 73 (4-214) (earlier turns that the cache did not hold; the 13634 (13634-13634) new tokens of the files are read once); first turn (nothing cached) first token: 0.82 s (0.79-0.83).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 765.6 | 0.824 | 97.4 |
| 1 | 3701 | 620 | 3081 | 2156.4 | 1.458 | 113.9 |
| 2 | 5234 | 3764 | 1470 | 1587.3 | 0.953 | 120.7 |
| 3 | 8678 | 5227 | 3451 | 2286.6 | 1.54 | 128.0 |
| 4 | 11591 | 8732 | 2859 | 2107.8 | 1.388 | 146.2 |
| 5 | 14453 | 11607 | 2846 | 2142.6 | 1.364 | 108.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; storage, CUDA version, background load and power limit were added by hand after the run. Total time of the suite: 91 s.
