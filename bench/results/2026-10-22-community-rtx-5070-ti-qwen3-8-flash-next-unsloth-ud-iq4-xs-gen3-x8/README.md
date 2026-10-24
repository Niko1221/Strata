# Community benchmark on NVIDIA GeForce RTX 5070 Ti

Measured on 2026-10-22 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5070 Ti, 16 GiB VRAM
- CPU: AMD Ryzen 7 9700X 8-Core Processor (8 cores, 16 threads); installed RAM: 93.1 GiB; WD_BLACK SN7100 1 TB NVMe for model and packs
- OS: Windows 11 (10.0.26200); GPU driver: 616.92; CUDA: not read
- Strata: project 0.1.44.1, commit not a git checkout; engine 0.1.44.1
- PCIe link: PCIe Gen3 x8, forced in the BIOS (same bandwidth as Gen4 x4, a typical second slot); the second card was disabled in Device Manager; all GPUs on CPU lanes, none behind the chipset
- Memory and CPU: RAM 96 GB as 4 DIMMs (2x 32 GB + 2x 16 GB DDR5) at JEDEC DDR5-4800, no EXPO; CPU Curve Optimizer -20
- Background workloads and power limits: Windows power plan "Balanced"; both GPUs undervolted with an MSI Afterburner V/F curve, 2797 MHz at 885 mV, power limit 300 W (default), memory +1000 MHz effective (15001 MHz instead of 14001); an idle agent app (goose) was open, no other load

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack unsloth-ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --pcie-frac 0.35 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97 --vram-reserve-mib 1024 --vram-reserve-later-mib 1179
```

Environment of the config: `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=0`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 27.2 (24.8-30.8) | 1.23 (1.07-1.43) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 482.8 (479.2-483.9) | not measured | 8.38 (8.35-8.46) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 951.2 (950.6-951.8) | not measured | 16.92 (16.90-16.93) |
| agent session, follow-up turns | 14581 (14568-14609) at the last turn | see below | up to 64 | 3 | see below | 33.1 (26.3-43.3) | 8.30 (8.07-8.38) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.5-68.8); earlier-turn tokens read again per session: 4 (3-144) (earlier turns that the cache did not hold; the 13729 (13728-13730) new tokens of the files are read once); first turn (nothing cached) first token: 3.33 s (3.24-3.38).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 186.1 | 3.325 | 23.2 |
| 1 | 3756 | 634 | 3122 | 378.3 | 8.298 | 26.3 |
| 2 | 5289 | 3749 | 1540 | 191.1 | 8.108 | 26.7 |
| 3 | 8733 | 5282 | 3451 | 414.6 | 8.376 | 33.1 |
| 4 | 11657 | 8795 | 2862 | 346.1 | 8.319 | 36.3 |
| 5 | 14609 | 11710 | 2899 | 350.5 | 8.321 | 34.0 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 294 s.
