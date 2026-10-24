# Community benchmark on NVIDIA GeForce RTX 5070 Ti

Measured on 2026-10-22 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5070 Ti, 16 GiB VRAM
- CPU: AMD Ryzen 7 9700X 8-Core Processor (8 cores, 16 threads); installed RAM: 93.1 GiB; WD_BLACK SN7100 1 TB NVMe for model and packs
- OS: Windows 11 (10.0.26200); GPU driver: 616.92; CUDA: not read
- Strata: project 0.1.44.1, commit not a git checkout; engine 0.1.44.1
- PCIe link: PCIe Gen5 x8; the second RTX 5070 Ti was disabled in Device Manager (with it present, the single-card run is slower); all GPUs on CPU lanes, none behind the chipset
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 61.2 (54.8-77.1) | 0.53 (0.44-0.61) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1849.7 (1807.8-1856.5) | not measured | 2.21 (2.20-2.25) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3309.2 (3273.0-3313.0) | not measured | 4.90 (4.89-4.93) |
| agent session, follow-up turns | 14611 (14541-14613) at the last turn | see below | up to 64 | 3 | see below | 68.1 (59.4-83.2) | 2.29 (2.16-2.42) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.7-68.8); earlier-turn tokens read again per session: 4 (3-73) (earlier turns that the cache did not hold; the 13729 (13729-13730) new tokens of the files are read once); first turn (nothing cached) first token: 1.12 s (1.09-1.15).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 540.6 | 1.147 | 50.3 |
| 1 | 3756 | 634 | 3122 | 1313.1 | 2.419 | 66.8 |
| 2 | 5289 | 3819 | 1470 | 681.1 | 2.199 | 59.4 |
| 3 | 8733 | 5353 | 3380 | 1428.3 | 2.405 | 60.0 |
| 4 | 11617 | 8755 | 2862 | 1221.2 | 2.394 | 79.0 |
| 5 | 14541 | 11642 | 2899 | 1272.7 | 2.316 | 62.0 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 96 s.
