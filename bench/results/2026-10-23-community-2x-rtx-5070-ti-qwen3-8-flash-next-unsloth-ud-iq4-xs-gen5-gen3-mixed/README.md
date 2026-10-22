# Community benchmark on NVIDIA GeForce RTX 5070 Ti + NVIDIA GeForce RTX 5070 Ti

Measured on 2026-10-23 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5070 Ti + NVIDIA GeForce RTX 5070 Ti, 32 GiB VRAM
- CPU: AMD Ryzen 7 9700X 8-Core Processor (8 cores, 16 threads); installed RAM: 93.1 GiB; WD_BLACK SN7100 1 TB NVMe for model and packs
- OS: Windows 11 (10.0.26200); GPU driver: 616.92; CUDA: not read
- Strata: project 0.1.44.1, commit not a git checkout; engine 0.1.44.1
- PCIe link: mixed links: one card PCIe Gen5 x8, the other Gen3 x8 forced in the BIOS (= Gen4 x4 bandwidth), as on a board with a full first slot and a weaker second one; CUDA0 is most likely the Gen5 card (lower PCI bus number), not verified; all GPUs on CPU lanes, none behind the chipset
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 54.9 (48.5-68.9) | 0.57 (0.47-0.65) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1067.5 (1058.1-1090.7) | not measured | 3.79 (3.72-3.83) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1854.3 (1848.1-1861.3) | not measured | 8.70 (8.66-8.71) |
| agent session, follow-up turns | 14589 (14577-14601) at the last turn | see below | up to 64 | 3 | see below | 69.4 (57.3-83.2) | 3.38 (2.96-3.63) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.4-68.8); earlier-turn tokens read again per session: 73 (4-143) (earlier turns that the cache did not hold; the 13729 (13728-13729) new tokens of the files are read once); first turn (nothing cached) first token: 2.30 s (2.28-2.31).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 268.4 | 2.308 | 48.7 |
| 1 | 3757 | 637 | 3120 | 871.5 | 3.622 | 57.3 |
| 2 | 5290 | 3820 | 1470 | 492.1 | 3.03 | 64.2 |
| 3 | 8734 | 5353 | 3381 | 941.6 | 3.629 | 69.2 |
| 4 | 11638 | 8776 | 2862 | 829.5 | 3.487 | 78.3 |
| 5 | 14601 | 11701 | 2900 | 853.2 | 3.426 | 64.0 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 138 s.
