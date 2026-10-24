# Community benchmark on NVIDIA GeForce RTX 5070 Ti + NVIDIA GeForce RTX 5070 Ti + NVIDIA GeForce RTX 5070 Ti

Measured on 2026-10-23 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5070 Ti + NVIDIA GeForce RTX 5070 Ti + NVIDIA GeForce RTX 5070 Ti, 48 GiB VRAM
- CPU: AMD Ryzen 7 9700X 8-Core Processor (8 cores, 16 threads); installed RAM: 93.1 GiB; WD_BLACK SN7100 1 TB NVMe for model and packs
- OS: Windows 11 (10.0.26200); GPU driver: 616.92; CUDA: not read
- Strata: project 0.1.44.1, commit not a git checkout; engine 0.1.44.1
- PCIe link: two cards PCIe Gen5 x8 (x16 slot split x8/x8), the third PCIe Gen4 x4 through an OCuLink adapter in the CPU-attached M.2 slot (Gen4 redriver, link fixed to Gen4; "auto" crashed this setup); all GPUs on CPU lanes, none behind the chipset
- Memory and CPU: RAM 96 GB as 4 DIMMs (2x 32 GB + 2x 16 GB DDR5) at JEDEC DDR5-4800, no EXPO; CPU Curve Optimizer -20
- Background workloads and power limits: Windows power plan "Balanced"; all three GPUs undervolted with an MSI Afterburner V/F curve, 2797 MHz at 885 mV, power limit 300 W (default), memory +1000 MHz effective (15001 MHz instead of 14001); an idle agent app (goose) was open, no other load

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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 68.8 (48.0-81.0) | 0.42 (0.30-0.56) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 888.9 (860.7-894.6) | not measured | 4.56 (4.54-4.69) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1864.2 (1840.5-1880.3) | not measured | 8.64 (8.56-8.76) |
| agent session, follow-up turns | 14579 (14569-14616) at the last turn | see below | up to 64 | 3 | see below | 103.9 (73.3-148.0) | 4.15 (3.61-4.40) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.6-68.8); earlier-turn tokens read again per session: 2 (2-74) (earlier turns that the cache did not hold; the 13729 (13729-13730) new tokens of the files are read once); first turn (nothing cached) first token: 2.65 s (2.65-2.66).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 231.6 | 2.65 | 53.8 |
| 1 | 3753 | 633 | 3120 | 717.4 | 4.372 | 73.3 |
| 2 | 5286 | 3816 | 1470 | 399.0 | 3.707 | 82.1 |
| 3 | 8730 | 5350 | 3380 | 772.3 | 4.4 | 97.9 |
| 4 | 11656 | 8794 | 2862 | 676.5 | 4.255 | 121.0 |
| 5 | 14616 | 11717 | 2899 | 698.9 | 4.206 | 99.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 145 s.
