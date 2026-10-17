# Community benchmark on NVIDIA GeForce RTX 3060 + NVIDIA GeForce RTX 4060 Ti

Measured on 2026-10-11 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

| | Card | VRAM | Layers | PCIe link | Host link, measured by the engine |
| --- | --- | ---: | --- | --- | ---: |
| CUDA0 | NVIDIA GeForce RTX 3060 | 12 GB | 0-20 | **x4 Gen3** (max width 4 of the card's 16) | **2.9 GB/s** |
| CUDA1 | NVIDIA GeForce RTX 4060 Ti | 16 GB | 21-47 and the head | **x8 Gen4** (max width 8 of 8) | **13.0 GB/s** |
| | | **28 GiB** | | | **4.5x apart** |

The 3060's x4 slot is the second slot of a B550 board, which is what makes this pair asymmetric. The engine probes the host->device link itself at every start and, reading 2.9 GB/s, drops the 3060's expert share to `pcie_frac 0.08` against a default of 0.55. Both cards report `pcie.link.gen.current 1` at rest (the idle link-speed saving); the generations above are the negotiated maxima.

- CPU: AMD Ryzen 7 3700X (Zen 2, 8 cores / 16 threads, no AVX-512), **AM4** socket, on an ASUS TUF GAMING B550-PLUS WIFI II
- RAM: 64 GiB DDR4-3000 across four DIMMs, dual-channel, at rated speed (`Speed` and `ConfiguredClockSpeed` both 3000, DOCP on); storage: not measured
- OS: Windows 11 (10.0.26100); GPU driver: 616.92; CUDA 13.0 (from the engine's `BUILD.json`, not read from the driver)
- Strata: project 0.1.43, commit 7695859a; engine 0.1.43
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: swift-1.5-iq3_xxs
- Context: 100000 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0
- Layer split across both cards: layers 0-20 on the 3060 (CUDA0), 21-47 and the head on the 4060 Ti (CUDA1), with `"gpu": [0, 1]` so the slower card is CUDA0. Written by hand, not by the placement search: `layer_split auto` picks an extreme on this pair (it chose a two-layer first stage on 0.1.40.2 and 0.1.41, collapsing decode to ~25 tok/s).

```text
strata.exe --pack swift-iq3_xxs --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 100000 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 512 --vram-reserve-later-mib 1024 --pipeline-windows 2 --trim-stage-weights --resident-experts --layer-split 21
```

Environment of the config: `STRATA_PREFILL_CPU_SHARE=auto`, `STRATA_RESIDENT_PIN=0`

`layer_split` and `gpu` are top-level keys of the install's config JSON, not entries of its `args`; the server is what appends `--layer-split 21` to the engine command line. The block above carries the split flag because that is what the engine really received.

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 46.8 (38.2-59.4) | 0.38 (0.33-0.52) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 508.3 (400.6-523.4) | not measured | 7.95 (7.71-10.09) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 910.7 (850.0-917.8) | not measured | 17.69 (17.54-18.93) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1087.2 (1083.5-1088.6) | not measured | 29.56 (29.50-29.64) |
| agent session, follow-up turns | 14539 (14519-14560) at the last turn | see below | up to 64 | 3 | see below | 56.8 (39.8-68.5) | 6.98 (3.77-10.30) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.7-68.8); earlier-turn tokens read again per session: 75 (74-144) (earlier turns that the cache did not hold; the 13634 (13634-13635) new tokens of the files are read once); first turn (nothing cached) first token: 3.08 s (3.01-3.59).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 661 | 0 | 661 | 185.7 | 3.592 | 34.9 |
| 1 | 3756 | 676 | 3080 | 300.3 | 10.299 | 46.0 |
| 2 | 5289 | 3819 | 1470 | 271.0 | 5.459 | 44.3 |
| 3 | 8733 | 5352 | 3381 | 344.0 | 9.868 | 48.5 |
| 4 | 11657 | 8726 | 2931 | 338.3 | 8.728 | 56.8 |
| 5 | 14560 | 11714 | 2846 | 368.2 | 7.773 | 51.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 336 s.

One configuration note that affects comparability: this run carries 0.1.42's new default `STRATA_ROUTE_TAIL_SKIP=7`, which the release measured at +13-16% decode on CUDA. It changes the answers slightly (KL 0.0058 on an RTX 3060 with IQ3_XXS, top-1 agreement 97.7%), so these numbers are not directly comparable with a 0.1.41 report of the same hardware with the switch off.
