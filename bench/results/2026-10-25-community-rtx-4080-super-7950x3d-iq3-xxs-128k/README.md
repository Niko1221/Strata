# Community benchmark: Qwen3.8-Flash-Next IQ3_XXS (engine 0.1.50)

Measured on 2026-10-25 by [@c-OOL-io](https://github.com/c-OOL-io). Standard
[strata_bench](BENCHMARK_PROTOCOL) battery on a single RTX 4080 SUPER desktop with a
Ryzen 7950X3D (Zen 4, AVX-512), 64 GB RAM. One representative arm of three, reported as
the median of the suite's internal runs.

## Hardware and software

- **GPU:** NVIDIA GeForce RTX 4080 SUPER, 16,376 MiB usable VRAM
- **CPU:** AMD Ryzen 9 7950X3D, 16 cores / 32 threads (Zen 4, AVX-512)
- **RAM:** 64 GB
- **OS:** Windows 11 (10.0.26200)
- **Strata:** v0.1.49.2 (commit 980d39f4), **engine 0.1.50**, prebuilt CUDA 13 package
- **Vision:** on (`--vision`), not exercised by these runs

## Model and configuration

- **Model:** Qwen3.8-Flash-Next IQ3_XXS, GSQ-RCO dynamic experts
- **Context:** 131,072; **KV int8**  with --kv-resident 32768
- **PLE:** `ple-fp8.gguf`; expert cache `auto`, prefill `auto`
- **Drafting:** MTP (`--spec 4 --mtp rt`)
- **Sampling:** greedy, `temperature 0`
- Auto-calibrated on this engine with `setup.py --calibrate` (tuned `--pcie-frac`,
  `--spec-min-p`, `--adapt-*`).

```text
strata.exe --pack iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf ple-fp8.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --pcie-frac 0.35 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

## Results

| Metric | Value |
| --- | ---: |
| Chat decode, tok/s (median) | **102.1** |
| Chat TTFT, s | 0.346 |
| 4K prompt, tok/s | 2385.4 (TTFT 1.72 s) |
| 16K prompt, tok/s | 3564.2 (TTFT 4.54 s) |
| Agent follow-up decode, tok/s | **107.8** |
| Agent turn TTFT, s | 1.69 |

Agent-cache row (per-session median over 3 sessions): cache hit 70.1%,
re-read 4 tok/session, 14201 new tok/session.

## Recall checks

Needle = a code word (`meadow-copper-504`) buried in a long haystack; 6/6 correct.

| Context | Depth | Prompt tokens | Found | Seconds |
| ---: | ---: | ---: | --- | ---: |
| 32k | 10% | 31640 | yes | 10.8 |
| 32k | 50% | 31640 | yes | 9.4 |
| 32k | 90% | 31641 | yes | 9.4 |
| 128k | 10% | 126287 | yes | 38.8 |
| 128k | 50% | 126285 | yes | 34.1 |
| 128k | 90% | 126285 | yes | 19.8 |

Raw runs: `needles.json`

## Notes

- Read the chat/agent decode as a range, not a point: short-chat decode swings roughly
  +/-30% run to run even warm; the table gives the median of the representative arm.
- Engine 0.1.50 (SDK async-tier auto-measure). The launch string above is the exact
  calibrated configuration, engine version recorded honestly as 0.1.50 (Strata v0.1.49.2).
