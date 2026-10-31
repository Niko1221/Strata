# Community benchmark: Swift (uncensored, huihui-ai) IQ3_XXS (engine 0.1.50)

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

- **Model:** Swift (uncensored, huihui-ai) IQ3_XXS, GSQ-RCO dynamic experts
- **Context:** 262,144; **KV int8**  with --kv-resident 32768
- **PLE:** `ple-fp8.gguf`; expert cache `auto`, prefill `auto`
- **Drafting:** MTP (`--spec 4 --mtp rt`)
- **Sampling:** greedy, `temperature 0`
- Auto-calibrated on this engine with `setup.py --calibrate` (tuned `--pcie-frac`,
  `--spec-min-p`, `--adapt-*`).

```text
strata.exe --pack swift-iq3_xxs --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf ple-fp8.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vram-reserve-mib 1000 --pcie-frac 0.00 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

## Results

| Metric | Value |
| --- | ---: |
| Chat decode, tok/s (median) | **109.8** |
| Chat TTFT, s | 0.320 |
| 4K prompt, tok/s | 2413.1 (TTFT 1.70 s) |
| 16K prompt, tok/s | 3677.5 (TTFT 4.40 s) |
| Agent follow-up decode, tok/s | **112.3** |
| Agent turn TTFT, s | 1.66 |

Agent-cache row (per-session median over 3 sessions): cache hit 70.0%,
re-read 74 tok/session, 14203 new tok/session.

## Recall checks

Needle = a code word (`meadow-copper-504`) buried in a long haystack; 6/6 correct.

| Context | Depth | Prompt tokens | Found | Seconds |
| ---: | ---: | ---: | --- | ---: |
| 32k | 10% | 31640 | yes | 9.5 |
| 32k | 50% | 31640 | yes | 0.3 |
| 32k | 90% | 31641 | yes | 9.3 |
| 256k | 10% | 248082 | yes | 76.3 |
| 256k | 50% | 248080 | yes | 76.3 |
| 256k | 90% | 248080 | yes | 76.3 |

Raw runs: `needles.json`

## Notes

- Read the chat/agent decode as a range, not a point: short-chat decode swings roughly
  +/-30% run to run even warm; the table gives the median of the representative arm.
- Engine 0.1.50 (SDK async-tier auto-measure). The launch string above is the exact
  calibrated configuration, engine version recorded honestly as 0.1.50 (Strata v0.1.49.2).
