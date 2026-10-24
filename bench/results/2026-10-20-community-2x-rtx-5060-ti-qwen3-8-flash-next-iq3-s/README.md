# Community benchmark: 2x RTX 5060 Ti 16 GB, Threadripper 3970X (TRX40, PCIe gen4), Flash-Next IQ3_S (0.2.1)

Measured on 2026-10-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run. An earlier suite on the same box with engine 0.1.44 (2026-10-17) is included below as the version comparison; the two runs share the hardware, the model, and most of the configuration, so only the engine version and the calibration results differ between them.

## Hardware and software

- GPUs: 2x NVIDIA GeForce RTX 5060 Ti 16 GB (16,311 MiB each), 180 W stock power limit, no overclock. PCIe **gen4, x8/x8** (max gen4, current gen4 under load per `nvidia-smi -q`).
- CPU: AMD Ryzen Threadripper 3970X, 32 cores / 64 threads, **AVX2 only (no AVX-512)**; the engine reported it runs the expert kernels on AVX2.
- Motherboard: ASRock Rack TRX40D8-2N2T (the AST2500 BMC runs the fan control; headless server).
- RAM: 128 GB installed, 4x 32 GB DDR4-2666, all four channels populated (quad-channel), 125.7 GiB MemTotal.
- OS: Ubuntu Server 24.04 (headless), kernel 6.8.0-147-generic, NVIDIA driver 595.99.02, CUDA 13.2 (toolkit 13.0 for the build).
- Storage: not measured. Background workloads: none at bench time; power limits: stock, not recorded per run.

## Model and configuration

- Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, **IQ3_S** (two GGUF shards), vision encoder `mmproj-...BF16.gguf` (not used by the suite).
- Context: 262,144 tokens (native); KV `int8` with `--kv-resident 32768` (KV streaming: 32,768 cells per QSA layer in VRAM, the full K/V in 3.09 GiB of pinned RAM) and `--ple-io ram` (the n-gram table read from RAM, which setup enabled automatically at 126 GB RAM).
- Layer split fixed at **25** across both GPUs; `--pipeline-windows 2` (the two stages overlap across verify windows).
- Expert cache `auto`; **calibrated fresh on 0.2.1** (`setup.sh --calibrate`, both GPUs): it chose **`--pcie-frac 0.00`** — with the 0.2.x file tier the CPU pool computes the missed experts faster than a PCIe copy, where the same calibration on 0.1.44 had chosen 0.20 — and **`--spec-min-p 0.50`** (was 0.70).
- **Deviations from the usual community setup, for transparency:**
  - The experimental speed projection (control vector, layers 4-44) is ON in all runs. The community reports do not use it; its cost is roughly 0.2-0.4 %/token.
  - `--prefill auto:32768` was measured **worse** on this two-card box on 0.1.44 (16K fresh 1531 vs 1895 tok/s; the big chunk borrows the second card's expert-cache slots) and is not set. The +21-35 % community gains for it apply to single-GPU setups.
  - `"parallel": 1` (one request at a time). The cost of `parallel: 2` on these cards was measured earlier on 0.1.44: 0.51 GiB of VRAM per slot per card, expert caches 5106/4393 -> 4528/3883 slots (-11 % per card).

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --ple-io ram --kv-resident 32768 --vision --control-vector-scaled Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --vram-reserve-later-mib 1123 --vram-reserve-mib 832 --layer-split 25 --pipeline-windows 2 --pcie-frac 0.00 --spec-min-p 0.50
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 83.1 (80.2-110.1) | 0.25 (0.23-0.27) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1229.8 (1228.3-1230.1) | not measured | 3.29 (3.29-3.30) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2033.9 (2031.6-2041.8) | not measured | 7.93 (7.90-7.94) |
| agent session, follow-up turns | 14639 (14633-14700) at the last turn | see below | up to 64 | 3 | see below | 85.0 (65.5-98.9) | 3.39 (3.00-3.58) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

## 0.2.1 against 0.1.44 (same box, same method)

| | 0.1.44 (2026-10-17) | 0.2.1 (this run) | delta |
| --- | --- | --- | --- |
| short chat decode | 93.6 (81.0-129.4) | 83.1 (80.2-110.1) | medians differ, ranges overlap heavily; short-chat decode on this box swings 2x between runs |
| 4K prompt | 1185.1 | 1229.8 | +3.8 % |
| 16K prompt | 2001.9 | 2033.9 | +1.6 % |
| agent follow-up decode | 90.4 (78.2-109.5) | 85.0 (65.5-98.9) | within the run-to-run spread |
| TTFT | 0.30 / 3.42 / 8.05 s | 0.25 / 3.29 / 7.93 s | slightly better everywhere |

The large sm_120 decode gains the 0.2.x release notes report (+13.8 %) were measured with every expert in VRAM; on this box the decode misses run on the CPU pool, and the measured decode is unchanged within noise. The prompt gains match `--ple-io ram` more than engine work. Both runs carry the ESP control vector, so this comparison is internal to this box, not to the community defaults.

## Comparison with the other 2x RTX 5060 Ti reports

| Report | Quant | Engine | PCIe / CPU | Decode (4K / 32K) | Prompt (4K) |
| --- | --- | --- | --- | --- | --- |
| 2026-10-02 (Efs-O) | IQ3_S | 0.1.34 | gen3 x8/x8, i9-9900KF | 62.9 short / 56.4 @20.9K | 1108 @20.9K |
| 2026-10-05 rerun (R6DJO) | IQ3_XXS | 0.1.39 | gen3 (X99), Xeon E5-2690 v4 | 84.8 / 86.5 | 911 |
| **this report** | **IQ3_S** | **0.1.46.2 (project 0.2.1)** | **gen4 x8/x8, 3970X** | **83.1 short** | **1230 @4K, 2034 @16K** |

All three changed engine, PCIe and CPU relative to each other; the numbers are not attributable to a single factor. What this report adds: IQ3_S on two 5060 Ti reaches the decode level of the IQ3_XXS rerun while reading prompts about twice as fast, when the cards get gen4 and a strong AVX2 CPU pool.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi` and the OS; background load and power limits are not recorded per run. The one-GPU `--calibrate` result (0.20/0.70) that preceded the two-GPU recalibration is invalid for this report and superseded by it.
