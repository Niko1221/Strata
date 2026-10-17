# Community benchmark: 2x RTX 5060 Ti 16 GB, Threadripper 3970X (TRX40, PCIe gen4), Flash-Next IQ3_S

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

This is the first report of Flash-Next **IQ3_S** (the same quantization as the 2026-10-02 2x5060 Ti report) on two RTX 5060 Ti with a modern engine (0.1.44), PCIe gen4 and a high-thread-count CPU. Compared to the 2026-10-02 report (56-63 tok/s decode on 0.1.34, PCIe gen3, 8-core i9-9900KF), this machine reaches 83-94 tok/s decode and 1185/2002 tok/s prompt at 4K/16K. Engine version, PCIe generation and CPU all changed between the two reports, so the difference is not attributable to any single change.

## Hardware and software

- GPUs: 2x NVIDIA GeForce RTX 5060 Ti 16 GB (16,311 MiB each), 180 W stock power limit, no overclock. PCIe **gen4, x8/x8** (max gen4, current gen4 under load per `nvidia-smi -q`).
- CPU: AMD Ryzen Threadripper 3970X, 32 cores / 64 threads, **AVX2 only (no AVX-512)**; the engine reported it runs the expert kernels on AVX2.
- Motherboard: ASRock Rack TRX40D8-2N2T (the AST2500 BMC runs the fan control; headless server).
- RAM: 128 GB installed, 4x 32 GB DDR4-2666, all four channels populated (quad-channel), 125.7 GiB MemTotal.
- OS: Ubuntu Server 24.04 (headless), kernel 6.8.0-147-generic, NVIDIA driver 595.99.02, CUDA 13.2.
- Storage: not measured. Background workloads: none at bench time (the server itself was serving); power limits: stock, not recorded per run.

## Model and configuration

- Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, **IQ3_S** (two GGUF shards), vision encoder `mmproj-...BF16.gguf` (not used by the suite).
- Context: 262,144 tokens (native); KV `int8` with `--kv-resident 32768` (KV streaming: 32,768 cells per QSA layer in VRAM, the full K/V of a 262K context in 1.55 GiB of pinned RAM).
- Expert cache `auto` with the shipped expert profile; `--pcie-frac 0.20` (calibrated on this box: each card holds the experts of its own layers locally, so far less is reloaded over PCIe than on a single-card setup).
- Layer split fixed at **25** (`--layer-split 25`): a small A/B on this box measured split 25 marginally ahead of split 20 (decode 87.3 vs 83.5 tok/s median, prompt 1209 vs 1190 at 4K; same engine, same day, 3 runs each).
- **Deviations from the usual community setup, for transparency:**
  - The experimental speed projection (control vector, layers 4-44) is ON in all runs below. The community reports do not use it; its cost is roughly 0.2-0.4 %/token.
  - `--prefill auto:32768` (the large chunk recommended for single-GPU setups) was measured **worse** on this two-card box: 16K fresh prompt 1531 tok/s vs 1895 with plain `auto`, with the engine warning that the big chunk borrows thousands of expert-cache slots of the second card. The +21-35 % gains community reports find for `auto:32768` apply to single-GPU setups, not to a layer split like this one.
  - After this bench, `"parallel": 2` was added to the config for two concurrent clients. It does not affect the solo path these numbers describe, and its cost is measured: each batch slot session takes **0.51 GiB of VRAM per card** (2 slots = 1.03 GiB per card), which shrinks the expert caches from 5106/4393 slots (9.19/8.85 GiB) to 4528/3883 (8.14/7.82 GiB), about -11 % per card.

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 832 --pcie-frac 0.20 --spec-min-p 0.70 --control-vector-scaled Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --layer-split 25 --pipeline-windows 2 --vram-reserve-later-mib 1123
```

(`"parallel": 2` sits at the top level of the server config, not in the engine args; it was added after this run.)

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 93.6 (81.0-129.4) | 0.30 (0.27-0.33) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1185.1 (1170.4-1185.9) | not measured | 3.42 (3.41-3.46) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2001.9 (2001.1-2004.9) | not measured | 8.05 (8.04-8.06) |
| agent session, follow-up turns | 14593 (14567-14603) at the last turn | see below | up to 64 | 3 | see below | 90.4 (78.2-109.5) | 3.55 (3.12-3.76) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.5-68.8); earlier-turn tokens read again per session: 5 (3-143) (earlier turns that the cache did not hold; the 13730 (13729-13731) new tokens of the files are read once); first turn (nothing cached) first token: 2.32 s (2.26-2.33).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 272.3 | 2.258 | 78.9 |
| 1 | 3755 | 633 | 3122 | 855.9 | 3.673 | 90.4 |
| 2 | 5288 | 3748 | 1540 | 493.4 | 3.148 | 92.9 |
| 3 | 8732 | 5281 | 3451 | 925.8 | 3.758 | 90.1 |
| 4 | 11641 | 8779 | 2862 | 801.1 | 3.605 | 90.5 |
| 5 | 14593 | 11696 | 2897 | 816.7 | 3.582 | 87.5 |

Per-request data: `results.json`.

## Comparison with the other 2x RTX 5060 Ti reports

| Report | Quant | Engine | PCIe / CPU | Decode (4K / 32K) | Prompt (4K) |
| --- | --- | --- | --- | --- | --- |
| 2026-10-02 (Efs-O) | IQ3_S | 0.1.34 | gen3 x8/x8, i9-9900KF | 62.9 short / 56.4 @20.9K | 1108 @20.9K |
| 2026-10-05 rerun (R6DJO) | IQ3_XXS | 0.1.39 | gen3 (X99), Xeon E5-2690 v4 | 84.8 / 86.5 | 911 |
| **this report** | **IQ3_S** | **0.1.44** | **gen4 x8/x8, 3970X** | **93.6 short** | **1185 @4K, 2002 @16K** |

All three changed engine, PCIe and CPU relative to each other; the numbers are not attributable to a single factor. What this report adds: IQ3_S on two 5060 Ti reaches the decode level of the IQ3_XXS rerun when the cards get gen4 and a strong AVX2 CPU pool.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi` and the OS; background load and power limits are not recorded per run. Total time of the suite: 123 s.
