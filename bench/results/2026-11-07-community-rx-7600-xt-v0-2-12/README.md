# Community benchmark on AMD Radeon RX 7600 XT 16 GB (gfx1102), CachyOS Linux, Ryzen 5 3600

Measured on 2026-11-07 by [DamirLukina88](https://github.com/DamirLukina88) with `tools/strata_bench.py` (suite 1, 3 runs per test per leg, medians with min-max ranges).
Tested **Qwen3.8-Flash-Next-Coder-IQ1_M** at 65,536-token context on a single **AMD Radeon RX 7600 XT 16 GB (`gfx1102`)** across a **4-leg interleaved A/B benchmark** (Leg 1 Base $\to$ Leg 2 Candidate $\to$ Leg 3 Base $\to$ Leg 4 Candidate) under `systemd --user` with `MemorySwapMax=0` (0 B engine swap).

**Main limitations:** Speed-only suite (no needle or coding correctness check in this run); measured on a modified local fork build (`0.1.44.1` base at commit `37f6ceea` on `DamirLukina88/D.L.-AI---Strata`, not an unmodified official `v0.2.12` release binary) with experimental speed projection (`cvec: project:4-44`) enabled; single desktop machine in normal use without GPU clock or power capture.

## Hardware and software

- **GPU:** AMD Radeon RX 7600 XT, 16 GiB VRAM (`gfx1102`; engine reports `gpu_vram_mib: 16368.0`, `vram_free_mib: 3500` at load), stock settings, power limit and clocks not recorded.
- **CPU / RAM:** AMD Ryzen 5 3600 6-Core Processor (6 cores, 12 threads, AVX2, `pool_workers: 5`); 31.3 GiB usable RAM (2x 16 GiB DDR4-3200).
- **Storage / PCIe:** NVMe SSD; engine probe `pcie_frac: 0.19`.
- **OS / Driver:** Linux 7.3.1-1-cachyos (`x86_64`), ROCm 7.2.4 HIP runtime (`gfx1102`), `hipBLASLt` and RDNA3 WMMA matrix cores active (`STRATA_HIP_WMMA=1`).
- **Engine Provenance:** Native HIP source build (`build-hip/strata`) from fork commit `37f6ceea` (`0.1.44.1` base):
  - **Base (Legs 1 & 3):** `0.1.44.1` with cherry-picked fixes through `v0.2.11`.
  - **Candidate (Legs 2 & 4):** Base plus `v0.2.12` correctness guards: MMQ row table margin (`#5108`, `mmq_ids_table_rows`), Adaptive PCIe overlap Flag-B ordering (`#5096`, `raise_flag_b_ordered`), adaptive range header (`#4844`), and `serve/` parser 256-char tail lookback (`#5409`) / MCP suffix resolution (`#5281`).
- **System Memory Policy:** Managed under `systemd --user` with `MemorySwapMax=0` to prevent CachyOS (`vm.swappiness=150`) from paging the 20.7 GiB resident expert arena (`arena_mib: 20717`) into zram.

## Model and configuration

- **Model and quantization:** `qwen3.8-flash-next-coder-iq1_m` (`Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf` + `00002-of-00002.gguf`, 256 resident MoE experts in RAM).
- **Context & KV:** 65,536 tokens (`--max-context 65536`), `--kv int8` (`kv_resident: 0`), `--session-state-codec bf16`.
- **Expert & Conversation Cache:** `--expert-cache auto` (`expert_slots: 3401`, `expert_cache_mib: 6663`, `--vram-reserve-mib 3584`), `--conversation-cache-slots 4` (`conversation_cache_min_free_mib: 2560`), `--prompt-cache 12 --prompt-cache-tail --prompt-cache-every 8192`.
- **Speculation & Control Vector:** `--spec 4 --spec-min-p 0.5 --mtp rt` (`engine_info`: `spec: 6, mtp_max: 4, lookup: 3`), experimental speed projection `--control-vector-scaled Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer`.
- **Benchmark Sampling:** Thinking off for the suite (`--think` false); temperature 0.

```text
strata --pack coder-iq1_m \
  --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
  --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf \
  --expert-profile expert-profile-custom.bin \
  --expert-cache auto \
  --prefill auto \
  --prompt-cache 12 \
  --prompt-cache-tail \
  --prompt-cache-every 8192 \
  --spec 4 \
  --spec-min-p 0.5 \
  --mtp rt \
  --max-context 65536 \
  --kv int8 \
  --resident-experts \
  --vram-reserve-mib 3584 \
  --control-vector-scaled Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 \
  --control-vector-layer-range 4 44 \
  --cvec-mode project \
  --cvec-dir per-layer \
  --session-state-codec bf16
```

Environment variables configured (`env` in config):
`STRATA_RESIDENT_PIN=0`, `STRATA_PREFILL_MMQ=1`, `STRATA_PREFILL_RING=96`, `STRATA_IO_THREADS=32`, `STRATA_PREFILL_CPU_SHARE=auto`, `STRATA_ADAPT_OVERLAP=1`, `STRATA_ADAPT_OVERLAP_RESIDENT=1`, `STRATA_DROP_CACHE_ON_EXIT=1`, `STRATA_ROUTE_TAIL_SKIP=7`, `STRATA_ROUTE_RESIDENT=0.5`, `STRATA_HEAD_MIX_MULTI=1`, `STRATA_CVEC_FUSE=1`, `STRATA_GDN_NOY=1`, `STRATA_GDN_CONVL2=1`, `STRATA_EMB_REUSE_ACCOUNT=1`, `STRATA_AUX_CPUS=auto`, `STRATA_SH_STREAM=1`, `STRATA_LFUSE=1`, `STRATA_TSUM=1`, `STRATA_QFUSE=1`, `STRATA_PLE_BATCH=1`, `STRATA_GDN_HEAD=1`, `STRATA_GDN_PP=2`, `STRATA_HIP_WMMA=1`, `STRATA_SELECT_SIMT=1`, `STRATA_COMMIT_BATCH=0`, `STRATA_HC_Q8=1`, `STRATA_PREFILL_WARM=1`.

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a running Strata server across 4 interleaved legs (Base 1 $\to$ Candidate 1 $\to$ Base 2 $\to$ Candidate 2), restarting the service cleanly between binary swaps:
- **short chat:** the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs (9 requests per leg).
- **4K / 16K prompts:** generated word lists sized by a calibration request, 24 new tokens cap, 3 fresh prompts each per leg.
- **agent session:** system prompt and four tools, then 5 turns that each add ~1,500 words of a project file and continue from the model's earlier answers, up to 64 new tokens per turn, 3 sessions per leg.

## Results

### 1. Per-Leg Results (Medians and Min-Max Ranges)

| Leg & Raw JSON | Workload | Actual Prompt Tokens | Reused Tokens | Generated Tokens | Runs | Prompt tok/s (median & range) | Decode tok/s (median & range) | TTFT s (median & range) |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| **Leg 1 (Base)** `results-leg1-base.json` | short chat | 88 (85-93) | 0 | 100 (100-100) | 9 | not measured | 26.3 (22.1-34.9) | 1.56 (1.49-1.60) |
| **Leg 1 (Base)** `results-leg1-base.json` | 4K prompt, fresh | 4021 (4021-4021) | 0 (0-0) | 24 cap | 3 | 307.0 (298.6-308.2) | not measured | 13.14 (13.09-13.52) |
| **Leg 1 (Base)** `results-leg1-base.json` | 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 333.3 (333.1-333.7) | not measured | 48.22 (48.17-48.26) |
| **Leg 1 (Base)** `results-leg1-base.json` | agent follow-up turns | 13403 (13402-13449) | see below | up to 64 | 3 | see below | 28.8 (23.8-32.7) | 11.32 (7.06-12.18) |
| **Leg 2 (Cand)** `results.json` | short chat | 88 (85-93) | 0 | 100 (100-100) | 9 | not measured | 24.4 (21.6-35.6) | 1.59 (1.52-1.68) |
| **Leg 2 (Cand)** `results.json` | 4K prompt, fresh | 4021 (4021-4021) | 0 (0-0) | 24 cap | 3 | 305.8 (298.3-307.0) | not measured | 13.19 (13.14-13.53) |
| **Leg 2 (Cand)** `results.json` | 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 334.6 (334.5-335.3) | not measured | 48.04 (47.95-48.06) |
| **Leg 2 (Cand)** `results.json` | agent follow-up turns | 13426 (13394-13465) | see below | up to 64 | 3 | see below | 28.0 (23.4-33.7) | 11.58 (7.22-12.15) |
| **Leg 3 (Base)** `results-leg3-base.json` | short chat | 88 (85-93) | 0 | 100 (100-100) | 9 | not measured | 25.0 (19.2-33.9) | 1.66 (1.61-1.74) |
| **Leg 3 (Base)** `results-leg3-base.json` | 4K prompt, fresh | 4021 (4021-4021) | 0 (0-0) | 24 cap | 3 | 303.0 (294.9-304.0) | not measured | 13.31 (13.27-13.68) |
| **Leg 3 (Base)** `results-leg3-base.json` | 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 334.7 (331.6-335.4) | not measured | 48.02 (47.93-48.48) |
| **Leg 3 (Base)** `results-leg3-base.json` | agent follow-up turns | 13391 (13380-13450) | see below | up to 64 | 3 | see below | 27.6 (22.0-35.1) | 11.49 (7.49-12.92) |
| **Leg 4 (Cand)** `results-leg4-cand.json` | short chat | 88 (85-93) | 0 | 100 (100-100) | 9 | not measured | 23.9 (20.6-34.1) | 1.65 (1.57-1.86) |
| **Leg 4 (Cand)** `results-leg4-cand.json` | 4K prompt, fresh | 4021 (4021-4021) | 0 (0-0) | 24 cap | 3 | 306.8 (300.2-307.8) | not measured | 13.15 (13.11-13.44) |
| **Leg 4 (Cand)** `results-leg4-cand.json` | 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 331.7 (326.4-333.7) | not measured | 48.47 (48.17-49.26) |
| **Leg 4 (Cand)** `results-leg4-cand.json` | agent follow-up turns | 13428 (13380-13438) | see below | up to 64 | 3 | see below | 28.0 (23.6-31.7) | 11.74 (7.33-12.76) |

### 2. Paired A/B Comparison Summary (Mean of Leg Medians)

| Metric | Base (Legs 1 & 3) | Candidate (Legs 2 & 4) | Delta |
| :--- | :---: | :---: | :---: |
| **4K Prompt Prefill** | 305.00 tok/s | **306.30 tok/s** | +0.4% |
| **4K Prompt TTFT** | 13.23 s | **13.17 s** | -0.4% |
| **16K Prompt Prefill** | 334.00 tok/s | **333.15 tok/s** | -0.3% |
| **Short Chat Decode** | 25.65 tok/s | **24.15 tok/s** | -5.8% (within fan/thermal noise) |
| **Agent Multi-Turn Decode** | 28.20 tok/s | **28.00 tok/s** | -0.7% |
| **Agent Cache Hit %** | 67.07% | **67.21%** | +0.2% |
| **Engine Swap (`VmSwap`)** | 0 B | **0 B** | 0 B |

### 3. Agent Session Conversation Cache (Leg 2 Candidate `results.json`, Run 1)

- **Cache hit over follow-up turns:** 67.2% (67.2-67.3% across Leg 2; 67.1-67.3% across Leg 4).
- **Earlier-turn tokens read again per session:** 5 (5-5) across all 4 legs.
- **First turn (cold cache) TTFT:** 4.48 s (4.45-4.55 s in Leg 2; 4.43-4.53 s in Leg 4).

| Turn (Leg 2, run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 647 | 0 | 647 | 143.5 | 4.548 | 20.2 |
| 1 | 2564 | 668 | 1896 | 215.3 | 8.850 | 23.4 |
| 2 | 4129 | 2627 | 1502 | 199.5 | 7.572 | 27.4 |
| 3 | 7573 | 4192 | 3381 | 279.3 | 12.153 | 26.9 |
| 4 | 10499 | 7636 | 2863 | 240.9 | 11.930 | 31.6 |
| 5 | 13426 | 10527 | 2899 | 246.0 | 11.834 | 33.7 |

Per-request raw data files included in this folder:
- `results.json` (Leg 2 Candidate)
- `results-leg1-base.json` (Leg 1 Base)
- `results-leg3-base.json` (Leg 3 Base)
- `results-leg4-cand.json` (Leg 4 Candidate)
