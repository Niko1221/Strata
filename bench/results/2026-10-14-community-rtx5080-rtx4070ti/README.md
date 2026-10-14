# 2026-10-14: RTX 5080 16 GB + RTX 4070 Ti 12 GB (unequal pair, both orders), Strata v0.1.42

Community benchmark report for `docs/TEST_REQUESTS.md`'s unequal dual-NVIDIA request:
the same pair measured in **both card orders** plus a helper-expert-cache arm, one single-card
control, and one recorded crash condition. All numbers are medians of 3 runs per tier
(decode arm: 5 runs), client wall time over `usage` tokens, per-run rows in `data/`.

## Hardware

- **GPUs**: NVIDIA RTX 5080 16 GB (sm_120; runs the shipped build's sm_89 PTX) + NVIDIA RTX 4070 Ti 12 GB (sm_89). PCIe link probed by the engine at 6.4 GB/s (best of 4). Driver 617.42 (2026-10). Both cards share the desktop (Windows 11 Home, WDDM; the 5080 is the primary display adapter — ~1.2 GB desktop tax).
- **CPU / RAM**: Intel Core i7-13700, 63.8 GB RAM.
- **Model**: Qwen3.8-Flash-Next-GSQ-RCO-**abliterated** IQ3_XXS, 2-shard GGUF + community `sc117` pack, 262,144-token context. `--kv int8`, `--kv-resident 32768`, `--spec 3 --spec-min-p 0.5` with the `mtp/rt` draft, `--expert-cache auto`, `--prefill auto`, `--vram-reserve-mib 1024`, `--vision` (encoder pinned to the 4070 Ti via `vision.cuda_device: 1`).
- **Engine**: v0.1.42 release build (`strata-windows-x64-cuda12.zip`, BUILD.json: CUDA 12.9, archs 60–89 + PTX). Windows. Thinking disabled engine-side (shared settings `reasoning_effort: none`); temperature 0.

## Method

Prompt tiers ~8K / ~70K / ~240K real `prompt_tokens` (CJK filler), 3 fresh runs per tier with a
per-run unique prefix salt (defeats the prompt cache), warmup request excluded. Decode arm: 5 runs
of a 512-token generation from a short prompt; decode rate = `completion_tokens / wall`. Arms were
run sequentially on a quiet machine (one engine restart per arm).

## Results (medians, tok/s)

| Arm | 8K prefill | 70K prefill | 240K prefill | decode512 |
|---|---|---|---|---|
| **A: single RTX 5080** | 2,613 | 3,798 | 3,549 | **85.0** |
| B: layer split, 5080 first (`layers 0-34 / 35-47`) | 1,120 | 2,333 | 2,525 | 75.4 |
| C: layer split, 4070 Ti first (`layers 0-13 / 14-47`) | 1,367 | 2,775 | not run (see crash) | 73.7 |
| D: 5080 + helper expert cache on 4070 Ti (3,306 experts, `--remote-expert-opt`) | 1,273 | 2,649 | 2,593 | 23.6 (spread 16.6–48.5) |

Per-tier spreads were ≤3% except D's decode (unstable, 16.6–48.5 tok/s across 5 runs) and the
decode arm's first run after load (cold) — medians above include it in A/B/C, exclusion noted per
row in `data/`.

## Observations

1. **Single 5080 wins every tier.** For this pairing every dual-GPU mode is a net loss: layer
   split in both orders (auto-split produced 35/13 and 13/35 layer splits) and the helper expert
   cache. The 5080's own expert cache already reached 92–97% decode hit rate, so helper rows
   mostly add PCIe detours. This pair's asymmetry (84 vs 60 SMs, one card also driving the
   desktop) may be what `#2052` describes; the auto-split balance at `v0.1.40.2` did not help here.
2. **Helper cache arm (D)**: engine log confirms the second card took 3,306 experts (5.31 GiB,
   `results return through pinned host rows`). Decode collapsed vs control (23.6 vs 85.0 median)
   with high spread — worth a look whether `--remote-expert-opt`'s complementary-cache assumptions
   break when the primary cache is already near-saturating.
3. **Crash condition (repro, not a number)**: arm B at the 240K tier, third consecutive
   237,976-token prompt: **hard kernel crash** (Kernel-Power 41; dump write failed — volmgr 161
   `BugCheckProgress 0x00020020`, no minidump). Last engine lines show normal KV streaming
   (82–84% VRAM hits) and cross-PCIe expert reads. The same 240K ×3 workload is stable on the
   single-card arm. Hypothesis worth checking: this build runs sm_120 through PTX JIT (BUILD.json
   archs cap at 89) and the layer-split + long-context KV-streaming path stresses JIT'd kernels
   in a way single-card does not. Windows 11, WDDM, driver 617.42; `CrashDumpEnabled` was set to
   kernel dump afterwards, so a repeat should produce a dump.
4. **P2P**: `cuDeviceCanAccessPeer` = 0 in both directions (WDDM GeForce) — any design that needs
   peer-mapped memory (e.g. a KV cold tier on a second card) is not testable on Windows GeForce;
   helper-cache-style pinned-row staging is the only cross-card route here.

## Reproduce

Data rows (per-run JSONL, client times + `usage` tokens) in `data/`. Engine config arms differ
only in: `--layer-split auto --split-device 1 --vram-reserve-later-mib 512` (B/C; card order via
`CUDA_VISIBLE_DEVICES=1,0` vs `0,1`) and `--expert-cache-device1 5000 --remote-expert-opt` (D).
