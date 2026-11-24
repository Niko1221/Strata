# Changelog

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses `v<upstream-strata>-ex.<N>` versioning (newest release first).

## [v0.2.20-ex.3] — 2026-11-24

Low-memory speed milestone targeting **VRAM ≤ 11.5 GiB** and **effective RAM ≤ 31 GiB** (leaving ~1 GiB OS headroom on 32 GiB headless Linux / 12 GiB GPU systems) with no quality loss. On RTX 5070 Ti (Qwen3.8-Flash-Next `IQ3_XXS`), partial-RAM decode rises from 53.0 to **77.6 tok/s median (80.0 tok/s peak, +46.4%)** at **11.45 GiB VRAM** and **~30.9 GiB effective RAM**.

### Added

- **In-place copy-kernel VRAM↔RAM swap (`exclusive_cache.hpp`, `fetch_blobs_kernel`)**: When `STRATA_ADAPT_FETCH_RESIDENT=1` admits a RAM-resident expert into VRAM and evicts a file-backed victim, `fetch_blobs_kernel` reads both sides and writes the evicted VRAM slot back into the promoted expert's RAM slot in place—0 SSD reads, 0 extra RAM, 100% bit-identical weights. Cuts cold-tier file reads by **−67.2%** (31,333 → 10,266) and lifts decode from 53.0 → 70.4 tok/s.
- **OS-independent zero-copy cold expert mapping (`STRATA_COLD_ZC_ALL=1`)**: Enables `zc_roles` direct reads of `gate`/`up`/`down` tensors from the GGUF memory mapping on Windows (where Linux's `cached_bytes()` residency query is unavailable), eliminating the single-threaded 10.88 ms/round staging `memcpy` (`dispatch jobs` 10.88 ms → 0.05 ms) with 100% bit-identical output. Lifts decode from 70.4 → 75.8 tok/s.
- **D2H swap threshold tuning (`STRATA_ADAPT_FETCH_MIN=2.0–3.0`, `STRATA_ADAPT_FETCH_MARGIN=1.0–1.5`)**: Reduces unnecessary bidirectional PCIe swaps in `fetch_blobs_kernel`, lifting decode from 75.8 → **77.6 tok/s median (80.0 tok/s peak)**.

### Changed

- Updated `STRATA_VERSION` to `0.2.20-ex.3`.
- Updated `README.md` `Results → Memory` table and `What exstrata changes` with `ex.3` numbers (`VRAM 11,725 MiB / −17.3%`, `RAM ~31,640 MiB / −55.8%`, `VRAM + RAM −49.4%`).

### Verified

- All `ex.3` additions are bit-identical in weight bytes and arithmetic (0 quantization added by exstrata).
- Optional `--kv k8v4` saves 90 MiB VRAM (Key INT8, Value rotated Q4_0); `--kv int8 --expert-cache 2760` achieves the same 11.45 GiB VRAM with zero KV quantization change.

## [v0.2.20-ex.2] — 2026-11-24

Rebased onto upstream **Strata v0.2.20** and added opt-in partial-RAM adaptive tiers for ≤ 12 GiB VRAM / ≤ 32 GiB RAM systems.

### Added

- **`STRATA_ADAPT_FETCH_RESIDENT=1`**: Re-enables `STRATA_ADAPT_FETCH=2` adaptive VRAM admission for RAM-resident experts in `--resident-experts --resident-budget-gib N` mode.
- **`STRATA_RAM_ADAPT_COLD=1`**: Dynamic RAM tier promotion between decode windows (`adapt()`), promoting frequently routed cold-tier experts into the fixed pinned RAM buffer by evicting never/rarely-used RAM-resident experts (0 extra RAM, bit-identical).

### Fixed

- **Windows `--native` + `--resident-experts` startup crash**: Skipped the unbuffered startup view-unmap path (`begin_startup_unbuffered`) when reading native GGUF shards in place (`src.transient(0, 0)`), preventing crashes on Strata v0.2.20.

### Changed

- Rebased onto upstream Strata v0.2.20 (`6a6db7ff5`).
- Adopted `v<upstream-strata>-ex.<N>` release versioning and organized `README.md` with `ex.1` / `ex.2` release labels and the `Strata` / `exstrata` / `exstrata (low RAM)` memory comparison table.

## [ex.1] — 2026-11-22

Initial exstrata memory-reduction release on Qwen3.8-Flash-Next (`IQ3_XXS`), cutting combined VRAM + RAM by **14.1%** (85,699 → 73,590 MiB) with no loss in task accuracy.

### Added

- **Compressed n-gram (PLE) table (`T58e`, `--ple-gguf`)**: 3-bit trellis coding (448-bit sliding 12-bit window into a 4,096-entry codebook + FP8 row scale), shrinking the 320M-row PLE table from 90 B/row (26.8 GiB) to 57 B/row (17.0 GiB, ~−9.8 GiB RAM).
- **Dense weights arena (`STRATA_DENSE_ARENA=1`)**: Single VRAM allocation for all 300 native dense projection matrices (−186 MiB VRAM, bit-identical).
- **Shared MTP draft head (`STRATA_MTP_HEAD_INDEX=1`)**: Reads draft head rows by index from the main output head instead of keeping a copy (−178 MiB VRAM, bit-identical).
- **Lazy CUDA module loading (`CUDA_MODULE_LOADING=LAZY`)**: Documented standard CUDA switch saving ~0.31 GiB VRAM and ~1.3 GiB RAM (bit-identical).

### Verified

- Passed two-stage verification gate: teacher-forcing KL **0.071** (≤ 0.10), 914-item paired accuracy **92.2%** vs. Strata **91.5%** (+0.77 pts, no accuracy loss), decode speed **0.994×** (95% CI: 0.983–0.997, ≥ 0.97).
