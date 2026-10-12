# 2026-10-12: Radeon AI PRO R9700 32 GB (gfx1201), ROCm 7.14.1, Ryzen 9 3900 (AVX2), 125 GiB RAM — UD-IQ4_XS and the opt-in HIP switches

## Machine

- GPU: one Radeon AI PRO R9700 32 GB (gfx1201) on a CPU-attached PCIe 4.0 x16 slot (B550 board). A second R9700 sits in
  the machine but is not used by Strata (`"gpu": 1`, one card, no layer split).
- CPU: Ryzen 9 3900 (Zen 2, 12 cores / 24 threads, AVX2, **no AVX-512**). RAM: 125 GiB DDR4. Model files on an NVMe SSD.
- Ubuntu 24.04, kernel 6.8, `amdgpu-dkms` 6.19.14, system ROCm **7.14.1** (HIP 7.14.60850, **hipBLASLt 1.4.1**, AMD clang 23).
- Strata `7695859` (0.1.43), built from source: `-DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DSTRATA_PREFILL_MMQ=ON -DSTRATA_BUILD_TESTS=ON`, llama.cpp `3cf03257`. HIP ctest
  **104 of 105** (`ple_parity`: needs the Q2_0 fixture; skipped: the hipBLASLt-table tests without a table,
  `expert_multi_test` without AVX-512).

## Model and settings

Unsloth **UD-IQ4_XS** at the pinned revision (SHA-256 match), `iq_pack.py --compat-bf16`, MTP draft layer from the
original checkpoint, draft vocabulary `cyrillic`.
`--resident-budget-gib 55` (all experts in RAM), `--expert-cache auto` (10,714 experts, 24.16 GiB; 576 MiB VRAM free
with everything loaded), `--prefill auto` (16,384), `--spec 4 --spec-min-p 0.5`, `--max-context 262144`, `--kv int8`,
`--kv-resident 32768`, default pool (11 workers). Server through `serve/server.py`, one request at a time.

## Method

`tools/strata_bench.py --long` against the running server, 3 runs per test, medians. One switch per arm, the server
restarted between arms (`runs/<arm>/README.md` and `results.json`, as `strata_bench.py` wrote them). The baseline measured twice (first and second session)
agreed within 2%.

## Results (tok/s; agent TTFT in seconds)

| Arm | chat decode | 4K prompt | 16K prompt | 32K prompt | agent decode | agent TTFT |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 57.3 | 1,296 | 1,753 | 1,788 | 74.4 | 2.79 |
| hipBLASLt table (below) | 58.8 | 1,311 | 1,755 | 1,795 | 70.0 | 2.79 |
| `STRATA_HIP_WMMA=1` | 58.9 | **1,617** | **2,547** | **2,687** | 71.7 | **2.24** |
| `STRATA_PF_FUSED=1` | 57.8 | 1,311 | 1,736 | 1,745 | 71.4 | 2.76 |
| `STRATA_ROUTE_RESIDENT=1` | **65.1** | 1,329 | 1,748 | 1,796 | **79.7** | 2.80 |
| `--prefill 8192` | 57.9 | 1,318 | 1,650 | 1,643 | 67.0 | 2.82 |
| WMMA + ROUTE_RESIDENT | 63.7 | 1,661 | 2,526 | 2,693 | 76.5 | 2.23 |
| the same, repeated | 63.1 | 1,624 | 2,538 | 2,707 | 82.1 | 2.24 |
| + hipBLASLt table | 66.0 | 1,626 | 2,563 | 2,720 | 76.0 | 2.23 |
| + `--pool-workers 10` | 65.7 | 1,637 | 2,550 | 2,719 | 78.6 | 2.23 |

- **`STRATA_HIP_WMMA=1`**: prompts +25% (4K) to +50% (32K), agent TTFT −20%; matches the +17..31% in AMD_HIP.md and
  grows with length here. Output bits change, as documented.
- **`STRATA_ROUTE_RESIDENT=1`**: decode +11% (chat) / +7% (agent) on this card and CPU. Reads like the "+1.6-2.4% (R9700
  32 GB)" in the merge note but larger here — a slower CPU (Zen 2, AVX2) computes the misses, so keeping routes resident
  pays more. Worth a line for AVX2 hosts.
- **`STRATA_PF_FUSED=1`**: no change with this pack (its experts are IQ3_S/IQ4_NL from Unsloth's file; the measured gain
  in AMD_HIP.md is on the native GSQ-RCO packs).
- **hipBLASLt 1.4.1 table** (`gfx1201-hipblaslt-100401.txt`, 135 rows): calibrated with `tune_hipblaslt
  --workspace-mib 32` over the **15 dense geometries this model actually logs** (found with a one-row seed table and
  `STRATA_HIPBLASLT_VERBOSE=1`: 10,214 launches over 231 shapes, T 67..16,384) at T = 64..16,384. Per GEMM 0.96x to
  3.06x over hipBLAS; end to end within noise (−1..+2%). This confirms the note in AMD_HIP.md that plain hipBLAS is
  already close on the R9700 with 1.4.1. The table is in this folder as data, not for shipping.
- **`--prefill 8192`**: 6-8% slower at 16K/32K than auto (16,384). The start-up WARNING that the lent cache slots'
  experts are "read from the SSD on every chunk" did not show in practice with 125 GiB RAM: a per-5-second NVMe read log
  over the whole session saw reads only at engine starts (the arena), none during prompts — the OS page cache held them.

## Two slots (`"parallel": 2`) against one, with WMMA + ROUTE_RESIDENT

`strata_bench.py --clients 2` (3 runs, medians), the server restarted between arms:

| | parallel 1 | parallel 2 |
|---|---:|---:|
| chat decode (one request) | 61.8 | 61.4 |
| 4K / 16K prompt | 1,631 / 2,551 | 1,603 / 2,509 |
| 2 clients at once: total tok/s | **55.0** | 51.5 |
| 2 clients: first token p50 / p95 | 1.61 / 3.58 s | **0.92 / 1.36 s** |
| 2 clients: request latency p50 / p95 | **3.89 / 5.54 s** | 4.96 / 6.03 s |

On one card the second slot answers the second client's first token sooner but finishes both later and lowers the
total; a single request is unchanged. Not measured: two long conversations interleaving (where one slot might evict
the other's conversation cache).

## Not tested

Quality benchmarks; contexts beyond 32K prompts; `--parallel`; the original GSQ-RCO packs on this ROCm (IQ3_S is the
next run); Windows.
