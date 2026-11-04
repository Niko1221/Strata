# Community tuning table for AMD Radeon RX 9060 XT 16 GB (gfx1200) on hipBLASLt 1.4.1 — a 100401 table with a served-path prefill A/B

Measured 2026-11-05 by er34testarossa. What was tested: this box's ROCm SDK reports
hipBLASLt **100401** (header version 1.4.1), and no gfx1200 table ships for that version
(`tools/hip/` carries `gfx1200-hipblaslt-100202.txt` only). A hand-built engine on this
library therefore falls back to plain hipBLAS unless `STRATA_HIPBLASLT_TUNING` is pointed at
a matching file — the documented mechanism in [docs/AMD_HIP.md](../../docs/AMD_HIP.md). This
folder tunes such a table **on this device against this exact library** with the tree's own
`tune_hipblaslt`, and measures the served engine with and without it using the tree's own
[tools/hip/bench_prefill.py](../../../tools/hip/bench_prefill.py).

Main limitation: the solution IDs are scoped to this hipBLASLt build — a ROCm update
invalidates the table, as it does the shipped tables. The A/B measures 4.2K and 8.8K prompt
prefill, not 100K+.

## Hardware and software

- RX 9060 XT 16 GB ×2 (gfx1200) in **helper (peer) mode** (`--peer-device 1
  --peer-reserve-mib 4096`), PCIe 4.0 x16; Ryzen 5 7600; 60 GiB DDR5; NVMe; Ubuntu 26.04.1,
  kernel 7.0.0-38. The desktop is not active during legs.
- Engine: source build of the **v0.2.10 / engine 0.1.56** tree (`git describe` = v0.2.10),
  CMake Release + HIP, `CMAKE_HIP_ARCHITECTURES=gfx1200`, linking the local SDK at
  `~/rocm-10.0.0/install` whose hipBLASLt reports **100401** at run time. The A/B and the
  table tuning both used this exact binary (md5 `db2016c77dbaac9734015d5c0656c0ba`).

## Model and configuration

Production serve config, unchanged between arms (Qwen3.8-Flash-Next GSQ-RCO IQ3_S, 2 shards;
`--expert-profile` + `--expert-cache auto`; `--peer-device 1 --peer-reserve-mib 4096`;
`--prefill 16384`; `--kv int8 --kv-resident 131072`; `--spec 4 --spec-min-p 0.5` with MTP;
`--max-context 262144`; control-vector scaled projection on). Engine env keys from the config
are identical in both arms.

The two arms differ **only** in `STRATA_HIPBLASLT_TUNING`: the `table` arm points at
`gfx1200-hipblaslt-100401.txt` (this folder, also placed in `tools/hip/`), the `off` arm
leaves it unset. The engine log confirms engagement per table leg:
`prefill gemm: hipBLASLt tuning enabled (32 rows, gfx1200, version 100401)`.

One honest deviation: the bench legs run as transient systemd units and do not inherit two
decode-path env knobs normally set by a systemd drop-in (`STRATA_ROUTE_RESIDENT_MTP=1` and
friends). They touch the decode/draft path only, not the prompt GEMMs, and both arms run
identically without them.

## How the table was made

See [tune-command.txt](tune-command.txt). The tree's `tune_hipblaslt` with
[shapes-canon.json](shapes-canon.json) (the same 16-shape set as the
[gfx1100 100500 community table](../2026-10-04-community-rx7900xtx-hipblaslt-100500/)),
tokens 4096/8192, workspace 32 MiB: 32 cases, every candidate correctness-gated (rel-L2
1e-4 / max-abs 1e-2, padding canary), every row its case's measured best. About 3 minutes
on one card with the engine stopped.

Honest note about three rows: in 3 of the 32 cases the measured-best solution is slower than
plain hipBLAS measured 0.94–0.99× as fast. They are kept because the end-to-end A/B below
ran against the complete 32-row file; if a maintainer prefers dropping rows that lose to the
default heuristic, removing exactly those three is a one-line-per-row change.

## Method

`tools/hip/bench_prefill.py` against the served engine: warmup, then fresh prompts of 4,210
and 8,830 actual fresh tokens (verified `reused = 0` by the driver), greedy, fixed seeds,
plus follow-up turns (reported but not compared). Three rounds, arms interleaved
(table, off / off, table / table, off), one fresh engine per leg (6 legs total). Prefill
tok/s is the engine's own `prompt … = 0 reused + N read in … ms (… tok/s)` line. Engine
restarts were fast (≈1 min) because the model stayed in the OS page cache across legs.
Every leg record is in [runs.json](runs.json) (message bodies omitted).

## Results

| Prompt (actual fresh tokens) | With this table, tok/s median (range, n) | No table, tok/s median (range, n) | Delta |
| --- | ---: | ---: | ---: |
| 4,210 | 1315 (1258–1406, n6) | 1311 (1252–1382, n6) | +0.3% |
| 8,830 | **1646** (1636–1653, n6) | 1620 (1611–1628, n6) | **+1.6%** |

- At 8,830 tokens the arms **do not overlap**: every table leg (1636–1653) beats every no-
  table leg (1611–1628). The table arm is also tighter.
- At 4,210 tokens the ranges overlap heavily; the +0.3% median gap is **not distinguishable
  from noise** on this box.
- No decode-path claim is made; the table only routes prompt GEMMs.

## Reproduce

1. `cmake --build build-hip --target tune_hipblaslt` in an existing HIP build dir.
2. Run [tune-command.txt](tune-command.txt) with the engine stopped.
3. Serve the production config with and without
   `STRATA_HIPBLASLT_TUNING=$PWD/gfx1200-hipblaslt-100401.txt`, one fresh engine per leg,
   and run `tools/hip/bench_prefill.py` per arm as in Method.
