# gfx1101 hipBLASLt table (RX 7800 XT, Windows, hipBLASLt 1.5.0): calibration and prompt speed with / without

Measured on 2026-10-16 by durikim1-cpu. `docs/AMD_HIP.md` says gfx1101 has no shipped hipBLASLt table. This folder backs
`tools/hip/gfx1101-hipblaslt-100500.txt`: how it was calibrated, the per-row timings, and a prompt-speed comparison
on the **unmodified prebuilt Strata 0.1.44.1 Windows HIP engine**, where the only difference between the two arms is
`STRATA_HIPBLASLT_TUNING`. Main limitation: one card, one model (Swift 1.5 IQ3_XXS) at a 32K context; `fallbacks=0`
was not checked and the HIP ctest was not run (the prebuilt Windows package has no test binaries).

## Hardware and software

- AMD Radeon RX 7800 XT 16 GB (gfx1101); Ryzen 5 9600X; 64 GB DDR5; NVMe SSD; Windows 11 Home build 26200;
  AMD driver 32.0.31041.1004 (version at report time).
- Engine: prebuilt Windows HIP **0.1.44.1** from setup (`engine/BUILD.json`: `"source": "prebuilt"`,
  ROCm `10.2.0a20260930`, `hipblaslt_version` 100500). Same `strata.exe` for both arms,
  SHA-256 `c4709f17ef974ffa34af4f0c357160a89249cabf6e9a698fbe6d0dc65a3243fa`.
- hipBLASLt: the `libhipblaslt.dll` shipped in that package, version 100500 (1.5.0),
  SHA-256 `4ecfc8e9407f89f9434b11533338738c42180eda4ee9478922d6c8c6ceb76795`.
- Background load: desktop apps open (not an idle machine).

## Calibration

The Windows package has no `tune_hipblaslt` binary, so [`scripts/calibrate_lt.py`](scripts/calibrate_lt.py) is a
ctypes port of `tools/hip/tune_hipblaslt.cpp` that calls the packaged DLLs unchanged (public hipBLASLt / hipBLAS C APIs
plus the DLL's exported `getIndexFromAlgo`; [`scripts/inspect_dll.py`](scripts/inspect_dll.py) finds that export).
For each case it times plain hipBLAS (`hipblasGemmEx`) and every hipBLASLt heuristic candidate with GPU events,
warm-up then three repetitions, 32 MiB workspace, and keeps the fastest candidate that passes the same numerical and
padding gates as the upstream tool (relative L2 ≤ 1e-4, max abs ≤ 0.01, no NaN, no writes past the output).

Geometries (no solution ids are reused from another GPU):

- the 16 dense GEMM geometries of `gfx1100-hipblaslt-100500.txt`, each at T = 8192, 6144, 4096 and 1024 (64 rows);
- `f16 1280 2560 1280` (the small-T expert GEMM added in the 43-row gfx1100 revision) at T = 64, 256, 1024, 4096,
  6144, 8192 (6 rows).

70 rows, all pass the accuracy gates (largest relative L2 1.9e-5, largest max abs 0.0049). Per GEMM against plain
hipBLAS: 0.81x to 15.9x, median 6.45x, geometric mean 5.80x. One row is slower than hipBLAS:
`bf16 1 2560 1 T=1024` (hipBLASLt 0.0327 ms vs hipBLAS 0.0266 ms, 0.81x). It is kept because the end-to-end runs below used the full
table; dropping it would send that call to the nearest T bucket of the same geometry, not back to hipBLAS (see the
gfx1201 note in `docs/AMD_HIP.md`). Per-row timings: [`calibration-rows.csv`](calibration-rows.csv).

Note for re-running the script elsewhere: it pins `HIP_VISIBLE_DEVICES=1` (this PC's iGPU is device 0) and refuses to
start unless a local `pipeline-hold.json` exists (my harness used it to keep benchmarks from running during
calibration). Adjust both for your machine.

## Prompt speed with and without the table

Same binary, same model files, same launch arguments; arm B adds `STRATA_HIPBLASLT_TUNING` (and
`STRATA_HIPBLASLT_VERBOSE=1`). The engine log of arm B reports
`prefill gemm: hipBLASLt tuning enabled (70 rows, gfx1101, version 100500)`.

- Model: Swift 1.5 (`Swift-Qwen3.8-Flash-Next-GSQ-RCO`) IQ3_XXS native pack. `--max-context 32768`, FP16 KV (default),
  `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --suffix-draft 0 --resident-budget-gib 24`,
  `STRATA_RESIDENT_PIN=0`, `STRATA_UNBUFFERED_LOAD=1`. Thinking off, temperature 0, output cap 384.
- Prompt: a synthetic English ledger ending with a summary request; a unique tag in the system message makes every
  request a fresh prompt (0 reused tokens). Three runs per length; engine timings for tok/s, client timings for TTFT.
- Launch configs with local paths replaced: [`config-without-table.json`](config-without-table.json),
  [`config-with-table.json`](config-with-table.json). Per-run data: [`runs.csv`](runs.csv).

| Prompt tokens | Runs | Prompt tok/s without → with (median, range) | Decode tok/s without → with | TTFT s without → with |
| ---: | ---: | --- | --- | --- |
| 1,060 | 3 | 161.7 (148.1–163.5) → **184.5** (177.7–186.5), +14% | 28.2 → 29.7 | 6.60 → 5.79 |
| 8,041 | 3 | 335.1 (331.2–336.6) → **510.8** (507.3–515.4), +52% | 29.6 → 29.0 | 24.07 → 15.80 |
| 24,054 | 3 | 404.6 (403.6–407.5) → **695.2** (694.8–698.3), +72% | 29.9 → 29.2 | 59.51 → 34.67 |

Decode is unchanged within run-to-run spread, as on the other tables. A 12-question correctness set (extraction,
JSON format, arithmetic, four Python functions, long-document retrieval) scored 11/12 in both arms; the miss in both is
the same discount-and-shipping arithmetic question.

The arms ran one after the other on 2026-10-16 (16:17 and 16:29 local time), each on a newly started server, without
flushing the OS file cache.

## Not tested

`fallbacks=0` with `STRATA_HIPBLASLT_VERBOSE=1` (the verbose summary line was not captured), the HIP ctest
(`hip_prefill_hipblaslt_gemm`), Flash-Next with only the table changed (later runs with this table also changed other
settings), Linux, and any hipBLASLt 1.5.0 build other than the one in the 0.1.44.1 Windows package.
