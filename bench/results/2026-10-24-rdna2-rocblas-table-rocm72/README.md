# RX 6900 XT (gfx1030): a rocBLAS table for ROCm 7.2.4 as Arch Linux packages it (2026-10-24)

The table shipped for gfx1030 (`tools/hip/gfx1030-rocblas-5.6.0.8d1ae90e.txt`, see
`bench/results/2026-10-05-rdna2-rocblas-table`) is for rocBLAS 5.6.0.8d1ae90e. Arch Linux's ROCm 7.2.4 (package
`rocblas 7.2.4-3`) reports rocBLAS `5.2.0.dabb6df2b98-dirty`, so the engine refuses that table there and runs the
default kernels. This is the same tuner's table for that build: `tools/hip/gfx1030-rocblas-5.2.0.dabb6df2b98-dirty.txt`.
Setup already points `STRATA_ROCBLAS_TUNING` at `tools/hip` on gfx103x, and the engine picks the file of its own
rocBLAS build. No code changes.

## Rig

- AMD Radeon RX 6900 XT 16 GB (gfx1030), PCIe 4.0 x16, Resizable BAR on; AMD Ryzen 9 5950X (16 cores), 64 GB DDR4;
  Arch Linux (Omarchy), kernel 7.2.8, ROCm 7.2.4 from Arch's packages (`rocblas 7.2.4-3`, `hip-runtime-amd 7.2.4-1`),
  `lib_dirs` `/opt/rocm/lib`.
- Model: Qwen3.8-Flash-Next GSQ-RCO Coder IQ1_M (2 shards, `--native`), the setup config of this PC:
  `--expert-cache auto --prefill auto --spec 4 --mtp <rt> --max-context 262144 --kv int8 --resident-experts
  --vram-reserve-mib 3072 --spec-min-p 0.5 --adapt-every 100000 --adapt-swaps 160 --adapt-decay 0.97`
  (`--adapt-every 100000` in both arms for the A/B, as in the 2026-10-05 report; the config uses 1).
  Env: `STRATA_HIP_PROMPT_F16=1 STRATA_SELECT_SIMT=1 STRATA_RESIDENT_PIN=0`, plus `STRATA_PREFILL_TIMING=1
  STRATA_ROCBLAS_VERBOSE=1` for the measurement. [config-on.json](config-on.json) is the on arm's config.
- Engine: 0.1.48.1, a local HIP build of upstream `7ddc17a3` (2026-10-23) with changes in `src/qwen35` only (the
  Qwen3.6 path, which this model does not use), `-DSTRATA_ENABLE_HIP=ON -DGPU_TARGETS=gfx1030 -DSTRATA_PREFILL_MMQ=ON`.

## Table

`LD_LIBRARY_PATH=/opt/rocm/lib build-hip/tune_rocblas --tuning-out tools/hip/gfx1030-rocblas-5.2.0.dabb6df2b98-dirty.txt`
with the defaults. It wrote 112 rows, 78 of them with a solution and the other 34 `default`.
[tune-summary.txt](tune-summary.txt) is the tool's full output. The largest gains are at small T:

- N = 10240 K = 2560 (GDN input projection) at T = 256 ... 1024: 7.4-7.7x.
- N = 12288 K = 2560 at T = 256: 6.0x.
- N = 1 K = 2560 at T = 512: 18.9x (the default's 1-column GEMM).

Every kept solution was checked against the default kernel's output: relative L2 about 2e-5, no padding writes.

## Method

The same as in the 2026-10-05 report. There is one server per arm, the same binary in both arms. The off arm has no
`STRATA_ROCBLAS_TUNING`, the on arm points it at `tools/hip`. The prompts are 12 in all: four lengths, three times
each. Each prompt starts at its own offset of `docs/DETAILS.md`, so no prompt reuses another one's prefix. Each prompt
ends with "Reply with one word: what is this text about?", and the request uses `max_tokens` 8 and temperature 0.

The metric is the GPU timeline from the engine's `strata prefill timing` line. Each arm ran three times
(`timing-<arm>-run<n>.txt`). A run is a fresh server with all 12 prompts.

## Results

The GPU timeline is given in ms for runs 1 / 2 / 3. Δ is the change of the mean.

| prompt tokens | table off | table on | Δ |
| ---: | ---: | ---: | ---: |
| 685 | 2239 / 2092 / 2048 | 1774 / 1781 / 1775 | -16.4% |
| 776 | 2208 / 2244 / 2195 | 2007 / 1893 / 1888 | -12.9% |
| 977 | 2498 / 2405 / 2385 | 1999 / 1990 / 1985 | -18.0% |
| 985 | 2591 / 2543 / 2547 | 2306 / 2102 / 2100 | -15.3% |
| 1084 | 2178 / 2179 / 2192 | 1920 / 1924 / 1913 | -12.1% |
| 1264 | 2019 / 2004 / 2026 | 1947 / 1956 / 2050 | -1.6% |
| 1353 | 2061 / 2022 / 2040 | 1967 / 1973 / 1963 | -3.6% |
| 1510 | 2079 / 2058 / 2079 | 2005 / 2004 / 1974 | -3.7% |
| 1543 | 2115 / 2212 / 2178 | 2193 / 2074 / 2116 | -1.9% |
| 10917 | 10100 / 10155 / 10169 | 9388 / 9372 / 9359 | -7.6% |
| 11396 | 10687 / 10486 / 10505 | 9637 / 9679 / 9632 | -8.6% |
| 12164 | 10626 / 10583 / 10595 | 10143 / 10158 / 10167 | -4.2% |

- **Below 1,152 tokens: -12 to -18%.** The `gdn` phase falls from about 380-560 ms to 150-190 ms. `hc read` also gets
  faster.
- **1,264-1,543 tokens: -2 to -4%.** Here the GDN projection already runs on rocBLAS's fast kernel, and only `hc read`
  gains (about 190-240 -> 120-155 ms).
- **10.9K-12.2K tokens: -4 to -9%.** The gain is mostly `hc read` (1.8-2.3 s -> 1.2-1.3 s).
- **No refusals:** the engine refused no solution (`tuned_launches=6268 fallbacks=0` per on-arm run).
- **Answers:** each arm gives the same answers in every repeat (runs 2 and 3, compared). Between the arms, 11 of 12
  answers are the same. The 1,089-token prompt of the second set differs at its 6th reasoning token ("request:" against
  "request."), see [answers-off.txt](answers-off.txt) and [answers-on.txt](answers-on.txt). The table's kernels
  compute a different FP16 rounding than the default ones (the tuner allows a relative L2 of 2e-3).
- Decode does not use these GEMMs; not measured here.

## Limits

- **Scope:** one machine, one model.
- **rocBLAS build:** the table holds only for rocBLAS `5.2.0.dabb6df2b98-dirty` (Arch's package). Another distribution's
  ROCm 7.2.4 build may report another version string. In that case the
  engine refuses the table and says so in its log.
- **Other cards:** gfx1031 / gfx1032 were not tested.
