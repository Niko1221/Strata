# gfx1100: bit-exact decode and prompt-path speedups (before / after)

The before / after data for the PR "HIP gfx1100: bit-exact decode and prompt-path speedups", measured with the
built-in benchmark `tools/strata_bench.py --long`.

- Machine: AMD Radeon RX 7900 XTX 24 GB (gfx1100), Ryzen 7 9800X3D, 62.4 GiB RAM, Linux 7.2.5, ROCm 10.2
  (TheRock 10.2.0a20260930); the desktop runs on the same GPU.
- Model: Qwen3.8-Flash-Next-GSQ-RCO IQ3_S, 262,144 context, `--kv int8`, `--spec 4` with the MTP drafter, no control
  vector, no vision.
- Both engines built on the same day with the same CMake options (`-DSTRATA_ENABLE_HIP=ON`,
  `CMAKE_HIP_ARCHITECTURES=gfx1100`, HIP flags `-DSTRATA_HIP_EXP_LAYOUTS -DSTRATA_HIP_MMVQ_IL`, Release).
- Before: upstream main `a0116b27`. After: this PR (`caa9b001` = `a0116b27` + the PR's commit).
- One config for every run (the full launch line and environment are in each report). It sets the PR's opt-in
  switches (`STRATA_WAIT_FUSE`, `STRATA_MMVQ_IL_PLAIN`, `STRATA_COMBINE_HITS`, `STRATA_ROUTE_PUB`, `STRATA_WAIT_COMBINE`,
  `STRATA_PF_LEAN_ISSUE`, `STRATA_PF_COPY2`); upstream's engine does not read them. `pr-defaults-only` is the PR build
  with those seven removed: only the PR's on-by-default changes.
- Runs alternated: upstream, PR, upstream, PR, then PR defaults-only. Each report is 3 runs per test, medians.

## Results

| test | upstream r1 | upstream r2 | PR r1 | PR r2 | PR defaults-only | PR vs upstream (mean of rounds) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chat decode tok/s | 84.8 | 88.7 | 96.5 | 97.3 | 92.3 | +11.7% |
| 4K prompt tok/s | 1049.0 | 1088.8 | 1118.0 | 1117.0 | 1116.6 | +4.5% |
| 16K prompt tok/s | 1360.5 | 1401.1 | 1427.1 | 1425.3 | 1426.3 | +3.3% |
| 32K prompt tok/s | 1421.7 | 1420.6 | 1445.6 | 1442.5 | 1444.8 | +1.6% |
| agent turn first token (s) | 3.97 | 3.82 | 3.75 | 3.75 | 3.73 | -3.7% |
| agent decode tok/s | 94.1 | 99.1 | 105.2 | 115.7 | 107.7 | +14.3% |
| agent cache hit % | 68.84 | 68.83 | 68.83 | 68.83 | 68.86 | same |

Every PR round beats every upstream round in each speed row. Decode medians move a few percent between runs on this
machine (the MTP / lookup drafts follow measured timings), so the decode rows have the widest ranges; see each report.

## Exactness

Teacher-forced KL (`kl-teacher-forced.txt`): both engines started directly with the same arguments, the expert
placement frozen (`--pcie-frac 0 --adapt-swaps 0`), the same expert slot count (7,229 on both), top-40 log-probs of
every position read through the verify windows after a 16K / 16K / 64K prompt-path read: 2,557 positions, 100.0% same
top-1, mean KL 0.0000, max KL 0.000, PPL 4.442 / 4.442. Upstream against itself also gives 0.0000 (the harness is
deterministic).

Not tested: CUDA and SYCL builds (no NVIDIA or Intel GPU here), other AMD architectures.
