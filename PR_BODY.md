PR title:
  HIP backend on gfx1200 (RDNA4, RX 9060 XT): the gfx1100 backend runs unchanged with three deltas

PR body (paste below the line):

---

This carries the merged AMD HIP backend (PR #121 by Konstantinos Korres, rebased onto 0.1.24, engines
0.1.25/0.1.26) to **gfx1200** (RDNA4, RX 9060 XT 16 GB, ROCm 7.2 / hipBLASLt 100202, wave32 verified
at run time) - a card this repository has not tested. The backend's assumptions hold on RDNA4: wave32
and the 64 KiB workgroup LDS limit are unchanged, so the gfx1100 code runs with **three deltas**:

- `cmake/hip_backend.cmake`: the arch gate accepts gfx12xx beside gfx11xx.
- `src/core/device.cu`: the runtime device check accepts `gfx1200` beside `gfx1100`.
- `include/math_constants.h` (new): a fallback for ROCm installs that do not ship
  `<math_constants.h>` (this 7.2 one); the hip_compat include dir is searched after the toolchain's,
  so a real header always wins. The engine references only `CUDART_INF_F/NAN_F/PI_F`, all defined.

One data file: `tools/hip/gfx1200-hipblaslt-100202.txt`, hipBLASLt solutions **calibrated on the
card** with `tools/hip/tune_hipblaslt` at the shapes the engine launches (the shipped gfx1100 tables
do not dispatch here; 7-15x over plain hipBLAS per shape, +43% prefill end-to-end). The table is
arch+version-guarded and falls back to plain hipBLAS anywhere else; the recalibration command is in
the doc, so other cards can regenerate their own.

Two bitwise-neutral tweaks on the portable attention path (the sm80 tensor-core kernel is compiled
out under HIP, so prefill runs the decode kernel): `#pragma unroll 4` on the value accumulation and
a prefill query batch of 64 instead of 32 for L2 reuse of overlapping selections. Greedy outputs
byte-identical across runs.

Tested: 30/30 ctest on the card (the two documented exclusions apply), 512-token and back-to-back
runs without stalls, 1.4 GiB VRAM free at steady state (`--vram-reserve-mib 1792`). Measured
(IQ1_M Coder, greedy, MTP): **31.0 tok/s decode / 541 prefill** on a 2,374-token prompt;
**761/26.3 at a 65K prompt**; **637/22.7 at 130K**. A rejected experiment is recorded in the
commit: gfx1200 reports WGP count (16) in `multiProcessorCount`; doubling ggml's `nsm` was A/B-tested
(726.6 vs 731.1 tok/s at 64K) and not kept.

New files: `docs/AMD_HIP_GFX1200.md` (setup, tests, recalibration, running, limits),
`bench/results/2026-09-30-gfx1200/README.md` (arm-by-arm tables and rejected arms),
`gfx1200-run.sh` (launcher for the measured configuration; paths via env vars).

---

I can't ssh my system to you even if i wanted due to conditions, maybe the patches made by GLM5.3-Flash from freebuff can help you to find something that we didn't see, thanks for the effort
