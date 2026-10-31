# Project Maya (maya-ds4), DeepSeek-V4.1 engine

The files under `src/dsv41/`, `include/strata/dsv41/`, `tests/dsv41/` and `tools/dsv41/` come from
https://github.com/mw00/project-maya (release v1.1.17, commit 8d9a9b5d), its DeepSeek engine `src/ds4`, `tests/ds4_*`,
`tools/ds4` and the kernel sources it builds on (`exl3`, `sampler`). Project Maya is itself built on Strata.

Copyright (c) 2026 Peasant Smith (mw00) and the Project Maya contributors, including needmorevram.
MIT License, the full text is in `LICENSE` beside this file. The notice stays with every copy.

Changes made while porting into Strata are listed in the git history of the `src/dsv41` directory
(the first commit of each file is the unmodified upstream copy).

Later changes taken from Project Maya releases v1.1.18 to v1.1.26 (the same approach re-written onto Strata's engine,
for example the long-prompt layer skip, the thinking budget, conversation slots) are covered by the same copyright
and permission notice, in `LICENSE` beside this file.

MAYA-B port (2026-10-31), staged merges onto the v1.1.17 base:
- v1.1.17 to v1.1.19: the decode PCIe lane (#312, Peasant Smith / mw00; ds4_pcie.hpp, ds4_pcie.cuh, the lane block of ds4_model.cu,
  tests/dsv41/ds4_lane_bench.cpp, the lane case of ds4_gguf_test), with the same tag step's Q8_K activation rounding, penalties,
  DISK_SKIP_AUTO, the stdin read(2) loop and the CPU-lane tuning. In Strata the lane (DS4_STREAM=1) and the Q8_K rounding
  (DS4_IQ_Q8K=1) are opt-in until their gates have run.
- #329 (mw00): the lane judged over whole decode steps.
- #344 (needmorevram; selective port): the expert cache's recent slots (rc_* fields, the fill thread, tab_set, batched table updates), opt-in
  DS4_RECENT=<n>. Not taken: its all-reduce over DMA, segmented top-k and the tensor-core index scores.
