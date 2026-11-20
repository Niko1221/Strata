# Q38 changes ported onto zapabob main layout (2026-11-20)

## Overview

The Q38 speed/format/control-vector work originally landed on the local `main`
checkout as commit `5a0df8e4b` using the old paths `src/qwen38/*`,
`strata_control/*` and `tests/terminal/*`. zapabob's `origin/main` (425df0b29)
had since reorganised those paths into `engine/q38/*`,
`terminal/strata_control/*` and `terminal/tests/*`. A direct push was therefore
non-fast-forward.

## Port method

- Created worktree `H:/Strata/worktrees/q38-upstream-port-20261120` from
  `origin/main` (zapabob, **not** Niko1221 `upstream`).
- `git cherry-pick -X theirs 5a0df8e4b` let git's rename detection map the old
  paths to the new ones. The two new files (`cvec_kernels.cuh`,
  `geometry_test.cpp`) were placed manually at `engine/q38/src/`.
- The auto-merge dropped two structural lines from the upstream `engine.cu`
  entry region. Repaired them against current main:
  - `namespace strata { namespace qwen38 {` before the CLI entry point.
  - Renamed the CLI body from `main` to `run` to match
    `engine/q38/include/strata/qwen38/engine.hpp`.
- Added `STRATA_Q38_STANDALONE=1` to the standalone `strata-q38` target so the
  `#ifdef`-guarded wrapper `main` is compiled. Upstream's standalone target was
  already missing this define.

## Verification

- Verification used a temporary shim that mirrors the pre-reorg layout with
  junctions into the new tree; committed sources were not altered for the shim.
- CUDA 13.5 / SM120 / 5 jobs: **CTest 18/18**.
- 9B pinned fixture and the synthetic control-vector path were smoke-tested;
  see the session transcript for token output.

## Residual risks

- The integrated full-engine build (`strata.exe` via
  `engine/q38/CMakeLists.txt`) was not rebuilt here; only the standalone Q38
  target was verified.
- As before, QSA sparse-attention semantics and general model quality remain
  unqualified.
