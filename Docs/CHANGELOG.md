# CHANGELOG

Important historical changes and decisions. No raw logs.

## 2026 — V100 Stage 1 (branch `feature/v100-moe`, base `1ee8b66`)

- `5f75af7` **baseline** — original Strata build recorded red on V100 (no sm_70 codegen; ctest registrations unusable). Doc: `Docs/v100-stage1-baseline.md`.
- `6bb6b94` **SM70 support** — `CMAKE_CUDA_ARCHITECTURES=70`, `FA_ALL_QUANTS`, tf32 fallback for the QSA scorer on cards without MMA/ldmatrix (SASS-verified: 0 ldmatrix/0 MMA). Decision: emulate, do not redesign the attention path.
- `ae7b4fb` **ctest usable on V100** — in-tree registrations fixed; 19/20 green (`ple_parity` red by design, needs Q2_0 fixtures). Doc: `Docs/v100-build.md`.
- `12979f4` **dangling-else fix** (`src/kernels/cpu/pool.cpp`) — the `else` after the `sched_getaffinity` branch bound to the inner `if (CPU_ISSET(...))`; on success the fallback core list was pushed once per UNSET mask bit → 54,263 "cores" / pool workers on a 56-CPU machine (~15-minute startup hang in `clone3`). Fixed with two braces. Root-cause repro kept in `/tmp/pool_repro*.cpp`.
- `fa146c9` **Stage 1 docs** — testing (determinism + llama.cpp token-level cross-check: 32/32 identical on the primary prompt), benchmarks (prefill ~400 tok/s; sustained decode 37–40 tok/s; peak VRAM 18.3/15.6 GiB on 32/16 GB), final report (PASS).
- **Fork + PR** — forked to `noorazman/Strata`; branch pushed; PR **Niko1221/Strata#3** opened against `main` (maintainer_can_edit). Local clone `/home/noorazman/dsh/strata/Strata` kept pristine (remote `local-mirror`).

### Decisions of record
- Model for Stage 1: user's Swift-1.5 IQ3_XXS GGUF (no downloads).
- `--pool-workers 28` (physical-core count) recommended on this box: +9–13 % decode vs the 55-worker default (SMT contention).
- 2 MB hugepages for the expert arena measured SLOWER on this box (12.2 vs 35–37 tok/s, reclaim pressure on the 72 GB PLE page cache) → pool left empty; the arena auto-uses hugepages when configured.
- Reference engine: vanilla llama.cpp (the `ik_llama.cpp` production fork crashes on this box at CUDA init/NCCL).
- Hit-path caveat stays documented, not patched: the GPU expert-cache path is upstream-declared not-correct and opt-in; token-level match with the reference is the acceptance bar, and it was met on this model/quant.
