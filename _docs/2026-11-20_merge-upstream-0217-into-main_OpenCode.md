# Merge 0.2.17 intake result into main — implementation log

## Overview
Merged the one-way upstream 0.2.17 intake result (`codex/upstream-oneway-20261119`,
commit `6b10cba43`) into the primary `main` branch, keeping main's own commits and
uncommitted work backed up. Official API changes are present and fork-only features
are retained.

## Background / requirements
- User: "main ブランチをこれに書き換える" then chose "成果をmainへマージ"
  (merge the result into main, preserving the uncommitted work and main-only commits).
- main HEAD before: `5a0df8e4b` (Q38 speedups), with 49 uncommitted files
  (0.6.6 release metadata + 25 `_docs` research logs) and 2 main-only commits
  (`5a0df8e4`, `e1af82da2` RAM32).
- The intake branch and main diverged at `bc733da8e`; the intake side already contains
  the same content for main's two commits (verified identical blobs for admission.hpp
  and ram_fit.py), so merging keeps all fork features.

## Decisions
- Backed up all 49 uncommitted files to
  `H:/Strata/evidence/main-uncommitted-before-0217-merge-20261120` (MANIFEST.txt) and
  `git stash@{0}` ("pre-0217-merge uncommitted 20261120") before touching main.
- Used `git merge --no-commit --no-ff 6b10cba43` (not a force replace).
- 9 conflicts, all resolved:
  - 4 add/add + 2 modify/delete + 1 add/delete: took the intake (new-layout) version;
    removed relocated old paths (`data/settings-register.json`,
    `include/strata/qwen38/admission.hpp`, `terminal/tests/test_ram_fit.py`).
  - `_docs/2026-11-20_ram32-syncstrata-port_OpenCode.md`: kept main's version
    (intake copy had a mojibake single character).
  - `engine/q38/src/CMakeLists.txt`: kept intake's `STRATA_Q38_STANDALONE=1` definition.
  - `engine/q38/src/engine.cu`: kept intake's `strata::qwen38::run` entry (official
    engine API) instead of main's standalone `int main`.

## Verification
- No conflict markers in any of the 3,282 changed code files.
- `/v1/models` has `max_model_len`, `context_length`, `max_context_length` and the
  `GET /v1/models/{id}` route with 404 handling.
- `terminal/data/settings-register.json` is valid JSON with 1,181 settings: official
  `STRATA_POOL_DEP` present and fork-only `STRATA_DS4_A_MB` retained.
- `engine/q38/src/engine.cu` exposes `namespace strata { namespace qwen38 { ... run(...) } }`.
- Targeted pytest `terminal/serve/test_server.py -q -k 'model or models'`:
  **8 passed, 15 subtests passed, 0 failed** in 6.27s (18 pre-existing DeprecationWarnings).

## Residual risk
- The stashed 0.6.6 release metadata is intentionally NOT re-applied (the intake
  branch carries newer 0.6.8/0.6.9 records); the `_docs` research logs remain available
  in the backup directory and stash if needed.
- Full Terminal/app suites, GPU tests, Q38 full-link/serve E2E were not run here;
  `releaseQualified=false` unchanged.

## Next actions
- Drop or selectively restore `stash@{0}` after confirming nothing in it is still needed.
- Run the broader suites before any release action.
