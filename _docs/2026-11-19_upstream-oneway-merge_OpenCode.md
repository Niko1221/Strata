# One-way upstream merge with fork-feature preservation — implementation log

## Overview
Built and ran a policy-driven Python one-way merge from official upstream
`Niko1221/Strata` `upstream/main` (58ee32277, v0.2.16+) into the Hakua fork in
an isolated worktree. Common code follows the official v0.2.16 layout; the
Q38 dense CUDA engine is kept in a dedicated area and registered on the
official engine API.

## Background / requirements (user decisions 2026-11-19)
- Target: `upstream/main` HEAD (all 4 commits after v0.2.16 included).
- Layout: hybrid — `engine/ app/ terminal/ docs/ tools/` for common code;
  fork-only engines in a dedicated area.
- Q38: port onto the official engine 0.1.62 API, not kept as an unrelated
  standalone binary.
- Scope: isolated worktree/branch only; primary checkout and its uncommitted
  experimental edits untouched; no push, no tag, no release.

## Selected standards / skills
- `milspec-codex-standard`, `deepresearch-defense-standard`.
- SOPs: Common, Implementation Start Gate, Skill Selection, DeepResearch,
  Security, Python.
- Primary sources: upstream git objects and the official v0.2.16 release
  notes (GitHub). No memory-only claims used for the API description.

## Key facts
- Merge base: `c77572806996349b49995e4f98d41b5294d36a21`.
- `c775728..upstream/main`: 7052 renames, 461 adds, 137 deletes, 102
  modifies — a near-total tree relocation.
- Official engine API: resident `strata --serve` process over stdin/stdout
  (no dlopen ABI); `ModelFamily` / `family_layout.hpp` registry; schema-2
  `engine-component.json` + `BUILD.json`.
- Fork Q38 architecture string: `general.architecture=qwen35`, dense 64 trunk
  layers, `expert_count == 0` — distinct from upstream `qwen35moe`.

## Changed files
- `tools/oneway_upstream_merge.py` — policy engine (dry-run/apply,
  blob batching via `git cat-file --batch`, rename-aware unified path space,
  ownership map, API reference rewriting, build-foundation override).
- `tools/verify_oneway_merge.py` — anchor + digest verification.
- Merged tree in this worktree: 9638 applied file writes, 127 deletions,
  7052 rename destinations.
- `engine/include/strata/artifact/model_family.hpp` — added `Qwen38Dense`.
- `engine/CMakeLists.txt` — opt-in `STRATA_ENABLE_Q38` / `add_subdirectory(q38)`.
- `engine/q38/{engine-component.json,BUILD.json,PORT_NOTES.md}` — schema-2
  component template and explicit remaining-work list.

## Commands run
- `git fetch upstream --prune`, `git worktree add -b codex/upstream-oneway-20261119`.
- `uv run --no-project --with tqdm python tools/oneway_upstream_merge.py ... --mode dry-run`
  (9771 paths, 7052 rename rules, 0 errors).
- Same with `--mode apply` into the isolated worktree.
- `uv run --no-project python tools/verify_oneway_merge.py ...`.
- `uv run --no-project python <port script> --repo <worktree>`.
- Python compile checks on five ported fork anchors; conflict-marker scan.

## Verification results
- Merge: 9771 unified paths, 0 errors; adopted 7080, retained 2257, three-way
  merged 295, deleted 127, add/delete conflicts 12.
- Verification: `ok=true`, 0 failures; fork anchors 16/16 present; upstream
  security/correctness anchors 13/13 present; 9638 applied-file SHA-256
  digests match; 0 conflict markers in the applied tree.
- 133 three-way conflicts: fork bytes retained by policy; marked diff3 review
  copies are under `H:/Strata/evidence/upstream-oneway-20261119-apply-r3/review/`.
- Q38 port: `Qwen38Dense` reachable from `model_family_from_arch("qwen35")`,
  `model_family_runs` includes it; CMake option present; schema-2 manifest
  created. Five representative ported Python files compile.

## Residual risks
- The Q38 port is a scaffold, NOT build-verified. Remaining semantic work is
  enumerated in `engine/q38/PORT_NOTES.md`: family layout table, `--serve`
  protocol adapter, main() entry switch, real build, signature, E2E.
- 133 conflicted files retain fork bytes, so upstream bug fixes inside them
  are not absorbed; each has a review copy for manual resolution.
- No full build, no E2E, no GPU test, no release action was performed.
- `releaseQualified=false` unchanged.

## Recommended next actions
1. Resolve the 133 review copies, prioritising host server and workflow files.
2. Complete the four PORT_NOTES items and build with `-DSTRATA_ENABLE_Q38=ON`.
3. Run the official Terminal/app suites before considering a merge to main.

## Addendum 2026-11-19 (port continuation)
- Q38 entry point added: `strata::qwen38::run`, family switch in
  `engine/src/program/generate.cpp`, official `ENGINE_REQUEST` completion line.
- `engine/q38/CMakeLists.txt` + GPU-engine link wiring added; CMake configure
  with `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_Q38=ON` now succeeds (exit 0).
- Two merge defects found and fixed during the configure:
  - fork-added `tests/core|prefill|hip|quant|program|cuda|platform|gdn|ngram`
    files had landed in `terminal/tests/`; 127 files were moved to
    `engine/tests/` and the relocation table in the merge script was updated.
  - the fork's stale `.gitignore` rule `/engine/` made all official engine
    source untracked; it was removed and `engine/` is now tracked (2714 files).
- `strata_q38` (and its three dependencies) **built successfully** with CUDA
  13.5.37 / sm_120 under UAC elevation (`BUILD_EXIT=0`). `strata_q38.lib`
  (2.97 MB), `strata_q38_ops.lib` (1.77 MB), `strata_q38_kv.lib` (825 KB) and
  `q38_spin8.lib` (227 KB) are in `H:/Strata/builds/q38-vs-135h/q38/Release/`.
  Evidence: `H:/Strata/evidence/upstream-oneway-20261119-build/verification.json`.
  See `engine/q38/PORT_NOTES.md` for the build-environment notes and the
  remaining items (full `strata` link + --serve E2E, family_layout, signed
  component, 133 conflict reviews).

