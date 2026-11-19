# RAM32 (Sync-Strata port) — landing on the new `terminal/` layout

Date: 2026-11-20 JST / OpenCode.

## Overview
Landed the already-implemented RAM32 port (reference session
`H:\SyncStrata-0.1.1\syncstrata-ram32gb.json`, original commit
`e1af82da2` on the old layout) onto the current `origin/main` new layout
(`terminal/...`) and pushed it. No new kernel or behavioral design — this is a
path rebase of a verified change.

## Background / requirements
User: "Make `zapabob/Strata` work fine on 32 GB RAM like SyncStrata does."
The reference session had completed the implementation on the old layout but
could not push because `origin/main` had been restructured (Terminal moved
under `terminal/`). Handover: `_docs/2026-11-20_ram32-handover_OpenCode.md`
in the main worktree.

## Assumptions / decisions
- Base: `origin/main` = `3f6e39f55` (new layout, includes the KV batch-store
  prefill optimization). Clean worktree `H:\Strata\worktrees\ram32-port-20261120`,
  branch `codex/ram32-port-20261120`. The user's main worktree was not touched.
- Manual path mapping, not cherry-pick (rename not detected by cherry-pick):
  - `strata_control/ram_fit.py` → `terminal/strata_control/ram_fit.py` (new file)
  - `tests/terminal/test_ram_fit.py` → `terminal/tests/terminal/test_ram_fit.py`
  - `serve/server.py` → `terminal/serve/server.py`
  - `serve/test_server.py` → `terminal/serve/test_server.py`
  - `setup.py` → `terminal/setup.py`
  - `data/settings-register.json` → `terminal/data/settings-register.json`
  - `tests/terminal/test_no_debugger_children.py` → `terminal/tests/terminal/...`
  - `docs/TROUBLESHOOTING.md` unchanged (root).
- The version-bump work in the user's main worktree (VERSION 0.6.7, READMEs,
  app package files) was left untouched and is out of scope here.
- Selected SOPs: common standard, implementation start gate, application
  development, Python. No web research was needed (primary source is the local
  reference session and the local git history).

## Changed files
- `terminal/strata_control/ram_fit.py` (new, 364 lines): stdlib-only memory
  read, commit-limit-aware budget, commit-heavy program names, two-stage
  squeeze with `STRATA_RAM_SQUEEZE` gate and `STRATA_FAKE_SQUEEZE` test fake.
- `terminal/serve/server.py`: import, squeeze before model load, commit-limit
  lines in `warn_budget_over_ram`, psutil-optional ctypes fallback.
- `terminal/setup.py`: commit advice after the "RAM budget" line.
- `terminal/serve/test_server.py`: `BudgetOverRam` extended to 7 cases.
- `terminal/tests/terminal/test_ram_fit.py` (new): 19 cases.
- `terminal/tests/terminal/test_no_debugger_children.py`: register `ram_fit.py`
  in `OTHER_PROGRAMS`.
- `terminal/data/settings-register.json`: `STRATA_RESIDENT_HEADROOM_GIB`
  launch read and TROUBLESHOOTING line-number shifts.
- `docs/TROUBLESHOOTING.md`: 32 GB commit-limit paragraph.
- `_docs/2026-11-20_ram32-syncstrata-port_OpenCode.md`: original implementation
  log (carried over).

## Commands run
- `git worktree add -b codex/ram32-port-20261120 <wt> origin/main`
- `git show e1af82da2:<file>` for the three new files
- `git diff e1af82da2^ e1af82da2 -- <modified files>` → path-rewritten patch →
  `git apply --ignore-whitespace`
- `py -3 -m py_compile terminal/strata_control/ram_fit.py terminal/tests/terminal/test_ram_fit.py`
- `py -3 -m pytest terminal/tests/terminal/test_ram_fit.py -q -p no:cacheprovider`
- `py -3 -m pytest terminal/tests/terminal/test_no_debugger_children.py -q -p no:cacheprovider`
- `py -3 -m pytest terminal/serve/test_server.py -q -p no:cacheprovider -k BudgetOverRam`
- `py -3 tools/settings_register.py --check` (ported vs clean comparison)
- `git diff --check`

## Test / verification results
- `py_compile`: exit 0 (both new files).
- `test_ram_fit.py`: **19 passed**.
- `BudgetOverRam` (serve/test_server.py): **7 passed**, 383 deselected.
- `test_no_debugger_children.py`: 16 passed, 2 failed — but the same 2 failures
  reproduce on a **clean `origin/main` worktree** (`updater.py` unregistered
  spawn site, `dsv41` mock expectation), so they are pre-existing and not
  caused by this port. The ram_fit registration line is present at line 194.
- `tools/settings_register.py --check`: identical output on clean and ported
  worktrees (45 lines, same set). Pre-existing drift only; this port adds no
  new drift.
- `git diff --check`: clean. No BOM in the new files.
- Reference evidence (old layout, from the implementation log): full
  `serve/test_server.py` 387 passed/3 skipped, rest of `serve` 1346 passed/
  24 skipped, RAM32 suites 7 passed/3 skipped, full `tests/terminal`
  3514 passed/16 failed/115 skipped (all 16 triaged, 1 fixed here).

## Residual risks
- The real two-stage squeeze is still not re-measured on this PC (only the
  test fake ran; the code is Sync-Strata's working implementation with this
  fork's names). A real `strata launch` with a RAM-budget model should confirm
  freed RAM and elapsed time.
- Squeeze runs only through `serve/server.py`; an app-host start that spawns
  the engine directly is out of scope.
- Not ported: per-start budget recompute, RAM-tiered pack choice,
  `STRATA_PARTIAL_PIN_GIB` pin rule (measured no faster here and changes
  numerics; left off by design).
- `tools/settings_register.py --check` remains non-zero due to pre-existing
  drift (unrelated STRATA_DS4_* / STRATA_Q38_* engine reads).

## Recommended next actions
1. One real `strata launch` with a RAM-budget model on this 32 GB PC to observe
   the squeeze and commit-limit lines in a live start.
2. Full Harness regression group before any release.
3. Decide how to reconcile the user's main worktree (uncommitted 0.6.7 bump)
   with the advanced `origin/main` — needs user confirmation; do not rebase
   automatically.
