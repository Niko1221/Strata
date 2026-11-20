# Upstream v0.2.17 one-way intake into the 0.2.16-ported Hakua fork — implementation log

## Overview
Applied official upstream `Niko1221/Strata` v0.2.17 (app/Terminal 0.2.17, engine/Control
0.1.63, feed sequence 129) on top of the fork worktree that already carries the one-way
0.2.16 intake plus Q38/RAM32/8-language custom features (VERSION 0.6.9). All fork-only
features are retained; official API changes (HTTP API, settings/env registry, CLI,
external integrations) are followed. No push, tag, release, or change to the primary
checkout (`C:/Users/downl/Documents/ChatGPT/Strata`, which has unrelated uncommitted edits).

## Background / requirements (user decisions 2026-11-20)
- Keep fork-only features; absorb official latest features / security fixes / bugfixes;
  follow the official API across HTTP API, CLI/launch, settings/env, external integrations.
- Do the intake in the existing "0.2.16-followed" worktree
  (`H:/Strata/worktrees/upstream-oneway-20261119`, branch `codex/upstream-oneway-20261119`),
  not in a fresh plain-merge worktree.
- A plain `git merge upstream/main` was impossible: the worktree keeps a hybrid layout
  (new `app/ terminal/ engine/` plus retained fork paths `apps/ serve/ src/`); a dry
  merge produced 7,802 conflicts and was aborted. All original worktrees were left clean.

## Selected standards / skills
- Skills: `milspec-codex-standard`, `deepresearch-defense-standard`.
- SOPs: Common, Implementation Start Gate, Skill Selection, DeepResearch, Security, Python.
- Primary sources: upstream git objects (fetched `v0.2.17`, `upstream/main` = 3142b66b9)
  and the official v0.2.17 release notes (GitHub) read 2026-11-20. No memory-only API claims.

## Key facts
- Previous 0.2.16 intake base `c7757280`, upstream side `58ee32277` (4 commits past the
  v0.2.16 tag, README/SUPPORTERS/Docker only) — recorded in
  `_docs/2026-11-19_upstream-oneway-merge_OpenCode.md`.
- Incremental base chosen: `58ee32277` (exact previously adopted upstream tip), fork =
  `HEAD` (0.6.9), upstream = `upstream/main` (0.2.17, 3142b66b9). `58ee322..0.2.17`
  is 543 files, +30,116 / -11,996.
- Official API anchors confirmed in the release notes and adopted below: `/v1/models`
  `max_model_len`/`context_length`/`max_context_length`, `GET /v1/models/{id}` (404 on
  miss); 33 new official `STRATA_*` settings; tool-call parser and error-response fixes.

## Changed files
- Tooling reused unchanged: `tools/oneway_upstream_merge.py`, `tools/verify_oneway_merge.py`.
- Applied tree: 373 adopt-upstream writes, 136 three-way merges, 2 deletes, 51 add/delete
  conflicts (fork retained), 5461 fork-only retained (0 errors).
- Manual conflict resolutions after apply:
  - `terminal/serve/server.py` — adopted official 0.2.17 `/v1/models` fields, context
    alias names, and `GET /v1/models/{id}` retrieve with 404; added `unquote` import.
  - `terminal/data/settings-register.json` — upstream 0.2.17 registry (1,125 settings)
    plus the 56 fork-only settings (DS4/Q38/Hakua) appended = 1,181 settings.
- Supporting scripts / evidence under `H:/Strata/evidence/upstream-0217-apply-r1/`:
  `merge_settings_register.py`, `record_manual_resolution.py`, `manifest.json`,
  `verification-final.json`, review copies.

## Commands run
- `git fetch upstream --prune` (primary checkout) — got v0.2.17 / 3142b66b9.
- Direct-merge dry run in the intake worktree → 7,802 conflicts → `git merge --abort`.
- `uv run --no-project --with tqdm python tools/oneway_upstream_merge.py --repo . \
   --base 58ee32277 --fork HEAD --upstream upstream/main \
   --evidence H:/Strata/evidence/upstream-0217-apply-r1 --mode apply` → 0 errors.
- `uv run --no-project python tools/verify_oneway_merge.py --repo . --manifest ...`.
- Manual merge + `record_manual_resolution.py` + re-verify.
- `python -m py_compile terminal/serve/server.py` (exit 0).
- `pytest terminal/serve/test_server.py -q -k 'model or models'` (result: appended below).

## Verification results
- Merge apply: 6,023 unified paths, 0 errors; adopt 373 / three-way 136 / delete 2 /
  add-delete conflict 51 / fork retained 5,461.
- Verify after manual resolution: `ok=true`, 0 failures; fork anchors 16/16 present,
  upstream anchors 13/13 present, 5,978/5,978 applied-file digests match.
- `terminal/serve/server.py` compiles; `/v1/models` fields, `{id}` route and `unquote`
  import all present.
- `settings-register.json` is valid JSON; new official settings STRATA_POOL_DEP (default),
  STRATA_SLOT_READ_WIDE / STRATA_P2P_HANDOFF (research), STRATA_ROUTE_TAIL_SKIP (default)
  and fork-only STRATA_DS4_A_MB all present.
- Targeted pytest (`terminal/serve/test_server.py -q -k 'model or models'`):
  **8 passed, 382 deselected, 15 subtests passed, 0 failed** in 7.39s. The 18
  DeprecationWarnings are the pre-existing positional `re.split` call in
  `serve/frontend.py:1124`, unrelated to this intake.
- Reviewed remaining high-priority conflicts (`native-control.mjs`, `speed_switches.py`):
  fork bytes are the correct choice there — upstream bytes would drop DS4 speed
  switches, Q36 launch contracts and the fork-only `flagValue` helper. Kept as-is.

## Residual risks
- ~78 review copies keep fork bytes, so upstream bugfixes inside those specific files may
  not be absorbed. Each has a diff3 review copy under
  `H:/Strata/evidence/upstream-0217-apply-r1/review/`. The two API-critical files were
  manually resolved; the remaining conflicts are mostly UI/TUI, snapshot goldens and
  engine source where fork semantics intentionally win.
- Q38 port is opt-in; full link / `--serve` E2E, GPU tests and signing are not part of this
  intake. Windows Authenticode signing remains absent; `releaseQualified=false`.
- The primary checkout retains its own 0.6.5 uncommitted edits and was not modified.
- CRLF warnings on git status are from the repo's existing `.gitattributes`, not corruption.

## Recommended next actions
- Review the remaining diff3 copies (prioritise `launch/screens.py`, engine
  `expert_source.cpp` / `prefill.cpp` / `q36_engine.cu`, and `native-control.mjs`).
- Run the broader Terminal/app suites before any merge into the primary `main`.
- Complete Q38 full-link + serve E2E before any release qualification.
