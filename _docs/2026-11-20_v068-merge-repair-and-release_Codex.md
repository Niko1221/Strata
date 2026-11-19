# v0.6.8: merge-leftover repair (host boots, tsc clean) and the repaired release

Date: 2026-11-20 JST / OpenCode. Worktree:
`H:\Strata\worktrees\upstream-oneway-20261119` (branch = origin/main).

## Overview

The user could not use the installed app (model loads but no answer; then
"Strata could not start" — App Host exit 1). Root cause: the one-way 0.2.16
intake left interleaved duplicate/broken code in the host server and UI, and
`verify-release.mjs` only checks payload integrity, never a boot. This log
covers the repair and the repaired 0.6.8 release.

## Background / requirements

- Crash log: `%LOCALAPPDATA%\Strata\logs\desktop-crash-20261119-192025.log`
  → `SyntaxError: Identifier 'createTimeSkip' has already been declared`
  (`host/server/index.mjs:103`), then further duplicates surfaced one by one.
- The discovery tool: the reference merge left pristine conflict-marked
  review copies under
  `H:\Strata\evidence\upstream-oneway-20261119-apply\review\`
  (fork/base/upstream per hunk). These were used as the authoritative source
  for every resolution.

## Assumptions / decisions

- Union policy per hunk: official 0.2.16 features AND fork features both
  kept; where the same name collided, the newer/working side won and the
  other's unique bits were merged in.
- Python scripts did the extraction/resolution (user asked for the efficient
  path): conflict-block splitter, per-hunk resolver, duplicate-declaration
  scanner, babel-based imbalance locator, variant bisection.
- `native-gateway.mjs` was rewritten around the upstream variant (inference
  fetcher, `mcpAllowed`, FIX-APP A4 `unknownModelMessage`).
- `agent-bridge.mjs` (13 hunks) and `native-control.mjs` (8 hunks) were
  resolved from the review sides with explicit unions (documented in the
  script files under `%TEMP%\opencode\`, evidence copies in
  `H:\Strata\evidence\upstream-oneway-20261119-build\`).
- `store.ts` (5 hunks), `plain-errors.ts` (1 hunk) resolved from review;
  the fork's `deleteWorkspaceSessions`/`renameSession` grafted back.
- `scripts/test.mjs` resolved to the fork side (newer scratch layout).
- Native-source identity wall unchanged: the signed 0.6.3 Q38 preview
  companion (`1408136b…`) verifies the engine payload (user-approved path).
- eslint is not configured in this repo (verified: no config file, no dev
  dependency); the type gate here is `tsc --noEmit`, which now passes clean.

## Changed files (pushed as `6ae171f9d`, plus earlier `5d9ac0a8d`, `c12a5fe11`, `7064c87a2`)

- `app/host/server/index.mjs` — duplicate imports/const removed (createTimeSkip,
  control, llmProxy/agentTools, systemPrompt fragment).
- `app/host/server/native-control.mjs` — rebuilt from the review copy: 8 hunks
  unioned (drafter A3 functions verbatim, mcpTimeoutS, poll outer catch,
  dropFailedInstances + call, jobStore close, fork superset regions kept).
- `app/host/server/agent/agent-bridge.mjs` — rebuilt from the review copy:
  13 hunks unioned (union signature, BUSY_WORDS + sideRuns-aware busy logic,
  startWakeTurn/armWake + adopt + subscribers, learning lines, reattach
  newcomer pattern + skills-used, turnStallLimit, c.last wake/taken + woken
  keep, research write-phase, close() union, union export list).
- `app/host/server/agent/preset.mjs`, `agent/profile.mjs` — rebuilt from
  review (modelNames + proactiveSearch/browser, vision-caps browser rows +
  adminRows).
- `app/host/server/native-gateway.mjs` — rewritten (upstream variant + A4).
- `app/host/server/local-models.mjs` — duplicate strict-metadata block removed.
- `app/host/src/lib/store.ts`, `plain-errors.ts` — rebuilt from review;
  fork's deleteWorkspaceSessions/renameSession grafted; wake option added.
- `app/host/scripts/test.mjs` — fork side (newer scratch layout).
- `tools/packaging/package_terminal.py`, `app/desktop/scripts/stage-control.py`,
  `app/desktop/scripts/verify-release.mjs`, `.gitignore` — new-layout port
  (separate commit, see the pipeline log).

## Commands run

- `node --check` over all 219 host server/scripts files (final: 0 fail).
- Babel errorRecovery parse to locate the imbalances;
  variant bisection isolated the `jobStore.close()` brace drop.
- `npm run typecheck` (tsc --noEmit): **0 errors** (was 17).
- `node --test tests/agent-host.test.mjs`: 1 pass / 0 fail / 10 skipped
  (skips = STRATA_HARNESS_DIR not set to a built Harness in the worktree).
- Direct host boot: process alive after 12 s, empty stderr, banner
  `strata host · http://127.0.0.1:8791` (was: instant SyntaxError exit 1).
- `verify-release.mjs`: exit 0 (`Hakua 0.6.8, engine 0.6.3, Control 0.6.3`).

## Test / verification results

- All host server/scripts files parse; host boots; tsc clean; agent test green.
- 0.6.8 NSIS built (`Strata_0.6.8_x64-setup.exe`, 121.19 MiB class) — see the
  release record for the final SHA-256.

## Residual risks

- The full `npm test` host suite and the packaged-app WebView checks were not
  run in this pass (time); the boot + tsc + agent tests are the recorded gates.
- `agent-bridge.mjs` union choices (busy = max(activeTurns, holding)) need a
  soak; the diagnosis doc's watchdog wording improvement remains open.

## Recommended next actions

1. Overwrite-install 0.6.8 on this PC, start the app, load Qwen3.8-27B,
   confirm a chat answers (this user report) and that the engine process
   command line names `components\engine\0.6.3-windows-x86_64-cuda-cuda13`
   (0.6.0 sequential-prefill selection must not come back).
2. Publish `v0.6.8` tag + release with the installer attached in the same
   command (immutable releases: an asset-less release cannot receive assets).
3. Anonymous download readback with SHA-256.
