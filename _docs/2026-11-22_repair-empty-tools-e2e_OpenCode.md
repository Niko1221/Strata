# Repair empty terminal/tools stubs and E2E regression — implementation log

## Overview
Found and fixed a regression introduced by the 0.2.17 one-way intake: 175
`terminal/tools/*.py` files were zero-length stubs, causing Control's
`gguf_reader` import to fail and `strata models list` to exit 1 repeatedly.

## Symptom (user: "GUI and engine erroring")
- Installed app is still 0.6.9 at `C:\Users\downl\AppData\Local\Programs\Strata`.
- Latest crash log `desktop-crash-20261121-135912.log`: WebView2
  `WindowsError(0x80070578)` ("invalid window handle") — an OS/WebView2
  startup issue, unrelated to this fork's code or version.
- 0.7.0 log `desktop-20261121-124619.log` showed repeated
  `models list: exit 1` while the engine component itself was installed.

## Root cause
Running the staged Control directly:
`ImportError: cannot import name 'BLOCK_GEOMETRY' from 'tools.gguf_reader'`.
- `terminal/tools/gguf_reader.py` was a 0-byte blob (`e69de29`) in the fork.
- Upstream 0.2.17 has the real 8,643-byte file; all 175 empty paths exist
  upstream with content. The intake created rename targets without writing
  the blob content.

## Fix
- `H:/Strata/evidence/repair-empty-tools.py` filled all 175 zero-length files
  from `upstream/main` (dry-run then apply; 0 skipped).
- Verified every repaired file now equals its upstream blob; 0 empty .py left.
- Re-staged Control (`stage-control.py`, exit 0); staged `gguf_reader.py` has
  `BLOCK_GEOMETRY`.
- `strata models list --output json` with the staged Control: **exit 0**,
  returned 4 run-config models.

## Playwright real-device E2E regression (results)
- `ui-flows.mjs` (Chrome headless + real staged Control + fake engine) was run.
  `restart` and `theme` scenarios passed; 12 scenarios failed only because
  Playwright looked for English button text (`Get started`, `Go straight to the
  app`) while the installed UI language is Japanese. The test file is
  byte-equivalent (modulo CRLF) to upstream 0.2.17, so the test is correct and
  the UI language is the cause, not a regression.
- An independent Playwright probe (`H:/Strata/evidence/probe-ui.mjs`) opened the
  packaged host: `health up: true`, `/api/native` 200, page title Strata, UI
  rendered in Japanese ("新しいチャット", "モデル", "Strataへようこそ /
  グラフィックスカードを確認しています…"), **pageerror 0, console errors 0**.
  The GUI renders correctly; WebView2 0x80070578 was a 0.6.9-era OS-side fault
  and does not reproduce here.
- Full empty-vs-upstream audit (`find-empty-vs-upstream.py`): after the 175-file
  repair, the only remaining empty-but-nonempty-upstream code file is
  `app/host/tests/fixtures/dsh-tool-fs-search-LICENSE` (test fixture, no runtime
  effect); the rest are docs/bench-results data intentionally omitted by the fork.
  The intake's empty-stub bug is fully repaired.

## Residual risks / next actions
- The published v0.7.0 installer contains the same broken stubs; per user
  decision it will be superseded by a new v0.7.1 publication, leaving v0.7.0
  in history.
- WebView2 0x80070578 is a separate OS-side fault; captured here, not fixed here.
