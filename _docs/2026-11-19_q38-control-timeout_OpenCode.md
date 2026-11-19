# Control model-list timeout: picker showed no models

## Symptom and root cause

In the normal installed profile, the chat model picker showed
"no local models downloaded" even though Qwen3.8 dense was registered.
Diagnostic: `H:/Strata/evidence/q38-picker-diag.json`. The page's own
`/api/native` returned `available:false, checking:true, models:[]` with
`Strata Control did not answer within 60 s (models list)`; repeated
polls kept the same error for over four minutes.

Direct measurement with the installed Control Python showed
`strata models list --output json` takes **78.1–81.5 s** on this host
(4 registered models, 4 install configs, `discovery_limit=128`),
while `strata components engines` takes under a second. The host's
`installedModels()` call in `control-runtime.mjs` had a hard 60 s
timeout, so it killed the scan and reported `models=[]`. The picker
correctly rendered an empty list from that stale result. This is not
a bug in `ChatModelPicker.tsx`.

## Change

`apps/host/server/control-runtime.mjs`: raise the
`installedModels()` timeout from 60 s to 180 s so a slow directory
scan finishes and the real model list reaches the UI. No other
timeout, no resource floor, no model config and no signature rule
changed. `engine()` already used 120 s for a native runner probe, so
a longer read for a disk-backed model scan is consistent.

## Verification

- `node --test tests/component-updates.test.mjs`: 23/23 pass.
  Added a test asserting `installedModels()` passes `180_000` to the
  Control runner and returns the model list instead of an empty set.
- `npm run typecheck`: pass.
- The real 78 s Control scan must now complete within the new window;
  that requires rebuilding the app host and re-running the installed
  GUI readback in the normal profile. Pending.

## Release note

This change is **after** the already published `v0.6.4` tag. It must be
built, installed-over, and verified in the installed GUI before it is
committed and published as a new tag (`v0.6.5`). Do not amend `v0.6.4`
or silently replace its asset. Keep stable Latest unchanged until the
new tag is published.
