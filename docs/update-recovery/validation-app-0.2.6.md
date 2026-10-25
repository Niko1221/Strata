# App 0.2.6 Update Recovery Validation

Measured on 2026-10-25, Windows x64, Node v24.18.0 and npm 12.0.2.
This reconciles #3578 at `4532a840712cea4df29ac1d67e9f7f78608c3b8b` with upstream main
`4b9e05a2a806a3fe24f7b2560d98626c37dade35` using a normal merge.

The conflict was two imports at the same location in `SettingsUpdates.tsx`.
The resolution keeps upstream `shownEngineVersion` and the PR's
`RecoveredUpdateHistory`. Upstream engine-version rendering remains in place;
confirmed recovered interruptions remain expandable history, while unresolved
failures remain alerts. The shared recovery predicate and error-text API are
unchanged.

## Passed

The exact locked Host dependencies were installed with lifecycle scripts disabled:

```text
npm ci --ignore-scripts --no-audit --no-fund --cache <task-cache>
```

`<task-cache>` is a directory in this task workspace; the private absolute path is omitted.

From `apps/host`, using task-owned settings, components and temporary directories:

```text
node --import ./tests/helpers/isolate-profile.mjs --test --test-reporter=tap --test-isolation=none tests/native-update-recovery.test.mjs tests/native-job-diagnostics.test.mjs tests/update-cache.test.mjs tests/ui4-fixes.test.mjs tests/component-updates.test.mjs tests/engine-version.test.mjs
# tests 63
# pass 63
# fail 0
# skipped 0
npm run typecheck
npm run build
```

The isolation helper was preloaded before any test imports. The rendered recovery
checks use synthetic jobs and read-only state/network stubs. They cover recovered
App and Engine history, unresolved failures, component separation, timestamp edge
cases and unchanged job records. The upstream engine-version helper, update/cache
and Control regressions are included. The production Vite build passed with its
large-chunk advisory. The PR diff against the pinned main passed `git diff --check`.

Commands ran on the resolved source before the merge commit, from
2026-10-25T14:18:37.341557+00:00 through 2026-10-25T14:18:50.516836+00:00.
The JSON receipt records the staged tree before these evidence files were added,
plus source blob IDs and SHA-256 hashes. Those source files did not change during
or after validation. The TAP artifact contains the actual test output without
private paths or account data.

- [Machine-readable receipt](validation-app-0.2.6.json)
- [Test output](validation-app-0.2.6.tap)

## Failed

No failure or skip in these 63 tests, type checking or the production build.

## Not Run And Limits

No fresh interactive browser capture, installed desktop shell, real update/restart,
signed packaging, GPU build or model execution was performed. The App 0.2.2 images
and App 0.2.3 receipts remain historical evidence. No installed App, model state,
release or deployment was changed.

The PR diff against the pinned main contains Host presentation/tests and evidence.
It changes no GPU backend source; CUDA/HIP/SYCL/Metal builds are not applicable to
this PR delta. The merge includes upstream work without claiming to independently
revalidate its engine or installer changes.
