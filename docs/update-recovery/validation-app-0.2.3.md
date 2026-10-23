# App 0.2.3 Update Recovery Validation

Measured on 2026-10-23, Windows x64, Node v24.18.0 and npm 12.0.2. This
reconciles #3578 at `a3b2e54e564ee4e4ce31c653dd5b4fcebb31a55c` with upstream
main `5c8dd322528cc84f68f461ad54a506b9da8fa8d4` using a normal merge.

Upstream now includes the recovery classification ported from #3578 and its
`jobErrorText` API. It explains a recovered interruption inside a red alert.
The remaining proposal in this PR is the presentation: put confirmed recovery
in informational, expandable history while retaining unresolved failures as
alerts. The merge reuses upstream `recoveredInterruptions`; it does not keep a
second recovery predicate or remove the upstream exports.

## Passed

From `apps/host`, after `npm ci --ignore-scripts` installed the locked packages
in an isolated checkout:

```text
node --test --test-reporter=tap --test-isolation=none tests/native-update-recovery.test.mjs tests/native-job-diagnostics.test.mjs tests/update-cache.test.mjs tests/ui4-fixes.test.mjs tests/component-updates.test.mjs
# tests 56
# pass 56
# fail 0
# skipped 0
npm run typecheck
npm run build
```

The production Vite build passed with its existing large-chunk advisory. The
PR diff against the new main passed `git diff --check`.

The render checks execute the actual React components with synthetic jobs and
read-only state/network stubs. They cover recovered App history, recovered
Engine history beside an unresolved App interruption, unchanged job records,
incomplete retries, invalid dates, component separation and ordinary failures.
The upstream helper/API tests and update/cache/control regressions also pass.

Commands ran on the resolved working tree before the merge commit. The JSON
receipt records the staged source blob IDs and working-tree SHA-256 hashes;
those source files were not changed after validation. The TAP artifact contains
the actual test output and has no account data or private paths.

- [Machine-readable receipt](validation-app-0.2.3.json)
- [Test output](validation-app-0.2.3.tap)

## Failed

No failure or skip in these 56 tests, type checking or the production build.

## Not Run And Limits

No fresh interactive browser capture, native desktop shell, real installer or
update restart, signed package, GPU or model execution was performed for this
reconciliation. The App 0.2.2 screenshots and receipt remain historical evidence;
they are not claimed as new App 0.2.3 browser acceptance. No installed App,
runtime/model state, release or deployment was changed.

The PR diff against main contains Host presentation/tests and this evidence.
It changes no GPU backend source; CUDA/HIP/SYCL/Metal builds are not applicable
to this PR. The merge includes upstream work without claiming to revalidate
those unrelated changes.
