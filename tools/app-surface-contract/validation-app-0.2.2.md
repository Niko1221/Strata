# App 0.2.2 Windows Compatibility

The published helper rejected the installed official App 0.2.2 before reading its
metadata, with `official_app_version_needs_validation`. This update explicitly
admits App 0.2.1 and 0.2.2 and records the inspected version in `surface.appVersion`.
Both versions must still pass the same reviewed route/method/owner anchors,
metadata bounds and fresh source identity checks. App 0.2.3 remains unsupported.

Validation ran on 2026-10-22 (Asia/Kuala_Lumpur), Windows x64, Node v24.18.0,
installed official App 0.2.2 and selected Control 0.1.47. The helper changes are
based on `8932fa9c08b748f8378eb90af5d4dd34d153e6a9`; the accompanying
[sanitized receipt](validation-app-0.2.2.json) identifies the tested source bytes,
installed source identity and exported manifest. Hashes describe tested filesystem
bytes; unchanged Windows checkout files may have CRLF line endings.

## Passed

From the repository root, omitting the local windowless-runner wrapper:

```text
node --test tools/app-surface-contract/contract.test.mjs tools/app-surface-contract/assets.test.mjs
```

19 tests passed, 0 failed, 0 skipped. This includes a 0.2.2 snapshot/export/fresh
verification case, refusal to verify that candidate against a different App
version, unchanged fixture files, and route/method/owner-anchor mutation rejection
for both reviewed versions. Existing privacy, source/Control/Python drift,
inventory, no-overwrite and unknown-version checks still pass.

The installed App also passed:

```text
node tools/app-surface-contract/cli.mjs snapshot --host <installed-0.2.2-host>
```

The installed-App acceptance harness then executed the exported functions in this
order, with an exclusively created candidate outside the App:

```javascript
const inspected = await inspectApp(host);
const snapshot = await readCapabilities(inspected);
const exported = await exportCandidate(inspected, candidate, snapshot);
const fresh = await readCapabilities(await inspectApp(host));
const verified = await verifyCandidate(candidate, {
  manifestSha256: exported.manifestSha256, current: fresh,
});
```

Observed results:

- `snapshot.state = compatible`; App and surface version both `0.2.2`.
- All 14 ownership areas passed their source anchors. The official eight Context
  values and Idle range 0–86400 seconds remained valid.
- 75 official frontend files exported byte-for-byte; fresh verification returned
  `verified`. The receipt records the source fingerprint and manifest SHA256.
- SHA256 comparison preserved 184 Host package/server/frontend files, the three
  selected Control source/bundle files and four existing private configuration
  files. No private paths, raw configuration or hardware catalog is published.

## Failed And Resolved

Before the fix, the new 0.2.2 regression failed with
`official_app_version_needs_validation`. The actual installed CLI snapshot
returned the same `needs_validation` reason. Both pass after the fix; no unresolved
failure remains in the checks listed above.

## Not Run And Limits

This establishes metadata and asset/source-identity compatibility. It does not
establish all API payload semantics, App rendering or a production companion
integration. The existing screenshots still depict the unchanged 0.2.1 fixture.

No Host, model, updater or deployment operation was requested. No real model
download/load/inference, GPU allocation, Idle timing, App install/update/reboot,
browser acceptance or macOS/Linux installed-App acceptance was performed. No
engine backend source changes, performance comparison or future-version support
are included. Upstream CI results remain separate from these local checks.
