# App 0.2.2 Acceptance — 2026-10-22

The public launcher and external package ran in the actual installed Windows
App 0.2.2, with the compiled public Tray candidate as the sole model owner.
The run exposed and fixed duplicate Storage cards. Current Model, Manage Models,
Storage and exact Use In Chat selection were then checked in the native App.
Scope agreement remains requested in [#3369](https://github.com/Niko1221/Strata/issues/3369).

## Tested Inputs

Extension source `71e26101131235b42422494bff0550cb193200cf` includes the focused
Storage fix. Its corrected immutable package is `1.0.0-088a8606771d`, manifest
SHA-256 `088a8606771d8182c2959cf9a2274dbbdba3e27a874e15ee699a10243df8bf92`.
The preceding package `1.0.0-dd445334ea48` supplied the unchanged launcher,
bridge, owner API and lifecycle implementation for the initial checks. The
[sanitized receipt](validation-app-0.2.2.json) binds the source files, both
packages, actual observations and remaining limits.

Tray source was `ff69390ecaeb6e130cbe1939d78c64d6fa60a81b`, compiled executable
SHA-256 `1d226b998942de822c5752caf74205266d0166da86a3b6dd40004f8fcb276e72`.
App/Host 0.2.2, Control/engine 0.1.47, Windows 11 x64, RTX 5080, Python 3.12.10
and the App's Node 24.12.0 were used. This is not an installed App upgrade test.

All activation state, test credentials, App data/settings/components/workspace,
temp and WebView2 profile were disposable and outside the App install directory.
Acquisition, automatic bundled-engine installation and access repair were disabled.
The real existing GGUF was read through a copied profile with 8192 context,
24 GiB resident budget and no vision. No model download/copy or production-client
configuration change was needed. The existing production Tray/server stayed running.

## Passed

- Public `extension_manager.py install --source <package> --root <external-root>
  --activate`, `launch.py --config <disposable-config> --check` and actual
  `launch.py --config <disposable-config>` succeeded. The exact native child
  Host reported App 0.2.2, Bridge API 1, compatible/ready, and the expected
  immutable package. Actual WebView2 arguments confirmed isolated browser storage.
- Current Model displayed the verified physical ID, loaded state, selected/effective
  context and idle setting. All eight context values came from the actual
  `/api/native` catalog. Manage Models retained `No downloaded models` and showed
  the connected model separately.
- Explicit Use In Chat from Current Model and Manage Models selected the native
  picker with title `Acceptance Local API · acceptance-local-model`. The source
  dialog closed, the credential sentinel remained intact and an unrelated model
  row retained its name and capacity.
- Real preload/unload received terminal owner acknowledgments. Context changes
  8192 → 65536 → 8192 and idle 0 → 300 were confirmed by owner and live API.
  Process restart restored 8192/300 without preload. Unloaded configured capacity
  is not proof of GPU allocation at 64K.
- Actual direct chat returned `OK` with child-only `STRATA_AGENT=off`. The existing
  server later unloaded automatically after 300 seconds without inference; the
  owner, live API and App agreed. This is model-connection acceptance, not a
  successful default agent-task run or a performance comparison.
- **Storage regression:** App 0.2.2 has two native directory sections. The old
  observer inserted two cards. A new regression failed with `2 !== 1`, then
  passed after anchoring one card to the enclosing panel. In the actual App,
  the corrected package showed exactly one card, preserved both directory
  sections and `0.0 GB`/`No downloaded models`, scrolled normally, and completed
  Storage Use In Chat with the exact physical target.
- **Recovery:** a disposable synthetic 1.0.1 package exercised activation history.
  `disable` made home HTML byte-identical to the installed official file; a control
  POST returned 503 without changing the owner request. Fresh native App startup
  succeeded without Current Model. `rollback` restored the exact original 1.0.0
  hash and Current Model returned after native restart. Preferences were unchanged.
  The corrected source package was then activated and rechecked in the real App.
- All 188 observed installed App/Host/Control files retained SHA-256. Original
  profiles, Tray settings and observed client configs stayed identical. The one
  delta among 206 observed files was the production owner's mutable
  `context-status.json`; it was read only and not restored. Only test processes
  were stopped, and production listener identities were preserved.

Regression commands (local windowless runner omitted):

```powershell
node --test tools/local-model-extension/owner_api.test.mjs tools/local-model-extension/extension_bridge.test.mjs tools/local-model-extension/host-preload.test.mjs tools/local-model-extension/package/ui.test.mjs
python -B -m unittest discover -s tools/local-model-extension -p test_*.py
```

**71 Node tests passed, 0 skipped**, using `STRATA_TRAY_SOURCE` and
`STRATA_TEST_PYTHON` for the real cross-language IPC case. **13 Python tests
passed**. The companion Tray passed 81 C# checks, 16 managed tests, 15 IPC tests
and 3 installed-App metadata checks. No backend source was touched.

## Failed Or Limited

The initial test profile referenced a nonexistent expert-profile file, producing
a terminal preload error. Only the disposable configuration was corrected.
The original Storage duplication was reproduced and fixed as described above.
Default agent chat reached the real server with 6545 prompt tokens at 8192
context but ended with a length limit and no answer; direct chat passed.
No effort, Fast, permissions or retry policy was changed to conceal that limit.

## Not Run

Live App/Engine installation, signature verification or upgrade/restart;
Windows reboot/autostart; physical Tray menus and Open App action;
multiple-monitor transitions; other GPU backends; larger-context memory fit;
full default agent-task acceptance. These results do not establish arbitrary
future App 0.2.x DOM compatibility or a general upstream plugin API.
