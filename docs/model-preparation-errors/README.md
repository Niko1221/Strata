# Selected Model Preparation Errors: App0.2.2 Validation

Preparation errors stay with the selected model. Canonical IDs and family/name aliases match; unrelated remembered or latest jobs stay out of the selected card. A matching failure stays visible.

Validated on 2026-10-22 with Windows x64 and Node v24.12.0. The machine-readable [receipt](validation-2026-10-22.json) records SHA-256 inputs, command arguments and retained output hashes. This continuation changes documentation only; passed checks were reused after source/hash verification.

## Passed

- PR #3361 at `b910b705fa24555c658f9b01e7ee443cf728b73c`: **43 passed, zero failed, zero skipped**.
- TypeScript checking and the Vite production build passed at this PR head. The existing anonymous before/after component images remain under [model-job-errors](../images/model-job-errors/after.png).
- A separate combined scratch candidate on App0.2.2 base `011c9e600a93bbc9fd451c563501b5831c32bbca` with both PR implementation patches passed **68 unique checks**, plus typecheck and Vite build.

The combined result is shared by the two reports. It is not the source submitted individually in either PR, and it must not be added to the individual results or counted twice. It consists of 39 core passes, 28 wider UI passes, and one previously blocked wider UI check after its missing source inputs were restored.

## Expected Baseline Failures And Setup Issues

The unpatched App0.2.2 implementation, with the new test fixtures, produced **17 passes and eight expected failures out of 25**: four model-job UI failures and four direct-API effort failures. Applying both implementation patches resolved all eight.

An initial dependency-copy layout prevented esbuild resolution. A later wider UI check lacked `setup.py` and `tools/strata_mcp.py` in the host-only archive. Both were scratch setup issues: the layout was corrected, exact-base files were restored, and only the blocked wider test was rerun (one pass). They required no runtime or test-source fix. Vite emitted large-chunk and external-output-directory warnings; both recorded builds exited zero.

## Source Binding And Combined Input

The receipt separates this PR's exact head from the combined source. Text hashes normalize CRLF to LF; byte hashes identify the actual Windows inputs. Compared installed App0.2.2 Host sources (package metadata, entry point, model providers, vision probe and agent profile) matched the unpatched base after CRLF normalization. The installation was read only.

The combined scratch input was an archive of `apps/host` at the App0.2.2 base, with `setup.py` and `tools/strata_mcp.py` from that same base. It received:

- #3361's `NativeModels.tsx` implementation diff from `b910b705` and its guided-setup regression file.
- #3363's `server/index.mjs` and `server/model-providers.mjs` implementation diff from `ab57c5da`, plus its API-effort and mock-API fixture files.

The #3363 implementation diff needed one context line instead of three beside changed App0.2.2 context. The implementation was unchanged. Full input hashes are in the receipt; no parent PRs or other implementation changes were applied to this scratch source.

## Portable Commands

The commands below preserve the executable arguments and working directory roles. Substitute absolute local paths for `{OUT}`, `$Node`, `$HostRoot` and `$HiddenRunner`; private operator paths are intentionally omitted. `$HostRoot` is `apps/host` under the exact PR checkout for the first block and under the combined scratch source for the second. Use existing dependencies at the receipt's recorded versions. No package download is part of these commands.

Set `TEMP` and `TMP` to disposable output subdirectories. API/provider checks used `PSModulePath=$env:SystemRoot\System32\WindowsPowerShell\v1.0\Modules`. For Harness checks, set `STRATA_HARNESS_DIR` to the staged Harness below and `STRATA_COMPONENTS_DIR` to a disposable components directory. Top-level Windows commands used `invoke-hidden.ps1`. A scratch ESM preload in `NODE_OPTIONS=--import=<file-URL>` forced child `spawn`/`spawnSync` to use `windowsHide:true` and called `syncBuiltinESMExports()`; it did not change product source. TAP reporter destinations persist output independently of PowerShell console redirection.

```powershell
# pr3361-exact-narrow-resolved: 18 passed, 0 skipped
& $HiddenRunner -FilePath $Node -ArgumentList @('--test','--test-reporter=tap','--test-reporter-destination={OUT}/pr3361-exact-narrow-resolved.tap','tests/guided-setup-empty-catalog.test.mjs') -WorkingDirectory $HostRoot

# pr3361-affected-ui: 25 passed, 0 skipped
& $HiddenRunner -FilePath $Node -ArgumentList @('--test','--test-reporter=tap','--test-reporter-destination={OUT}/pr3361-affected-ui.tap','tests/model-picker.test.mjs','tests/native-fit.test.mjs','tests/languages-and-catalog.test.mjs','tests/qa1-downloads.test.mjs','tests/native-fb5.test.mjs') -WorkingDirectory $HostRoot

# pr3361-typecheck: exit 0
& $HiddenRunner -FilePath $Node -ArgumentList @('node_modules/typescript/bin/tsc','--noEmit') -WorkingDirectory $HostRoot

# pr3361-build: exit 0
& $HiddenRunner -FilePath $Node -ArgumentList @('node_modules/vite/bin/vite.js','build','--outDir','{OUT}/dist/pr3361') -WorkingDirectory $HostRoot
```

The combined baseline command ran before the implementation patches. The remaining commands ran after them. The initial wider UI run's one missing-input failure is shown accurately; the selected rerun completes the 68 unique successful checks.

```powershell
# app022-before-resolved: 17 passed, 8 failed, 0 skipped
& $HiddenRunner -FilePath $Node -ArgumentList @('--test','--test-reporter=tap','--test-reporter-destination={OUT}/app022-before-resolved.tap','tests/guided-setup-empty-catalog.test.mjs','tests/api-reasoning-effort.test.mjs') -WorkingDirectory $HostRoot

# app022-after: 39 passed, 0 failed, 0 skipped
& $HiddenRunner -FilePath $Node -ArgumentList @('--test','--test-reporter=tap','--test-reporter-destination={OUT}/app022-after.tap','tests/guided-setup-empty-catalog.test.mjs','tests/api-reasoning-effort.test.mjs','tests/model-providers.test.mjs','tests/openrouter-reasoning.test.mjs','tests/thinking-direct.test.mjs') -WorkingDirectory $HostRoot

# app022-affected-ui: 28 passed, 1 failed, 0 skipped
& $HiddenRunner -FilePath $Node -ArgumentList @('--test','--test-reporter=tap','--test-reporter-destination={OUT}/app022-affected-ui.tap','tests/model-picker.test.mjs','tests/native-fit.test.mjs','tests/languages-and-catalog.test.mjs','tests/qa1-downloads.test.mjs','tests/native-fb5.test.mjs') -WorkingDirectory $HostRoot

# app022-fit-missing-input-resolved: 1 passed, 0 failed, 0 skipped
& $HiddenRunner -FilePath $Node -ArgumentList @('--test','--test-name-pattern=the tiers of the engine','--test-reporter=tap','--test-reporter-destination={OUT}/app022-fit-missing-input-resolved.tap','tests/native-fit.test.mjs') -WorkingDirectory $HostRoot

# app022-typecheck: exit 0
& $HiddenRunner -FilePath $Node -ArgumentList @('node_modules/typescript/bin/tsc','--noEmit') -WorkingDirectory $HostRoot

# app022-build: exit 0
& $HiddenRunner -FilePath $Node -ArgumentList @('node_modules/vite/bin/vite.js','build','--outDir','{OUT}/dist/app022') -WorkingDirectory $HostRoot
```

## Harness And Limits

The Harness fixture used the real staged DeepSeek Harness `0.2.1-alpha.3` at `571902028a75c64c1d767f4ae5fd099485f0ddf0`, connected to a scripted loopback model. Its bundle SHA-256 is `66975a57810290957c506d5c3906f8479d2e64f56ec65a5d1dc19dc81c224b3a`. This staging also contains the separate PR #3684 `strata-pi-ai-context-safety-v2` patch; neither this PR nor the combined App0.2.2 implementation input submits that packaging patch.

The component tests render actual React components with anonymous state. The Host and Harness are real disposable processes; their model replies are scripted. **No native App window, packaged installer, model download/preparation, real GPU/model inference, live cloud-provider inference or cross-platform execution was tested.** These results establish source/component and protocol behavior within that scope. Runtime and test sources, installed application files and production configuration were unchanged.
