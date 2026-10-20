# Synthetic companion UI tests

These fixtures use the real companion bridge's TLS certificate and pairing authentication with an in-memory synthetic host. They supply saved history, Thinking, ordered progress, browser action cards, approvals, continuation, Stop, and model download management. They never discover or control a running Mac app, download model files, start a model, or open a browser.

Requires macOS, Node.js 24.12.0 or later, Xcode, an installed iPhone simulator, and the matching `apps/companion-bridge/bridge.mjs` from the Mac companion changes. From the repository root:

```sh
node apps/ios/test-support/shared-conversation-fixture.mjs --test \
  'platform=iOS Simulator,name=iPhone 17 Pro'
```

Select a simulator installed on your Mac. The command starts the fixture, runs only `CompanionSmokeTests/testSharedMacHistoryLiveBrowserApprovalAndContinuation`, then removes its private temporary TLS identity and stops both loopback listeners. The test uses the app's **Forget this Mac** flow to replace any pairing saved in that simulator, and leaves an offline fixture pairing afterward. Use a disposable simulator if you want to preserve an existing simulator connection. It never changes a physical phone's pairing.

For the model download review, target selection, progress, Stop, retry, relaunch, and older-Mac fallback workflow, start a fresh combined fixture:

```sh
node apps/ios/test-support/companion-fixture.mjs --test-downloads \
  'platform=iOS Simulator,name=iPhone 17 Pro'
```

This runs only `CompanionSmokeTests/testModelDownloadReviewProgressCancelRetryAndTargetSelection`. The combined fixture also supports `--test` for the shared-chat workflow. The original `shared-conversation-fixture.mjs` remains available for existing test commands. Each run requires a fresh invitation and performs the same simulator-only setup and cleanup described above. Download controls require the Mac companion changes' model-management routes; API-only providers are not download targets.

When the Mac and iOS changes are in separate checkouts, set `STRATA_TEST_BRIDGE_MODULE` to the matching checkout's `apps/companion-bridge/bridge.mjs`. The default import is relative to this repository. `STRATA_TEST_DERIVED_DATA` optionally selects a build directory; the default is `apps/ios/build`.

For a test runner that manages Xcode itself, run the fixture without `--test`. It prints only the path of a private temporary `connection.json` containing `invitation` and `control` values. Pass them as `STRATA_TEST_INVITATION` and `STRATA_SHARED_FIXTURE_CONTROL` in that runner's environment. With command-line `xcodebuild`, use the `TEST_RUNNER_` prefix; the `--test` command handles this forwarding automatically. The loopback control endpoints are synthetic test controls, not Mac API routes. Stop the fixture with Control-C when finished; normal termination removes the temporary files.

For download tests, run `companion-fixture.mjs` without a test flag and pass its control URL as `STRATA_MODEL_MANAGEMENT_FIXTURE_CONTROL` instead. The `--test-downloads` command forwards this value automatically. Keep the invitation and generated connection file private.

Pairing invitations expire after five minutes and work once. Start a fresh fixture for each UI run. Keep XCTest result bundles private: UI automation can record the values entered into the pairing form. No TLS keys, invitations, tokens, or result bundles belong in source control.
