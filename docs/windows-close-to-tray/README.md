# Windows Close To Tray Acceptance

This opt-in candidate keeps the existing App Host alive when a main-window X
hides the Windows App. A companion supplies its own live process ID and absolute
image path; the shell holds the verified process handle. Starting the App again
uses the existing single-instance restore path. Explicit `--quit` and owner exit
use the existing Host shutdown path.

Normal X without a verified opt-in owner still closes the App. The existing
resource lookup, updater, Host command, shutdown token and Windows Job stay in
place. Linux and macOS do not compile or call the new module. No Engine, Control,
server or frontend code is changed. The separate Tray proposal is #3365; its
managed-mode scope remains separate from this native lifecycle hook.

## Reproduce The Opt-In Behavior

Use a disposable App installation and a real, visible companion. Supply the
three environment values described in [Closing](../CLOSING.md#optional-windows-close-to-tray)
when launching `strata-desktop.exe`. Do not start the App inside a companion Job
that is meant to contain only its server workers.

1. Record the App, Host, companion and listener identities. Confirm the App UI
   and an authenticated owner browser surface both reach the same Host.
2. Click the actual native X. Confirm that the window disappears while the same
   App and Host remain, and that browser metadata and Monitor still work.
3. Launch the same App executable again from the companion. Confirm the same
   window returns with unchanged App and Host identities.
4. Hide it again. If hardware testing is intended, send one short prompt through
   the owner browser and confirm the completed reply. Metadata alone is not
   evidence of inference.
5. Send `strata-desktop.exe --quit`. Confirm the App and its Host exit cleanly.
   A cold `--quit` must not start a Host. Test owner exit separately.

## Real Deployed Reference

The screenshot below is an actual acceptance capture of a separately deployed
native Extension derived from official App 0.2.2 commit
`011c9e600a93bbc9fd451c563501b5831c32bbca`, not a mockup. Its integration source
was `b1308ce6dbbcbe5bd859953486f1a5acfc625ddd`, native package
`0.1.0-ef4a4e88db40`, on Windows x64 with Engine 0.1.47, an RTX 5080 16 GB,
61.4 GB usable RAM and a local Huihui V2 Swift IQ3_S model.

During that test, native X removed the App from the visible-window inventory;
App, Host, companion and 8080/8081 listener identities stayed unchanged. The
running Tray's real Open App command restored the same native window. Physical
taskbar-icon clicking was not automated. The window was hidden again before
owner Web generated `CLOSE_TO_TRAY_FINAL_OK`. Models metadata and Monitor each
returned HTTP 200. The model was loaded at effective Context 204800 (200K), with
1800-second idle unload and the current model's Sampling baseline preserved.
Explicit Quit and a subsequent Tray cold-open were also observed.

The public screenshot contains only QA text, generic App UI, a public model
display label and incidental usage figures. It contains no account identity,
private path, hostname, credentials, original user screenshot or unrelated chat.
The usage figures are not a benchmark or a speedup claim.

![Actual Owner Web Reply While The App Was Hidden](web-background.png)

The reference Extension also selected installed resources and handled companion
startup outside its server Job. Those site-specific launcher/deployment changes
are not part of this upstream candidate. The screenshot does not establish that
the submitted candidate was installed or that a future App version will work.

## Candidate Checks And Limits

Run on native Windows from `apps/desktop/src-tauri`:

```text
cargo test --locked -- --test-threads=1
cargo build --locked --release
```

The targeted Windows workflow runs these against the submitted Rust source in a
disposable build checkout. It substitutes an inert compile-only page and no
bundled resources; it does not create a distributable installer, download an
Engine, run the App or start a model. The existing crash-report test now checks
the package version rather than a stale literal 0.2.0; runtime logging is unchanged.
New regressions check missing/mismatched owners, an exact live executable handle,
owner exit, and exact `--quit` argument handling.

Candidate CI results are linked in the PR at the tested commit. The deployed
reference and its screenshot are separate evidence, not proof of candidate
installation. Signed-installer update, updater exit, Windows logoff, candidate
GUI execution, other App versions and Linux/macOS builds remain unverified.
Scope agreement for this optional native hook is requested in #2034.
