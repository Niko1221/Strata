# App 0.2.2 Acceptance — 2026-10-22

The compiled portable Tray ran with the installed Windows App 0.2.2 and the
public Local Model Extension candidate. Real preload, unload, a short direct
chat, context/idle changes and 300-second automatic unload were observed.
This is a bounded acceptance run, not a benchmark or complete Windows release
certification. Scope agreement remains requested in [#2034](https://github.com/Niko1221/Strata/issues/2034).

## Provenance And Isolation

- Tray source: `ff69390ecaeb6e130cbe1939d78c64d6fa60a81b`.
- Portable `StrataTray.exe` SHA-256:
  `1d226b998942de822c5752caf74205266d0166da86a3b6dd40004f8fcb276e72`.
  All 30 original tool inputs and distributed managed files were checked against
  the source snapshot before adding this documentation. Runtime code is unchanged.
- App/Host: installed 0.2.2; bundled Control and existing CUDA engine: 0.1.47.
  Windows 11 x64, RTX 5080 (16303 MiB), Python 3.12.10, bundled Node 24.12.0.
- Existing Swift Qwen3.8 IQ3_XXS GGUF; a disposable profile used 8192 context,
  24 GiB resident budget, no vision, and separate loopback ports 18138/18139.
  Original profile files were not rewritten and no weights were downloaded/copied.
- An existing Python environment supplied official-server requirements; its base
  site-packages supplied `aiohttp` through child-only `PYTHONPATH`. Neither Python
  installation was changed. The bundled Control Python alone lacks `aiohttp`.
- App home, settings, components, models directory, workspace, temp and WebView2
  profile were disposable. Acquisition, bundled-engine installation and access
  repair were disabled. Actual WebView2 process arguments confirmed the isolated
  profile. The existing production Tray and its three listeners stayed running.

The [sanitized receipt](validation-app-0.2.2.json) records UTC observations,
source/package hashes, API/owner state and preservation checks. Extension source
`71e26101131235b42422494bff0550cb193200cf` includes the Storage correction found
during this run; initial lifecycle checks used its unchanged predecessor.

## Passed

1. **Build and regression:** 81 C# checks; all 16 managed Python cases in two
   invocations (12 configuration/update/fake-backend checks, then the four real
   mock-server cases); 15 file-IPC tests; 3 installed-App metadata checks. The
   four added integration executions were
   `LifecycleTests.test_official_mock_lifecycle_auth_and_config`,
   `test_protected_proxy_does_not_add_credentials`,
   `test_foreign_port_not_adopted`, and
   `test_stream_drains_before_stop_and_rejects_new_request`. All passed.
   Production/test executables used the production compiler inputs; a scratch
   build-script copy only wrapped the compiler with a windowless process runner.
2. **Real lazy startup:** the compiled Tray started its real controller and the
   installed Control server with `--engine strata --lazy`. Owner and live API
   confirmed unloaded, 8192 selected/effective context, no active requests,
   engine version 0.1.47 and `Use Official Component Updater` policy.
3. **Owner controls from App:** explicit preload/unload reached the existing
   serial worker through UUID-bound file IPC. Terminal acknowledgments, App UI,
   `/health`, `/v1/status` and model identity agreed. Context changed
   8192 → 65536 → 8192 while unloaded; idle 0 → 300 was confirmed by owner and API.
   Configured capacity while unloaded does not prove memory fit at 64K.
4. **Generation and automatic unload:** an actual App direct chat returned `OK`
   with `STRATA_AGENT=off` in that disposable App child. No further inference was
   submitted. The real server later logged `model unloaded after 300 s idle`;
   App/owner/API confirmed unloaded at 03:31:11 UTC. Engine processes had ended
   and GPU memory fell back to about 1.4 GiB. These are observations, not a speedup.
5. **Restart and cleanup:** a fresh compiled Tray process restored saved 8192/300
   and stayed unloaded. Stopping only the test Tray ended its owned controller
   and server; test listeners closed. Production Tray/listener identities remained
   unchanged. This exercised Windows Job cleanup, not the real Exit menu action.
6. **Preservation:** all 188 observed installed App/Host/Control files retained
   their SHA-256. Of 206 total observed files, 205 stayed identical; the sole delta
   was the existing production owner's mutable `context-status.json`. Original
   profiles, Tray settings and observed client configurations stayed identical.
   That live status file was read only and was not restored.

## Failed Or Limited

- Initial disposable configuration named a nonexistent expert-profile file.
  Preload failed with a terminal `RuntimeError`. Correcting only that test profile
  to the existing readable profile allowed preload; no product workaround was added.
- Default agent chat reached the server with 6545 prompt tokens at 8192 context,
  ended with a length limit and displayed no answer. Direct chat subsequently
  returned `OK`. Full agent-task acceptance is not claimed and effort behavior
  was not changed to hide this result.
- The Extension initially duplicated its Storage card on App 0.2.2's two native
  directory sections. That was fixed in #3373, with a failing regression followed
  by 71 passing Node tests and an actual App recheck. Tray runtime code was unchanged.

## Not Run

Physical notification-area menu interaction, Open App and graceful Exit menu
actions; Windows reboot/autostart; cross-monitor transitions; other GPU backends;
larger-context allocation/inference; live installer/signature/update/restart.
App launch in this run used the public Extension `launch.py`, so it does not
substitute for the Tray's Open App menu acceptance. No backend source was changed.
