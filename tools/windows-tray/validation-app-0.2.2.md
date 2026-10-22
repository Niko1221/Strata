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


## Follow-Up: Public Native Menus And Open App

A fresh isolated run used the same compiled public Tray binary and runtime
source, with `Acceptance Menu Model`, selected/effective 8K Context and separate
ports 18148/18149. A user-supplied native screenshot confirmed the real menu's
Profile, effective Context/Idle, lifecycle controls and updater ownership labels.
The Windows helper could target the App window but did not expose the Tray
popup, so menu interactions are explicitly human-assisted.

**Open App passed at 2026-10-22T04:22:48.4492562Z.** The actual menu started the official App
0.2.2: App PID 16504 had Tray PID 29212 as parent;
Host PID 2180 was the App's child. `/api/health` reported the
managed `http://127.0.0.1:18148/v1` URL, physical model `acceptance-menu-model`
and Context 8192. This overrode the deliberately unused inherited port 19999.
Native automation observed the real App window and rendered chat controls;
WebView2 used the isolated test profile. No Extension launcher/preload was used.
The clean native model picker remained empty: launch/backend handoff does not
register the externally owned model as a native download or select it in chat.

Context/Idle and lifecycle menu dispatch are awaiting human-assisted execution.

## Follow-Up: Default-Agent Output Budget

The earlier no-answer result is reproduced independently in the shipped App
0.2.2 Host/default Harness, with 6540 native prompt tokens at Context 8192.
The bundled pi-ai clamp sent `max_completion_tokens: 1`. A staged candidate
Harness copy using [draft #3684](https://github.com/Niko1221/Strata/pull/3684)
sent 104 instead and returned `OK`: 25 completion tokens, finish reason `stop`.
Model, Context, Medium effort and Manual mode were identical. This is Host API
acceptance; the candidate was not injected into the original installed App or
the portable Extension package. Long agent tasks at 8K remain unverified.

The source-bound [budget report](https://github.com/medking82/Strata/blob/2ee3fbe32aa3c27a0159fb837137231063b1a58b/apps/host/server/agent/output-budget-validation.md)
separates the synthetic 1135-token fixture from the live 104-token cap and lists
the full staging/installer and other-provider gaps. No performance gain is claimed.

## Not Run

Windows reboot/autostart; cross-monitor transitions; other GPU backends;
larger-context allocation/inference; live App/Engine installer, signature or
upgrade/restart; multiple-profile switching and auxiliary menu open/copy actions;
long default-agent tasks at 8K. The bounded run does not certify future App
layouts or a general upstream plugin API. No backend source changed here.

The menu-dispatch steps not confirmed above remain pending.
