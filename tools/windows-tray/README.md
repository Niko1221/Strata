# Optional Windows Tray

Windows 10/11 x64 and .NET Framework 4.8 are required. The Tray uses the official
favicon, PerMonitorV2 scaling and system menu colors. It is not installed or
started by the default Strata installer.

## Build

From the repository root in PowerShell:

```powershell
.\tools\windows-tray\Build.ps1
```

Keep the entire `dist/windows-tray` directory together, including
`StrataTray.exe.config` and `managed/`. No .NET SDK or NuGet restore is needed.
Right-click the notification-area icon to open the menu; double-click opens the
Web UI. Windows may place the icon in the taskbar overflow.

## Use an existing server

```powershell
.\dist\windows-tray\StrataTray.exe --port 8095
```

This anonymous read-only mode keeps its previous behavior: API/model status,
Open Web UI, Refresh status, Copy diagnostics, and Exit Tray (keep server
running). It never manages that server. A protected API reports `auth required`.
Only one read-only instance per port is allowed in a Windows session.

## Let the Tray manage Strata

1. Finish the normal Strata setup first. Keep its original model JSON files.
2. Install the optional sidecar dependency in that Python environment:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r tools/windows-tray/managed/requirements.txt
   ```

3. Copy `tools/windows-tray/tray.example.json` to
   `tools/windows-tray/tray.json`. Set `python`, `repository`, and each entry in
   `profiles` to your Python executable, official checkout and existing model
   JSON respectively. Relative paths are resolved from the Tray JSON directory.
   The example model filename is a placeholder; use the JSON made by your setup.
4. Stop any separate launcher on the chosen ports. `port` is the stable public
   **loopback-only** endpoint (default 8095); `upstream_port` is the distinct
   private loopback port (default 8096). Occupied ports are refused, never adopted.
5. Launch:

   ```powershell
   .\dist\windows-tray\StrataTray.exe --config tools/windows-tray/tray.json
   ```

Managed mode requires Python 3.10+ and the official server's Python dependencies.
It starts the API with `--lazy`, leaving the chosen model unloaded until the
first inference request or an explicit **Preload model** action. Boot, restart,
profile/context changes and upgrades do not automatically preload. Status polling
does not load a model. Idle unload defaults to five minutes when absent; existing
profile values and saved choices (including thirty minutes) remain authoritative. A Windows Job contains only the owned
sidecar/server descendants; closing the owning Tray forcibly during inference
can interrupt that inference. Use the menu's graceful Exit for normal shutdown.
Do not launch a second Tray against the same config/state directory.

Every profile needs `model_name`, string `args` containing exactly one
`--max-context` plus its value, and an existing `exe` for the Strata engine.
Other official settings and model paths are preserved. Profiles must share the
same authentication and origin policy. A policy edit requires restarting the
Tray. Credentials stay in the original profile/environment; the proxy forwards
client authentication unchanged and never supplies its control credential to
an anonymous inference request.

## Menu guide

- **Model profile**, **Context** (64k/128k/256k), and **Unload after idle**
  (never/5/10/30 minutes) persist the selection. The current profile's other
  context/idle values are preserved until a choice is made. Selected and
  effective values are shown separately; an error is not reported as applied.
- **Start server**, **Apply configuration / Restart**, **Stop server**,
  **Preload model**, and **Unload model now** control the owned server. Changes
  close admission, wait for admitted requests/streams and official activity to
  drain, then act. New requests receive HTTP 503 during this boundary. A drain
  timeout fails the action without killing an active request.
- **Open configuration** opens the selected profile's original JSON using its
  default Windows application. Save the file, then choose **Apply configuration
  / Restart**. Saving alone does not restart inference. Missing files disable
  the action. **Open Tray configuration** opens the profile-list/port settings;
  those settings take effect on the next Tray launch. If the saved selection no longer
  matches the profile list, the next launch resets selections to the first profile
  and the configured defaults.
- **Open Web UI** opens the stable loopback endpoint in your browser. Its
  anonymous reachability status does not verify a signed-in session or generation.
- **Open logs** opens the adjacent `<tray-config-stem>.state/logs` directory.
  Logs are local and can contain private server information; inspect before sharing.
- **Copy diagnostics** copies an allowlisted snapshot: timestamps, lifecycle,
  API/model state, active requests, context/idle selections and update state.
  Paths, profile labels, model IDs, credentials, prompts and raw errors are omitted.
- **Exit Tray and stop server** drains requests, then closes the owned processes.
  It does not stop a separately launched server.

Selections, generated runtime config and update records live in
`<tray-config-stem>.state` beside the Tray JSON. Keep it private: the generated
profile retains the original API key. Original profile files are never rewritten.
Do not place this directory in a public repository or a shared writable folder.

## Updates

Managed mode checks the official stable GitHub release on startup and daily,
with a manual **Check now** action. Checking does not install anything.
Automatic installation is opt-in (`automatic_upgrade: false` by default).
To enable installation, add the asset matching your installed official engine:

| Backend | `update_asset` |
| --- | --- |
| CUDA 13 | `strata-windows-x64.zip` |
| CUDA 12 | `strata-windows-x64-cuda12.zip` |
| Windows HIP | `strata-windows-x64-hip.zip` |

Choose **Install available update**, or enable **Automatically upgrade when
idle**. Automatic activation waits for at least 60 seconds of idle time, then
uses the same admission/drain boundary. The updater stages a new runtime without
changing your Git checkout or overwriting the previous runtime. It requires the
published asset size and SHA-256, rejects unsafe ZIP entries, checks backend
metadata and unchanged Python requirements, and verifies API model identity,
context metadata and engine version before recording the new active runtime.
The API stays unloaded: context before first use is configured capacity, not
proof of native GPU allocation. Runtime version is independent of Git source pins. Source
comes from the fixed official GitHub tag URL over HTTPS; its hash is recorded.
These checks trust the official release publisher; they are not code signing.

Changed Python requirements, unknown backend metadata, custom vision/library
profiles, unavailable assets and incompatible runtime families require a manual
update using official setup. Locally compiled/SYCL engines are not rebuilt by
the Tray. An unrecognized engine version is shown as unknown and prevents automatic
version comparison, but does not prevent normal startup. Failures disable automatic
installation and show an error; there is
no silent rollback or repeated activation loop. Previous files are retained.
Use local logs to resolve the cause, then explicitly restart or install again.
Tray UI code itself is updated by rebuilding this directory, not by replacing a
running executable. No login gateway, remote host or machine-specific model
configuration is included.

## Screenshots and verification

<img src="screenshots/favicon.png" width="32" height="32" alt="Official Strata favicon">

![Earlier installed Tray, visual reference](screenshots/local-tray-reference.png)

This original PNG is a visual reference from the author's earlier local Tray;
its model names and layout are not a screenshot of this generic candidate.
The candidate uses the same favicon and native styling. Multi-monitor transitions
and a real GPU upgrade need hardware acceptance; mock tests do not establish them.

```powershell
.\tools\windows-tray\Build.ps1 -Test
.\dist\windows-tray\TrayTests.exe
.\.venv\Scripts\python.exe -B tools/windows-tray/managed/test_managed.py
```

The managed suite uses disposable loopback ports, the unmodified official mock
server and a streaming fixture. No model download or GPU load is required.
MockEngine always reports loaded and a fixed 32768 context; tests do not claim
real GPU unload or allocation coverage. Existing engine, server, frontend and
setup source files are unchanged by this feature.
