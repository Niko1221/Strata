# Windows Tray (opt-in)

A small notification-area client for an **already running** Strata server.
It shows API reachability and model load state, opens the local Web UI, and
copies a short diagnostics report. It does not start, stop, install or update
the server. Closing the Tray leaves the server running.

## Build and run

Requires Windows 10/11 x64 with .NET Framework 4.8. No NuGet packages or .NET SDK
are needed. From the repository root in PowerShell:

```powershell
.\tools\windows-tray\Build.ps1
.\dist\windows-tray\StrataTray.exe
# If the existing server uses another port:
.\dist\windows-tray\StrataTray.exe --port 8080
```

Keep `StrataTray.exe.config` next to the executable; it enables PerMonitorV2
rendering. The default port is 8095, matching `serve/server.py`. Start Strata
with the existing run script as usual. Nothing adds the Tray to startup or to
the installer. One instance is allowed per port in the current Windows session.

Right-click the notification-area icon for the menu; double-click opens the
Web UI. Windows may place new icons in the taskbar overflow. The Tray and executable use the official green layer-stack favicon.
`Strata.ico` is a raster conversion of `serve/web/favicon.svg`, with 16, 20,
24, 32, 40, 48, 64, 128 and 256 px frames; the artwork is unchanged. Status text uses system menu colors and
stays noninteractive. Other menu actions retain standard Windows rendering.

## Menu and daily use

<img src="screenshots/favicon.png" width="32" height="32" alt="Official Strata favicon">

![Local full-featured Tray — visual reference only](screenshots/local-tray-reference.png)

**Visual reference from the author's installed full-featured Tray, not the
read-only client shipped by this PR.** The original PNG shows the shared favicon
and native menu styling more clearly. Model/context switching, idle management,
updates, logs, lifecycle/Web UI status and stopping the server shown here belong
to that separate local integration and are not included in this PR. This PR's
menu contains API/model status, Open Web UI, Refresh status, Copy diagnostics,
and Exit Tray (keep server running). Multi-monitor behavior remains unverified.

1. Start the server with your existing Strata run script, then launch the Tray
   with the same port. Launching the Tray does not start the server.
2. Expand the taskbar notification overflow if needed. Right-click the green
   Strata icon to see status; updates arrive automatically every five seconds.
3. Choose **Refresh status** to check immediately. A running check is not duplicated.
4. Choose **Open Web UI**, or double-click the icon, to open the configured local
   port in your default browser. Sign in there normally if the server requires it.
5. Choose **Copy diagnostics**, then paste into a text editor or issue draft.
   Review the report before sharing. It records the last check and its timestamp.
   If another app holds the clipboard, retry after the notification.
6. Choose **Exit Tray (keep server running)** to remove the icon. Stop the server
   separately with its original launcher when you want to free the model resources.

If the API is unavailable, check that the server is running and the port matches.
`auth required` is expected for a protected API: the Tray deliberately does not
store an API key. A duplicate launch for the same port exits without another icon.

## What the status means

The client sends an anonymous `GET /v1/models` every five seconds, with a
three-second timeout and a 64 KiB response limit. It only connects to
`127.0.0.1`; redirects, proxy use, cookies and default credentials are disabled.

- `available` means a recognized models response was received; it does not
  prove that generation or the Web UI works.
- `loaded` / `unloaded` comes from the model's nested status. An unloaded model
  can load on the next inference request. Alias entries refer to the same model.
- `not listed` means a valid empty model list, without guessing why it is empty.
- `auth required` means HTTP 401/403. This version deliberately does not read or
  send an API key. Use the browser's normal authentication for the Web UI.
- `unavailable`, `timeout`, `invalid response` and `redirect (not followed)` are
  separate failures. They clear any previously displayed model state.

Copy diagnostics includes capture/check timestamps, the loopback port, API/model
state and HTTP status. It omits model names, response bodies, raw exceptions,
credentials, config files, paths and prompts. The clipboard is changed only by
that menu action. The check timestamp identifies the age of the polled snapshot.

## Checks

```powershell
.\tools\windows-tray\Build.ps1 -Test
.\dist\windows-tray\TrayTests.exe
```

The checks use local HTTP fixtures without a GPU or downloads. They cover
loaded/unloaded and alias responses, empty/invalid lists, HTTP failures,
authentication, redirects, response size, cancellation, diagnostics omissions,
noninteractive status rows and actual PerMonitorV2 activation.

For official mock-server compatibility, start this in a separate terminal using
Strata's existing Python environment:

```powershell
python -B -m serve.server --engine mock --host 127.0.0.1 --port 18095
.\dist\windows-tray\TrayTests.exe 18095
```

Before release, manually inspect the menu at the target display scaling, open
the Web UI, copy diagnostics and exit while confirming the server stays running.
The GPU backends, server, web assets and setup defaults are untouched.
