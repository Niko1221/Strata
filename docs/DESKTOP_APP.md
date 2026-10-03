# Windows desktop app

The optional desktop client hosts Strata's existing web app in a dedicated Windows window using Microsoft
WebView2. It uses the Python environment, model configurations and saved chats in your Strata project. The app
does not bundle model weights or replace the inference engine.

## Requirements and build

First complete [Strata setup](INSTALL.md) and prepare a native model. The desktop target is Windows x64 and needs
the Microsoft WebView2 Runtime. The build uses a project-local .NET 10 SDK; the published app includes its .NET
runtime, so users do not need to install that runtime globally. The build helper needs Python 3.11 or newer.

From the Strata project folder, run:

```powershell
.venv\Scripts\python.exe tools\build_desktop.py
powershell -NoProfile -File tools\install_desktop.ps1
```

The build helper downloads the pinned Microsoft SDK from Microsoft's release metadata, verifies its SHA512 and
caches it under `.local/desktop-build/`. It also restores the Microsoft WebView2 NuGet package. These are build
dependencies, separate from Strata's model downloads. The resulting application is `dist/Strata/Strata.exe`.

The shortcut installer creates Strata shortcuts on the desktop and in the Start menu. Existing shortcuts with
those names are backed up under `.local/desktop-shortcut-backups/`. Shortcuts point to the project and contain no
API key. Close the desktop app before rebuilding it; the model server can keep running.

## Opening and using the app

Open a Strata shortcut or run `START-App.bat`. You can also specify the project explicitly:

```powershell
.\dist\Strata\Strata.exe --root "C:\Projects\Strata"
```

If a Strata server is already running at `http://127.0.0.1:8080`, the app connects without changing its model. If
the server is stopped, the app starts the local launcher without opening another browser. It remembers the last
confirmed native configuration; otherwise it uses the launcher's installed default. Native weights load when a
request needs them, so the first answer can take longer.

The Chat page has the same native and additional model controls, history, drafts, compaction and optional recall
as the browser. The app shares `Strata-data/chat-history/history.sqlite3`; it does not create or move conversations
into a separate database. Its WebView profile, unsaved-draft recovery state, model preference and status file live
under `.local/desktop/`. Browser-specific display and thinking preferences do not migrate automatically.

Only one normal desktop window opens per project. Opening another shortcut restores that window. The tray control
hides it, and double-clicking the Strata tray icon brings it back. Reconnect checks the server again and starts it
if needed.

Closing the window or choosing the tray's exit action closes the app while leaving the server available to other
clients. To release model memory, wait for generation to finish and use `STOP-Models.bat`. The command below closes
only the desktop app:

```powershell
.\dist\Strata\Strata.exe --root "C:\Projects\Strata" --quit-app
```

## Optional API authentication

No API key is needed when the local server is unprotected. If the server requires a key, the desktop host can read
it from an explicitly supplied `--api-key-file`, the project's `.secrets/strata-api-key.txt`, the installed native
configuration or `STRATA_API_KEY`. Files take precedence, followed by the environment, then the installed configuration.
Use the same key configured on the server. Keep key files and machine-specific
configurations outside version control.

An explicit file avoids putting the key itself on the command line:

```powershell
.\dist\Strata\Strata.exe --root "C:\Projects\Strata" --api-key-file "C:\Private\strata-api-key.txt"
```

The host adds authentication only to protected requests sent to `http://127.0.0.1:8080`. It does not pass the key
to the page's JavaScript, browser storage, URL, shortcuts or diagnostic output. The API key input is hidden in the
desktop page because the host manages authentication. In a normal browser, the existing key entry remains available
under About and is stored in that browser when saved.

WebView requests and navigation are restricted to the local Strata origin, with support for the app's data and
local blob image attachments. External links clicked by the user open in the normal browser. External requests
inside WebView are blocked. WebView host objects, password autosave and automatic permission grants are disabled.
The app does not configure remote access or expose a listening service beyond loopback.

## Status and developer checks

The app writes status and scalar counters to `.local/desktop/status.json`, without raw chat contents or key values.
Opening a window alone does not prove that a model can generate a response. The origin and request policy can be
checked without loading a model:

```powershell
.\dist\Strata\Strata.exe --policy-test
```

For an integration check, `--diagnostics` accepts a directory inside the project's `.local/`. It uses a temporary
WebView profile, checks local endpoints, sends a short synthetic prompt, saves an About-page preview and exits.
This check requires a working local server and model and may load model weights. It does not publish or export
your chat history.

```powershell
.\dist\Strata\Strata.exe --root "C:\Projects\Strata" --diagnostics "C:\Projects\Strata\.local\desktop-check"
```

No inference speed improvement is claimed for the desktop wrapper. Response speed still depends on the model,
hardware and inference settings.

## Removing the desktop client

Close the app, remove its shortcuts and remove `dist/Strata/`. If shortcuts existed before installation, restore
the saved copies from `.local/desktop-shortcut-backups/`. Keep the model files, Python environment, configurations,
secrets and chat database to continue using Strata in a browser.

References: [Microsoft WebView2](https://learn.microsoft.com/en-us/microsoft-edge/webview2/) and
[WebView2 request handling](https://learn.microsoft.com/en-us/microsoft-edge/webview2/how-to/webresourcerequested).
