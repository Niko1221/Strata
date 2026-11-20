# Native Menu Examples

These PNGs are **simulated menu states**, not desktop screenshots. The fixture
constructs the same `ManagedMenu` class used by production `ManagedTray`, applies
anonymous snapshot data, and renders the actual WinForms menu and dropdowns
using `ToolStrip.DrawToBitmap`. Submenus are placed beside the main menu for
readability, without a pointer/hover state. Captions are outside the controls.
The production handlers are left unbound: no Tray icon, controller, server,
App, browser, updater, model or inference is started. No private configuration,
paths, credentials, model files, user details or network responses are read.

| Example | Fixture State |
| --- | --- |
| [Managed Menu](managed-menu.png) | Anonymous Example Profile; unloaded; selected/effective 128K and 30 minutes; automatic upgrades off |
| [Context](context-menu.png) | All eight official choices; 128K selected |
| [Idle](idle-menu.png) | Never/5/10/30 Minutes; 30 Minutes selected |
| [Engine Updates](engine-update-menu.png) | Official component updater required; Tray installation disabled |
| [App Updates](app-update-menu.png) | Simulated App v0.1.0 with v0.2.0-beta available; installation remains in the App |

The caption uses the exact embedded official `Strata.ico` asset, identical to
`apps/desktop/src-tauri/icons/icon.ico` in this source revision. Its SHA-256 is
`7121314b01445e61127f078509c60cbb7f0c05d92716f9d65b9b39523503e36e`.

To reproduce from the repository root on Windows with .NET Framework 4.8:

```powershell
.\tools\windows-tray\Build.ps1 -Examples
.\dist\windows-tray\TrayMenuExamples.exe --out "$PWD\tools\windows-tray\screenshots"
```

The fixture verifies native layout and nonblank output. Menu-state regression
checks run with `Build.ps1 -Test` and `TrayTests.exe`. Pixel dimensions/colors
can vary with Windows DPI, fonts and system theme; the fixture data is fixed.
These examples do not establish hardware capacity, real loading/unloading,
server health, update installation, taskbar placement or multi-monitor behavior.
