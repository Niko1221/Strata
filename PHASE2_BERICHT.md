# Phase 2: Erweiterungspunkte für UI und Backend (Bericht)

Stand: 02.11.2026, Branch `dev-sandbox`, aufgebaut auf Phase 1 (`abe93a81` … `9c8e0504`).
Alles Neue ist nur mit `STRATA_DEV=1` aktiv. Nichts in `src`, `include`, `serve`, `sycl`, `metal`, `third_party` geändert.
Bedienung und alle Schlüssel: `docs/DEV_BUILD.md`, Abschnitt "Erweitern und Fehlersuche (Phase 2)".

## Was gebaut wurde

**UI (`apps/host/src`)**
- Slot-Registry `src/dev/registry.ts` (reine Logik, ohne React): Slots `sidebar`, `chat.toolbar`, `chat.message.actions`,
  `settings.tabs`, `monitor.widgets`, `preview.panel`; vorhandene Elemente mit festen IDs; ausblenden, umbenennen, umsortieren, ersetzen.
  `dev.config.json` gewinnt gegen Extensions.
- React-Schicht `src/dev/dev.tsx`: `DevElement`, `DevSlot`, `useDevList`, Fehlergrenze pro Extension (mit Komponenten-Stack),
  Extension-Loader, Token-Overrides, Fehlerberichte der Seite an den Host. Inert, bis der Host `dev` meldet.
- Feste IDs in Sidebar (`sidebar.*`), Chat-Toolbar (`chat.toolbar.*`), Composer (`chat.composer.*`), Nachrichten-Aktionen
  (`chat.message.*`), Settings-Tabs (`settings.tabs.*`) und allen Seiten (`screen.*`).
- Extensions aus `dev/extensions/ui` über das virtuelle Vite-Modul `virtual:strata-dev-extensions` (`vite.config.ts`); im Vite-Dev-Server
  lädt eine Änderung/neue Datei die Seite neu. Ohne `STRATA_DEV` ist das Modul ein leeres Objekt.
- Dev-Panel `src/dev/DevPanel.tsx` (lazy, nur in Dev geladen): Extensions + Hooks an/aus mit Fehlern und Reload, UI-Inspektor,
  Logs (Quellen host/engine/harness/hooks/ui, Filter, Pause, Kopieren, Als Datei speichern), Requests (Dauer, Tokens, tok/s, Hook-Zeiten, Fehler).
  Öffnen: Klick auf das DEV-Badge oder Strg+Umschalt+D.
- Beispiel-Extension `dev/extensions/ui/info-button/index.tsx`: Info-Button in der Chat-Toolbar.
- Overrides für `/loop` (an/aus, maxRuns) und Standardmodus in `src/dev/overrides.ts` (ohne Imports, damit `store.ts` die Dev-UI nicht mitzieht).

**Backend (`apps/host/server`)**
- `dev-hooks.mjs`: Hooks `before_request`, `on_token`, `after_response`, `on_model_load` (JS-Module) und `.py`-Hooks als Kindprozess
  mit JSON über stdin/stdout; eigene Routen unter `/dev/<name>/...`; Fehlergrenze (Wurf, kein JSON, Zeitlimit -> Hook aus + Log).
- `agent/plugins/strata-dev-hooks.mjs`: `on_tool_call` als Harness-Plugin, eingefügt nach `strata-gate` (nur in Dev, `agent/profile.mjs`).
- `dev-log.mjs`: Log-Puffer pro Quelle (Konsole des Hosts, Harness-Ausgabe aus `harness-runtime.mjs`, Hooks, UI-Fehler);
  Engine-Zeilen kommen aus den vorhandenen Puffern von `supervisor.mjs` und Control.
- `dev-api.mjs`: `/api/dev/state|logs|requests|debug|reload|hooks|client-error`; Requests aus dem vorhandenen Serving-Log (`serving.mjs`).
- `dev-mode.mjs`: `devSettings()` (logging.verbose, tokens, ui, hooks, overrides), `devChatMode`, `devToolsSettings`, `devExtensionsDir`.
- Overrides per `dev.config.json`: Websuche (an/aus, Anbieter, ohne Nachfrage) im Agent-Profil, Rechtefreigabe (Standardmodus,
  erlaubte Modi) im Profil und beim Chat-Turn, Ordner-Modus (an/aus, Standardordner) beim Turn und bei `/api/chat/folder`, `/loop` in der UI.
- Ausführliches Logging (`logging.verbose`): jede Host-Anfrage mit Status und Dauer.
- `ports.mjs`: Liegt der Host-Port in einem von Windows reservierten Bereich (`netsh … excludedportrange`), nennt Dev den Bereich.
- Host-Neustart bei Codeänderung: `dev-build.ps1 -HostOnly` / `npm run dev` startet den Host schon mit `node --watch` (seit Phase 1 in
  `scripts/dev.mjs`); Hooks lassen sich zusätzlich ohne Neustart per Reload neu laden.

**Build und Werkzeuge**
- `vite.config.ts`: mit `STRATA_DEV=1` React-Entwicklungsversion (Fehler mit Komponenten-Stack), unminifiziert, Source Maps (waren schon an),
  `/dev/`-Proxy, `resolve.dedupe` für React in Extensions. Ohne Schalter dieselben Werte wie vorher.
- `dev-build.ps1`: baut die UI für die Dev-App mit `STRATA_DEV=1` (nur für diesen Schritt).
- `/dev-debug`: Vorlage `dev/claude-commands/dev-debug.md` (holt `GET /api/dev/debug`: letzter Fehler mit Stack, Hooks, fehlgeschlagene
  Requests, Fehler/Warnungen und letzte Logzeilen aller Quellen).

## Geänderte und neue Dateien

Geändert: `apps/host/server/{index.mjs, dev-mode.mjs, ports.mjs, agent/profile.mjs, agent/harness-runtime.mjs}`,
`apps/host/src/{app/App.tsx, app/Sidebar.tsx, components/ArtifactPanel.tsx, components/ChatToolbar.tsx, components/ComposerControls.tsx,
components/DevBadge.tsx, lib/app-mode.ts, lib/loop-runner.ts, lib/store.ts, screens/Chat.tsx, screens/Monitor.tsx, screens/Settings.tsx}`,
`apps/host/vite.config.ts`, `apps/host/tests/dev-mode.test.mjs`, `dev-build.ps1`, `dev.config.json`, `dev/README.md`, `docs/DEV_BUILD.md`.

Neu: `apps/host/server/{dev-hooks.mjs, dev-api.mjs, dev-log.mjs, agent/plugins/strata-dev-hooks.mjs}`,
`apps/host/src/dev/{registry.ts, dev.tsx, DevPanel.tsx, overrides.ts, dev.css, virtual.d.ts}`,
`apps/host/tests/{dev-registry.test.mjs, dev-hooks.test.mjs}`, `dev/extensions/ui/info-button/index.tsx`,
`dev/extensions/backend/{example.mjs, _example_python.py}`, `dev/claude-commands/dev-debug.md`, `PHASE2_BERICHT.md`.
Entfernt: die zwei `.gitkeep` in `dev/extensions/*` (die Ordner haben jetzt Inhalt).

## Wie Live-Gleichheit geprüft wurde

1. **Code-Pfade**: Jede Änderung an bestehendem Code hängt an `STRATA_DEV=1` (Host: `DEV`/`devHooks`/`DEV_SETTINGS` sind `null`;
   Profil: `activeDevSettings()` ist `null`; UI: `useDev()` ist `null`, bis `/api/app/mode` ein `dev` liefert, was nur ein Dev-Host tut).
   Ohne Dev gibt `DevElement` seine Kinder unverändert zurück, `DevSlot` nichts, `useDevList` die eigene Liste; `devChatMode`,
   `devModeOverride`, `devLoopMaxRuns`, `devLoopEnabled` geben die Live-Werte zurück. Vite-Konfiguration ohne Schalter: dieselben Werte.
2. **Tests** (siehe unten), darunter: Agent-Profil ohne Dev enthält weder Plugin noch Override (`"defaultPreset": "strata-manual"`,
   Websuche drin); Log-Puffer ohne Dev nimmt nichts an; Overrides ändern ohne Dev nichts.
3. **Gerenderter DOM, alt gegen neu**: UI von `HEAD` (vor Phase 2, temporärer Worktree) und vom neuen Stand jeweils ohne `STRATA_DEV`
   gebaut, beide Hosts ohne Dev gestartet (isolierte Scratch-Datenordner, Agent aus), sieben Seiten in Chrome (headless) geöffnet und
   `document.body.innerHTML` verglichen: Chat, Skills, Tasks, Serving und Settings-Dialog gleich; Models und Monitor unterscheiden sich
   nur in Umgebungswerten (Checkout-Pfad in einer Fehlermeldung "kein Strata-Ordner", schwankender GPU-Takt), nicht im Aufbau.
   `/api/app/mode` beider Hosts gleich; `/api/dev/state` ohne Dev: 404 (wie vorher).
4. **Bundle**: Der Live-Build enthält die Beispiel-Extension nicht (leeres virtuelles Modul); er enthält den (inaktiven) Dev-Code der
   Registry. Das Dev-Panel ist ein eigener Chunk, der nur in Dev geladen wird. Der JS-Code ist damit nicht byte-gleich, das Verhalten schon.

Live-App und `C:\Users\jonat\Documents\Strata` wurden nicht verändert (geprüft: keine Datei dort in der Testzeit geändert); alle
Testläufe nutzten Scratch-Ordner und eigene Ports (18792, 18811–18814). Modelle wurden nur gelesen, kein Lasttest.

## Testergebnisse

- `npm test` in `apps/host`: **1281 Tests, 1178 bestanden, 0 fehlgeschlagen, 103 übersprungen** (vorher 1259/1156/0/103; neu sind
  22 Tests in `dev-registry.test.mjs` und `dev-hooks.test.mjs`: Registry, Overrides, Hooks inkl. Python-Hook, Zeitlimit, Fehlergrenze,
  on_token-Budget, Routen, on_model_load, on_tool_call-Plugin, Log-Puffer, Debug-Bericht, Profil mit/ohne Dev, Portbereiche).
  Ein bestehender Test (`card-labels.test.mjs`) prüft die Modus-Zeile in `store.ts` wörtlich; sie ist unverändert geblieben,
  der Dev-Standardmodus kommt als zusätzlicher Spread dahinter.
- `npm run typecheck`: sauber.
- `python -m pytest tests/terminal`: **2038 bestanden, 6 fehlgeschlagen, 62 übersprungen**. Der System-Python hat die Abhängigkeiten
  nicht (rich, textual, jinja2 …), daher lief es in einer temporären venv im Scratch-Ordner mit `requirements.txt` +
  `requirements-monitor.txt` + pytest. Die 6 Fehler (`test_bundle_imports` 1, `test_launch_dsv41` 4, `test_launch_tui::FolderStep` 1)
  treten **genauso auf dem Stand vor Phase 2 (HEAD) auf** (in einem temporären Worktree nachgeprüft); sie hängen an der echten
  Hardware/Platte dieses PCs (z. B. erwartet ein Test "4 cards", hier steckt eine RTX 5070). Phase 2 ändert keinen Python-Code.

## Prüfung im Dev-Modus

Host mit `STRATA_DEV=1` und der Dev-UI (`STRATA_DEV=1 vite build`) in Chrome (headless), Scratch-Datenordner:
DEV-Badge da; Info-Button der Beispiel-Extension in der Chat-Toolbar, Klick zeigt den Toast; Dev-Panel öffnet über das Badge,
Reiter Extensions zeigt `info-button` (geladen, `chat.toolbar.info`) und `example.mjs` (Events, Route `/dev/example/ping`);
Logs zeigen Zeilen; Inspektor markiert beim Überfahren `chat.toolbar.options`; keine Seitenfehler.
Mit einer Test-`dev.config.json`: `sidebar.monitor` ausgeblendet, `sidebar.models` als "Modelle" an erster Stelle, `settings.tabs.engine`
ausgeblendet, `--st-accent` = `#e0218a`. Host-Smoke-Test: `/dev/example/ping` antwortet, ein Chat-Request läuft durch
`before_request`/`after_response`, `/api/dev/requests` zeigt die Hook-Zeiten.

**Dev-App gebaut** mit `dev-build.ps1 -NoStart` (Rust-Build 3 min 13 s): Installer
`apps\desktop\src-tauri\target\release\bundle\nsis\Strata Dev_0.2.9_x64-setup.exe` (105,6 MiB), nicht installiert. Die gebaute
`strata-dev.exe` einmal kurz gestartet: Host auf 47832 meldet `dev` (Label DEV, Config aus dem Repo), Hook `example.mjs` geladen,
`/dev/example/ping` antwortet, das UI-Bundle enthält den Dev-Panel-Chunk und React in der Entwicklungsversion. Danach wieder beendet.
Live (`%LOCALAPPDATA%\Strata`, Ports 47831/8791/8080) wurde dabei nicht berührt.

## Offene Punkte

- **`/dev-debug` liegt als Vorlage in `dev/claude-commands/dev-debug.md`.** Das Anlegen unter `.claude/commands/` hat der Auto-Modus von
  Claude Code als Selbständerung der Claude-Konfiguration blockiert. Einmal kopieren:
  `copy dev\claude-commands\dev-debug.md .claude\commands\dev-debug.md` (oder mir die Freigabe geben).
- Die installierte Dev-App enthält die UI-Extensions aus dem Stand ihres Builds; Hot Reload gibt es mit `dev-build.ps1 -HostOnly`.
- Python-Hooks laufen pro Aufruf als eigener Prozess (Startzeit ~50–150 ms): gut für `before_request`/`after_response`/`on_tool_call`,
  für `on_token` nicht erlaubt.
- Websuche- und Rechte-Overrides gelten ab dem nächsten Agent-Start (das Profil wird dann neu geschrieben); `on_tool_call` lädt die Hooks
  beim Agent-Start, Reload im Panel wirkt dafür erst nach einem Agent-Neustart.
- Eine synchrone Endlosschleife in einem JS-Hook kann nicht abgebrochen werden (Grenze von JavaScript); Zeitlimits greifen für
  asynchrone Hooks und messen synchrone (`on_token`) danach.
- `npm run check` (Audit-/Kontrast-Skripte mit Browser) wurde nicht ausgeführt; `npm run typecheck` ist sauber.
- `apps/host/dist` (nicht versioniert) enthält jetzt die Dev-UI aus dem Dev-App-Build. Ein Live-Build baut sie mit `npm run build`
  ohne `STRATA_DEV` ohnehin neu.
- Die 6 vorhandenen Python-Testfehler (hardwareabhängig, schon vor Phase 2) sind nicht Teil dieser Phase.
