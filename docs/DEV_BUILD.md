# Strata Dev

Strata Dev ist die Entwicklerversion von Strata. Sie läuft neben dem normalen Strata (hier "Live") und berührt es nicht:
eigener Name, eigene Ports, eigene Datenordner, und sie aktualisiert sich nie selbst. Ohne den Dev-Schalter baut und läuft die App genau wie bisher.

Zielsystem: Windows. Die Skripte sind PowerShell, jeweils mit einer `.bat` zum Doppelklicken.

## Was anders ist als Live

| | Live | Dev |
| --- | --- | --- |
| Name, Identifier | Strata, `ai.strata.desktop` | Strata Dev, `ai.strata.desktop.dev` |
| Programm | `strata-desktop.exe` in `%LOCALAPPDATA%\Programs\Strata` | `strata-dev.exe` in `%LOCALAPPDATA%\Programs\Strata Dev` |
| Fenster und Host | Port 47831 | Port 47832 |
| Host im Entwicklermodus (`npm run dev`) | 8791, Vite 5199 | 8792, Vite 5200 |
| Engine/Server | 8080 (llama.cpp), ab 18500 (Strata-Engine) | ab 8081 |
| App-Update | an | aus (der Update-Feed ist der von Live) |
| Badge | keins | "DEV" neben dem Logo |

So wird Dev eingeschaltet:

- die Tauri-Config `apps/desktop/src-tauri/tauri.dev-build.conf.json` (Name, Identifier, Programmname), als Overlay wie `tauri.release.conf.json`;
- das Cargo-Feature `dev-build` (Port, Fenstertitel, kein Updater, Dev-Datenordner: `src-tauri/src/devbuild.rs`);
- `STRATA_DEV=1` für den Host. Das setzt die Dev-App selbst.

## Wo Dev seine Daten ablegt

Alles unter `%LOCALAPPDATA%\StrataDev`. Die Dev-App setzt diese Variablen beim Start; nicht von Hand setzen.

| Was | Variable | Ordner |
| --- | --- | --- |
| Chats, Host-State, Skills, Agent | `STRATA_HOME` | `StrataDev\home` |
| Profile und Presets | `STRATA_PROFILE_DIR` | `StrataDev\profiles` |
| Tuning, Time-skip | `STRATA_TUNE_DIR`, `STRATA_TIMESKIP_DIR` | `StrataDev\tuning`, `StrataDev\time-skip` |
| Übriger State (Engine-Einträge, Runtime) | `STRATA_STATE_DIR`, `STRATA_RUNTIME_DIR` | `StrataDev\state`, `StrataDev\runtime` |
| Desktop-Logs, Fenster-Einstellungen | `STRATA_DESKTOP_DATA_DIR` | `StrataDev\desktop` |
| App-Einstellungen, Run-Configs | `STRATA_SETTINGS_FILE`, `STRATA_INSTALL_DIR` | `StrataDev\app-settings.json`, `StrataDev\install` |
| Datenordner-Notiz für das Terminal | `STRATA_TERMINAL_SETTINGS_FILE` | `StrataDev\terminal-settings.json` |
| Standard-Ordner für Chats | `STRATA_WORKSPACE` | `StrataDev\workspace` |
| Browser-Speicher des Fensters | (WebView2, pro Identifier) | `%LOCALAPPDATA%\ai.strata.desktop.dev` |

Dazu `STRATA_ENGINE_PORT=8081` und `STRATA_NO_REPAIR_ACCESS=1` (die Rechte-Reparatur beim Start geht sonst durch die Ordner von Live).

Mit Live geteilt, damit nichts doppelt heruntergeladen wird:

- die Engine-Komponenten in `%LOCALAPPDATA%\Strata\components`;
- die Modelldateien (zum Beispiel `Documents\Strata\Models`).

Dev liest diese nur. Ausnahme: Ein Modell oder eine Engine, die du in Dev herunterlädst, landet im selben geteilten Ordner.

## Einstellungen: `dev.config.json`

`dev.config.json` im Repo-Ordner enthält die Schalter von Dev. Der Host liest sie einmal beim Start, nur in Dev. Eine kaputte Datei
hält nichts an: Der Host schreibt eine Warnung ins Log und startet ohne die Datei; das DEV-Badge wird rot und zeigt die Warnung beim Überfahren.

```json
{
  "version": 1,
  "badge": { "label": "DEV" },
  "features": {},
  "overrides": {}
}
```

`features` und `overrides` werden in den nächsten Phasen gefüllt. Extensions, Skills und Modelle für Dev kommen nach `dev/` (siehe `dev/README.md`).

## Bauen und starten

Du brauchst: Node.js 24.12 oder neuer, Rust (rustup) und die Bundle-Teile in `apps\desktop\src-tauri\resources` (node, control, harness).
Das Skript prüft das zuerst. Fehlt etwas, sagt es was und wie man es bekommt, und baut nichts. NSIS für den Installer holt Tauri beim ersten Build selbst.

```bat
dev-build.bat                      :: UI und Dev-App bauen, Dev-App aus target\release starten
dev-build.bat -Install             :: zusätzlich still nach %LOCALAPPDATA%\Programs\Strata Dev installieren und von dort starten
dev-build.bat -NoStart             :: nur bauen
dev-build.bat -HostOnly            :: nur Host + UI im Entwicklermodus (Host :8792, UI http://127.0.0.1:5200), ohne Rust-Build
dev-build.bat -ResourcesFromLive   :: fehlende Bundle-Teile node und harness aus dem installierten Strata kopieren (es wird nur gelesen)
```

Control (`resources\control`) kommt immer aus dem Repo: `python apps\desktop\scripts\stage-control.py`. Ein aus Live kopiertes
Control kann älter sein und kennt die Dev-Schalter nicht; dann landen die Prozesseinträge der Dev-Engine in `%LOCALAPPDATA%\Strata\procs`
(so beim ersten Test am 01.11.2026 mit Live 0.2.7). Bricht der Download mit einem SSL-Fehler ab, hilft `SSL_CERT_FILE` mit dem Zertifikats-Bundle von certifi.

Der Installer liegt danach in `apps\desktop\src-tauri\target\release\bundle\nsis\Strata Dev_<Version>_x64-setup.exe`.

Derselbe Build von Hand:

```bat
npm --prefix apps\host run build
cd apps\desktop
npx tauri build --config src-tauri\tauri.dev-build.conf.json --features dev-build --bundles nsis
```

## Zurücksetzen

```bat
dev-reset.bat
```

Zeigt, was gelöscht wird, und fragt vorher. Gelöscht werden nur `%LOCALAPPDATA%\StrataDev` und der Browser-Speicher des Dev-Fensters
`%LOCALAPPDATA%\ai.strata.desktop.dev`. Live, Modelle und Engine-Komponenten bleiben. Strata Dev vorher schließen.

Wer Strata Dev deinstalliert und "Also remove my Strata data" ankreuzt, löscht nur `%LOCALAPPDATA%\StrataDev`, nie die Ordner von Live.
