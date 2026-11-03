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

`features` ist noch frei; was ab Phase 2 dazukommt, steht unten unter "Erweitern und Fehlersuche". Extensions, Skills und Modelle für Dev kommen nach `dev/` (siehe `dev/README.md`).

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

## Erweitern und Fehlersuche (Phase 2)

Alles in diesem Abschnitt gibt es nur mit `STRATA_DEV=1`. Ohne den Schalter rendert die App wie Live, und `/api/dev/...` sowie `/dev/...` gibt es nicht.

### Dev-Panel

Klick auf das DEV-Badge oder `Strg+Umschalt+D`. Reiter:

- **Extensions**: UI-Extensions und Backend-Hooks, jeweils an/aus, mit Fehlern pro Extension/Hook und Laufzeiten. "Reload" liest
  `dev.config.json` und die Hooks neu (ohne Host-Neustart) und lädt die Seite neu.
- **Inspector**: zeigt beim Überfahren die feste ID eines Elements; Klick kopiert sie. Darunter alle IDs der aktuellen Seite.
- **Logs**: Host, Engine (Puffer aus `supervisor.mjs` und Control), Harness, Hooks und UI-Fehler. Filter, Pause, "Copy", "Save as file".
- **Requests**: jede Anfrage aus dem Serving-Log (`serving.mjs`) mit Dauer, Tokens, tok/s, Fehler und der Zeit jedes Hooks.

### Feste IDs und Slots

Vorhandene Elemente (Auswahl): `sidebar.new-chat`, `sidebar.tasks`, `sidebar.workspaces`, `sidebar.engine`, `sidebar.skills`,
`sidebar.monitor`, `sidebar.models`, `sidebar.more`, `sidebar.more.serving`, `sidebar.more.settings`, `sidebar.theme`,
`chat.toolbar.model`, `chat.toolbar.subagents`, `chat.toolbar.options`, `chat.toolbar.previews`, `chat.composer.folder`,
`chat.composer.controls`, `chat.composer.send`, `chat.message.copy`, `chat.message.rewind`, `chat.message.retry`, `chat.message.more`,
`settings.tabs.appearance` … `settings.tabs.about`, `screen.chat`, `screen.models`, `screen.serving`, `screen.monitor`, `screen.tasks`,
`screen.skills`. Der Inspector zeigt sie.

Slots für neue Elemente: `sidebar`, `chat.toolbar`, `chat.message.actions`, `settings.tabs`, `monitor.widgets`, `preview.panel`.
Code: `apps/host/src/dev/` (`registry.ts` Logik, `dev.tsx` React-Teil, `DevPanel.tsx`).

### UI-Extensions (`dev/extensions/ui/`)

Eine Extension ist `dev/extensions/ui/<name>/index.tsx` (oder `.ts/.jsx/.js`, oder eine einzelne Datei `<name>.tsx`). Namen mit `_`
am Anfang werden übersprungen. Sie exportiert eine Setup-Funktion:

```tsx
export default function setup(strata) {
  strata.slot('chat.toolbar', { id: 'info', render: ({ sessionId }) => <strata.ui.IconBtn name="info" title="Info" onClick={() => strata.toast('Hallo', sessionId)} /> });
  strata.hide('sidebar.monitor');              // ausblenden
  strata.rename('sidebar.models', 'Modelle');  // umbenennen
  strata.move('sidebar.models', 1);            // umsortieren (kleiner = weiter vorn; die App nutzt 10, 20, 30 ...)
  strata.replace('screen.monitor', MyMonitor); // ersetzen; die Komponente bekommt das Original als children
}
```

Jede Extension läuft in einer eigenen React-Fehlergrenze: Ein Absturz zeigt eine kleine Fehlermarke, der Rest der App läuft weiter,
und der Fehler (mit Komponenten-Stack) steht im Dev-Panel und im Log. Mit `dev-build.bat -HostOnly` lädt Vite die Seite bei jeder
Änderung an einer Extension neu. Die installierte Dev-App enthält die Extensions aus dem Stand ihres Builds (`dev-build.ps1` baut die
UI dafür mit `STRATA_DEV=1`: Extensions eingebaut, React-Entwicklungsversion, unminifiziert, Source Maps).

Beispiel: `dev/extensions/ui/info-button/index.tsx` (Info-Button in der Chat-Toolbar).

### Backend-Hooks (`dev/extensions/backend/`)

Ein Hook ist ein JS-Modul (`.mjs`/`.js`) oder ein Python-Skript (`.py`). Beschreibung aller Exporte: Kopf von
`apps/host/server/dev-hooks.mjs`; Beispiele: `example.mjs` und `_example_python.py` (ohne `_` umbenennen, um ihn zu laden).

| Hook | Wann | Darf ändern |
| --- | --- | --- |
| `before_request(body)` | vor jeder Chat-Anfrage (`/api/chat/stream`) | ja: ein zurückgegebenes Objekt ersetzt die Anfrage |
| `on_token(frame)` | jeder gestreamte Text-Frame, synchron | ja: ein zurückgegebenes Frame-Objekt |
| `after_response(result)` | Antwort fertig (Text, requestId, Fehler, Dauer) | nein |
| `on_model_load(model)` | ein Modell ist bereit | nein |
| `on_tool_call(call)` | jeder Tool-Aufruf des Agenten (Harness-Plugin `agent/plugins/strata-dev-hooks.mjs`, nach `strata-gate`) | `{ deny: 'Grund' }` lehnt ab |
| `routes` | eigene Routen unter `/dev/<name>/<pfad>` | – |

Python-Hooks bekommen `{"event", "payload"}` als JSON auf stdin und geben ein JSON-Objekt aus (`{"payload": ...}` oder `{"deny": "..."}`).
Welche Ereignisse sie nehmen, steht in einer Kommentarzeile: `# strata-hooks: before_request, after_response` (`on_token` geht für
Python nicht). Eigene Tools baut man weiter über Settings → Tools (`tools-settings.mjs`, `custom-tools-mcp.mjs`), nicht über Hooks.

**Fehlergrenze**: Ein Hook, der wirft, kein JSON liefert oder länger als `hooks.timeoutMs` (Standard 2000 ms; `on_token`:
`hooks.tokenBudgetMs`, Standard 50 ms) braucht, wird abgeschaltet und geloggt; die Anfrage läuft ohne ihn weiter. "Reload" oder
sein Schalter im Dev-Panel schaltet ihn wieder ein. Eine synchrone Endlosschleife kann JavaScript nicht unterbrechen.

### `dev.config.json`: alle Schlüssel ab Phase 2

```json
{
  "logging": { "verbose": false },
  "tokens": { "--st-accent": "#e0218a" },
  "ui": {
    "elements": { "sidebar.monitor": { "hidden": true }, "sidebar.models": { "label": "Modelle", "order": 1 } },
    "extensions": { "info-button": { "enabled": true } }
  },
  "hooks": { "timeoutMs": 2000, "tokenBudgetMs": 50, "disabled": [] },
  "overrides": {
    "webSearch": { "enabled": true, "provider": "duckduckgo", "withoutAsking": false },
    "loop": { "enabled": true, "maxRuns": 100 },
    "folder": { "enabled": true, "defaultPath": "C:\\Users\\<du>\\Documents\\StrataDev-Projekte" },
    "permissions": { "defaultMode": "manual", "allowedModes": ["manual", "plan", "bypass"] }
  }
}
```

- `logging.verbose`: "ausführliches Logging", jede Host-Anfrage mit Status und Dauer im Log.
- `tokens`: überschreibt Design-Tokens aus `apps/host/src/styles/kit/tokens.css` (nur `--name` mit einfachem Wert), in beiden Themes.
- `ui.elements`: vorhandene Elemente ausblenden, umbenennen, umsortieren oder durch eine Extension-Komponente ersetzen (`"replace": "<id>"`).
  `dev.config.json` gewinnt gegen eine Extension.
- `overrides.webSearch`: Websuche an/aus, Anbieter (`duckduckgo`, `exa`, `perplexity`, `deepseek`; die mit Schlüssel brauchen ihn in
  Settings → Tools), Frage in Manual überspringen. `overrides.loop`: `/loop` an/aus, höchste Anzahl Antworten.
  `overrides.folder`: Ordner-Modus an/aus, Standardordner. `overrides.permissions`: Standardmodus für Chats ohne eigene Wahl und
  erlaubte Modi (ein nicht erlaubter wird zum Standardmodus).

Nach einer Änderung: "Reload" im Dev-Panel (Websuche und Rechte gelten ab dem nächsten Agent-Start).

### Fehlersuche

- Host-Neustart bei Codeänderung: `dev-build.bat -HostOnly` startet den Host mit `node --watch`, die UI mit Vite-Hot-Reload.
- Port belegt: Liegt der Port in einem von Windows reservierten Bereich (Hyper-V, WSL, Docker), sagt Dev das mit dem Bereich
  (`netsh int ipv4 show excludedportrange protocol=tcp`).
- `/dev-debug` in Claude Code holt den Bericht von `GET /api/dev/debug` (letzter Fehler, Hooks, fehlgeschlagene Requests,
  Logzeilen). Die Vorlage liegt in `dev/claude-commands/dev-debug.md`; einmal nach `.claude/commands/` kopieren.

## Skills und Modelle (Phase 3)

Auch das gibt es nur mit `STRATA_DEV=1`; Live liest weder `dev/skills` noch `dev/models` und kopiert Skills weiter nur als Text.

### Skills (`dev/skills/<name>/`)

Format wie bei Claude-Skills (`skill-struktur.md`): ein Ordner mit `SKILL.md` (Frontmatter `name`, `description`, optional `compatibility`
und beliebige weitere Felder) und beliebigen Unterordnern (`scripts/`, `references/`, `assets/`, `templates/` …).

- **Quellen**: eingebaute Skills, `<data>/skills` (in der App geschrieben), `<workspace>/skills` (hinzugefügt) und neu `dev/skills`
  (IDs `dev-<name>`). Ordner, die mit `_` beginnen, werden übersprungen.
- **Für den Agenten** wird jeder Skill-Ordner ganz nach `<data>/agent/skills-live/<name>/` kopiert. Die Harness nennt beim Laden das
  "Base directory", daher funktionieren `references/aws.md` oder `scripts/x.py`. Die SKILL.md dort behält alle Frontmatter-Felder;
  nur `name`, `description` und bei ausgeschalteten Skills `disable-model-invocation` werden gesetzt. Kopiert wird nur bei einer Änderung.
- **Validator** beim Laden (Skills-Seite, Spalte "Check"): Frontmatter vorhanden, `name` in kebab-case, `description` nicht leer
  (Fehler); über 500 Zeilen, `name` ungleich Ordnername, kein Text (Warnungen).
- **Auswahl im direkten Chat**: wie bisher die Trigger-Phrasen, dazu Skills, deren `description` mindestens zwei wichtige Wörter mit
  der Nachricht teilt.
- **Pro Chat**: Der Skill-Knopf in der Chat-Toolbar wählt die Skills dieses Chats (nichts gewählt: alle). Der direkte Chat nutzt nur
  diese; bei einem Agent-Turn nennt die Nachricht sie, weil die Harness eine Skill-Liste für alle Chats hat.
- **Versionen**: Jeder geänderte Stand (Ordnerinhalt bis 5 MB und die Änderung aus der App) wird gespeichert
  (`<data>/dev/skill-versions`, höchstens 30 je Skill). Skills-Seite → Versionen → "Restore".
- **Bearbeiten** eines Dev-Skills in der App schreibt in seine eigene SKILL.md (Frontmatter bleibt), nicht daneben.
- **Tools**: `tools.json` im Skill-Ordner (`{ "tools": [{ "name", "description", "command", "shell"?, "timeoutSec"? }] }`) wird als
  Befehls-Tool über den vorhandenen Weg (Settings → Tools, `custom-tools-mcp.mjs`) angeboten, ausgeführt im Skill-Ordner.
- **Skripte** laufen nie über einen eigenen Weg: entweder über die Shell des Agenten oder als Befehls-Tool, beides unter
  Manual/Plan/Bypass (Manual fragt, Plan führt nicht aus).
- **Import/Export**: Skills-Seite → Pfad eines Ordners oder einer `.zip` → "Import" (nach `dev/skills`, "replace" überschreibt);
  "Export .zip" pro Skill.
- **Beispiele**: `meeting-notes` (nur SKILL.md), `csv-stats` (SKILL.md + `scripts/` + `tools.json`), `cloud-deploy`
  (SKILL.md + `references/` mit den Varianten aws/gcp/azure).

### Modelle (`dev/models/*.json`)

Ein Overlay legt sich beim Start auf eine Run-Config `strata-<model>.json`, ohne die Datei zu ändern. Strata Control wendet es in
`presets.overlay` an (`strata_control/dev_models.py`), also bei jedem Start (App, Terminal, Neustart); ein getuntes Preset kommt danach
obendrauf. Beispiel: `dev/models/_example-qwen.json` (ohne `_` kopieren, damit es wirkt).

| Schlüssel | Wirkung |
| --- | --- |
| `model` | für welche Run-Configs: ihr `model_name` als Glob, z. B. `qwen3.8-flash-next-*` (Pflicht) |
| `label`, `source`, `quant` | Anzeige; `source` ist `{ "hf": "org/repo", "file"? }`, `{ "gguf": "Pfad" }` oder `{ "server": "URL" }` |
| `context` | `--max-context` |
| `vision` | `vision` der Run-Config |
| `args` | Engine-Optionen: `{ "--kv": "q8_0", "--mtp": null }` (null entfernt, true = Schalter) |
| `env` | Umgebung (null entfernt) |
| `thinking` | `reasoning_effort` für Anfragen ohne eigene Angabe |
| `sampling` | wird in `sampling` gemischt |
| `chat_template` | Pfad zu einer `.jinja` (relativ zur JSON): eine Kopie des Tokenizer-Ordners im Dev-Datenordner nutzt sie |
| `profiles`, `profile` | eigene Profile und das Standardprofil |

**Profile**: "Schnell" (`schnell`), "Coder" (`coder`), "Max Qualität" (`max-qualitaet`) gibt es immer; ein gleichnamiges Profil in der
Datei ersetzt sie. Umschalten im Model-Menü (Bereich "Strata Dev · models"); die Wahl steht in `<STRATA_HOME>/dev-model-profiles.json`
und gilt ab dem nächsten Start des Modells.

**Model-Menü**: "Add model" nimmt eine lokale `.gguf` (wird geprüft: Größe, ob die Strata-Engine sie kennt), ein Hugging-Face-Repo
oder einen laufenden OpenAI-kompatiblen Server bzw. Anbieter (wird als API-Verbindung gespeichert und steht sofort im Modell-Picker
des Chats). Overlays lassen sich bearbeiten, duplizieren und löschen. Eine RAM/VRAM-Warnung aus der erkannten Hardware blockiert nie.
Eine `.gguf`, die die Strata-Engine nicht kennt, ist als "would run via llama.cpp" markiert: Dieser Weg ist ein Vorschlag
(`PHASE3_BERICHT.md`), noch nicht aktiv.

### DeepSeek in Dev

- Die Engine schreibt ihr Expertenprofil (`--expert-profile …/data/expert-profile.bin`) während des Laufs. Dev bekommt eine eigene
  Kopie in `<STRATA_HOME>/expert-profiles/`, damit das geteilte Pack in `Documents\Strata` unverändert bleibt; geht die Kopie nicht,
  liest die Engine nur (`--profile-readonly`).
- Nach "Use a local GGUF" lädt Dev die volle Modell-ID (`deepseek-v4.1-flash-<größe>`) statt `deepseek41-<größe>`.

## Zurücksetzen

```bat
dev-reset.bat
```

Zeigt, was gelöscht wird, und fragt vorher. Gelöscht werden nur `%LOCALAPPDATA%\StrataDev` und der Browser-Speicher des Dev-Fensters
`%LOCALAPPDATA%\ai.strata.desktop.dev`. Live, Modelle und Engine-Komponenten bleiben. Strata Dev vorher schließen.

Wer Strata Dev deinstalliert und "Also remove my Strata data" ankreuzt, löscht nur `%LOCALAPPDATA%\StrataDev`, nie die Ordner von Live.
