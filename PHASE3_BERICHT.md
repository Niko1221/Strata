# Phase 3: Skills und Modelle (Bericht)

Stand: 02.11.2026, Branch `dev-sandbox`, aufgebaut auf Phase 2 (`c91c9f29`).
Alles Neue ist nur mit `STRATA_DEV=1` aktiv. Nichts in `src`, `include`, `serve`, `sycl`, `metal`, `third_party` geändert;
`SUPPORTED_SELECTIONS` in `strata_control/models.py` ist unverändert. Bedienung: `docs/DEV_BUILD.md`, Abschnitt "Skills und Modelle (Phase 3)".

## Befund: Progressive Disclosure in der Harness

Gelesen in der gebündelten Harness (`resources/harness`, Pakete `dsh-skill-filesystem`, `dsh-tool-skill`, `dsh-skill`):

1. **Stufe 1, nur Name und Beschreibung**: Die Harness durchsucht `skills-live`, liest dabei jede SKILL.md (um das Frontmatter zu
   parsen) und gibt dem Modell nur einen Katalog `<available_skills>` mit `name: description` (die Beschreibung ggf. gekürzt). Dazu der
   Hinweis, einen Skill erst über das `skill`-Tool zu laden und vorher seinen Anweisungen nicht zu folgen.
2. **Stufe 2, SKILL.md bei Bedarf**: Das Tool `skill` lädt den Text erst, wenn das Modell es aufruft, und liefert ihn als
   `<skill_content>` mit der Zeile **"Base directory for this skill: <Ordner>"** und der Anweisung, relative Pfade dagegen aufzulösen und
   Ressourcen nur bei Bedarf zu laden.
3. **Stufe 3, Ressourcen**: Liest das Modell selbst (`read`, Shell) aus diesem Ordner. Bisher lag dort nur die SKILL.md (Text-Kopie),
   deshalb liefen Verweise wie `references/aws.md` ins Leere. **Jetzt (Dev) liegt der ganze Ordner dort.**
4. Die Harness beobachtet nur Änderungen an `*/SKILL.md`; Ressourcen-Dateien lösen keinen Neuladen aus (unnötig, sie werden bei Bedarf gelesen).
5. Der direkte Chat ohne Agent (`skill-selection.mjs`) hatte kein Stufenmodell: Trigger-Phrase → ganzer Text in den Systemprompt.
   In Dev entscheidet zusätzlich die `description`. Der ganze Text wird weiterhin eingefügt, weil der direkte Chat kein Tool zum Nachladen hat.

## Was gebaut wurde

**Skills** (Host `apps/host/server`, UI `apps/host/src/dev`)
- `dev-skills.mjs`: `dev/skills` als dritte Quelle (`dev-<name>`); Skill-Ordner ganz nach `skills-live` kopiert, nur bei Änderung
  (Signatur-Datei), Frontmatter mit allen unbekannten Feldern erhalten; Validator; Versionen (bis 5 MB je Stand, höchstens 30, Restore);
  `tools.json` → Befehls-Tools; Import (Ordner oder `.zip`/`.skill`) und Export als `.zip`; Bearbeiten eines Dev-Skills schreibt in
  seine SKILL.md.
- `dev-zip.mjs`: kleiner Zip-Leser/-Schreiber mit `node:zlib` (Schutz gegen Pfade außerhalb des Ordners).
- `skill-selection.mjs`: `selectSkillsDev` (Trigger + Beschreibung + Auswahl pro Chat); `selectSkills` (Live) unverändert.
- `agent/tools-settings.mjs`: optionaler Parameter `extraCommands`; die Skill-Tools landen in derselben Tool-Datei wie die eigenen
  Befehls-Tools (`custom-tools-mcp.mjs`). Ohne Parameter (Live) wie bisher.
- `index.mjs`: Dev-Quellen in `skills()`, `materializeSkillFolders` statt der Text-Kopie, Versionen bei jeder Synchronisation,
  `pickSkills` (Live: `selectSkills`), `skillIds` pro Chat (direkter Chat: Filter; Agent-Turn: die Nachricht nennt die erlaubten Skills),
  PATCH schreibt Dev-Skills in ihre Datei, Routen `/api/dev/skills/...`.
- UI: `DevSkills.tsx` (Skills-Seite: Import, Quelle, Prüfung, Dateien, Tools, Versionen mit Restore, Export; Chat-Toolbar: Skill-Wahl
  pro Chat), `dev-state.ts` (Registry und `useDev` ohne den Extension-Loader, damit Screens es importieren können).
- Beispiele: `dev/skills/meeting-notes` (nur SKILL.md), `dev/skills/csv-stats` (SKILL.md + `scripts/csv_stats.py` + `tools.json`),
  `dev/skills/cloud-deploy` (SKILL.md + `references/aws.md|gcp.md|azure.md`, Frontmatter mit eigenem `metadata`-Feld).

**Modelle**
- `strata_control/dev_models.py` + ein Aufruf am Anfang von `presets.overlay` (nur bei `STRATA_DEV=1`): Overlays `dev/models/*.json`
  (context, vision, args, env, thinking, sampling, chat_template, profiles) und das gewählte Profil; danach das getunte Preset wie
  bisher. Weil der Server (`serve/server.py`, `apply_saved_tuning`) `presets.overlay` bei jedem Start aufruft, wirkt das Overlay ohne
  Änderung an `serve`. Die Run-Config-Datei wird nie geändert. Ein Chat-Template wird über eine Kopie des Tokenizer-Ordners im
  Dev-Datenordner gesetzt (der Server liest das Template aus dem Tokenizer-Ordner; einen eigenen Schlüssel dafür gibt es nicht).
- Profile "Schnell", "Coder", "Max Qualität" (eingebaut, pro Datei überschreibbar), Wahl in `<STRATA_HOME>/dev-model-profiles.json`.
  Sie liegen als benannte Overlays unter den getunten Presets aus `presets.py` (Preset = Messung für diesen PC, Profil = Absicht).
- `dev-catalog.mjs`: Routen `/api/dev/models` (Liste, Anlegen/Duplizieren/Bearbeiten, Löschen, Profil, Prüfen einer lokalen `.gguf` über
  die vorhandene Control-Prüfung `localPreview`) und die RAM/VRAM-Warnung.
- UI `DevModels.tsx` im Model-Menü (`NativeModels` bzw. der alte Model-Screen): Overlays mit Profil-Wahl, Bearbeiten (JSON), Duplizieren,
  Löschen; "Add model" für lokale `.gguf`, Hugging-Face-Repo oder laufenden OpenAI-kompatiblen Server (als API-Verbindung über
  `/api/model-providers`, also sofort im Chat-Picker). Warnung, keine Blockade.
- `dev-native.mjs` in `native-control.mjs` (nur Dev): DeepSeeks `--expert-profile` zeigt auf eine Kopie in `<STRATA_HOME>/expert-profiles/`
  (sonst `--profile-readonly`), damit das geteilte Pack in `Documents\Strata` nicht beschrieben wird; `loadAfter` nach "Use a local GGUF"
  nimmt die installierte volle ID (`deepseek-v4.1-flash-<größe>` statt `deepseek41-<größe>`).

## Befund und Vorschlag: beliebige GGUF-Modelle über llama.cpp

- `server/supervisor.mjs` kann `llama-server` starten (`-m <gguf>`, Kontext, KV, GPU-Schichten aus der alten Tuning-Konfiguration), sucht
  die Datei unter `$STRATA_HOME/engine` oder im PATH. Dieser Weg gehört zum **alten Host-Pfad** (`models.mjs`, nur mit
  `STRATA_REAL_ENGINE=1`, alter Varianten-Katalog). Die installierte App nutzt den **nativen Pfad** über Control (`native-control.mjs` →
  `serve/server.py` → Strata-Engine) und kommt dort nicht vorbei. Strata liefert kein `llama-server` mit.
- **Heute in Dev**: Ein beliebiges GGUF lässt sich nur indirekt nutzen: selbst `llama-server` starten und im Model-Menü als
  "OpenAI-kompatibler Server" hinzufügen. Das funktioniert schon (API-Verbindung, Chat und Agent).
- **Vorschlag** (braucht dein OK, ändert `SUPPORTED_SELECTIONS` nicht):
  1. Ein Dev-Runner im Host, der für ein Overlay mit `source.gguf` und `runner: "llama.cpp"` einen `llama-server` (Pfad in
     `dev.config.json`, z. B. aus den offiziellen llama.cpp-Releases) auf einem eigenen Port (ab 8090) startet und beendet, Logs in den
     Dev-Logs. Bausteine aus `supervisor.mjs` (Argumente, Health-Check, Log-Puffer) wiederverwenden.
  2. Der laufende Server wird automatisch als API-Verbindung eingetragen (wie "OpenAI-kompatibler Server"), so dass Chat-Picker, Agent,
     Skills und Hooks ohne neuen Weg funktionieren.
  3. Im Model-Menü die Kennzeichnung "läuft über llama.cpp" (statt "Strata-Engine") mit dem Hinweis, dass Tuning, Presets, MTP/DSpark und
     die Speicherplanung der Strata-Engine dort nicht gelten.
  4. `SUPPORTED_SELECTIONS` bleibt die Liste für die Strata-Engine; nur wenn du willst, dass Control solche Modelle auch einrichtet
     (Pack, Run-Config), wäre sie zu öffnen. Das empfehle ich nicht.

## Geänderte und neue Dateien

Geändert: `apps/host/server/{index.mjs, dev-mode.mjs, native-control.mjs, skill-selection.mjs, agent/tools-settings.mjs}`,
`apps/host/src/{components/ChatToolbar.tsx, components/NativeModels.tsx, dev/dev.tsx, dev/dev.css, dev/overrides.ts, lib/store.ts,
lib/types.ts, screens/ModelRun.tsx, screens/Skills.tsx}`, `strata_control/presets.py`, `docs/DEV_BUILD.md`, `dev/README.md`.

Neu: `apps/host/server/{dev-skills.mjs, dev-zip.mjs, dev-catalog.mjs, dev-native.mjs}`, `apps/host/src/dev/{dev-state.ts, DevSkills.tsx,
DevModels.tsx}`, `apps/host/tests/dev-skills.test.mjs`, `strata_control/dev_models.py`, `tests/terminal/test_dev_models.py`,
`dev/skills/{meeting-notes, csv-stats, cloud-deploy}/…`, `dev/models/_example-qwen.json`, `PHASE3_BERICHT.md`.
Entfernt: `dev/skills/.gitkeep`, `dev/models/.gitkeep`.

## Wie Live-Gleichheit geprüft wurde

1. **Code-Pfade**: Jede Änderung an bestehendem Code hängt an Dev: Host `DEV_SKILLS` ist `null` (Skills wie bisher: `extra`,
   `materializeSkills`, `selectSkills`), `extraCommands` ist `null`, `isDevMode()` in `native-control.mjs`, `STRATA_DEV=1` in
   `presets.overlay`; UI-Komponenten geben ohne Dev `null` zurück, `devSkillsOverride` liefert ohne Dev `{}`. Die Funktion
   `materializeSkills` in `skill-files.mjs` ist unverändert.
2. **Tests**: u. a. Live-Text-Kopie ohne Ressourcen; `selectSkills` ohne Trigger wählt nichts (Live); Tool-Datei ohne `extraCommands`
   ohne Skill-Tools; `presets.overlay` ohne `STRATA_DEV` gibt die Konfiguration unverändert zurück (dasselbe Objekt).
3. **Gerenderter DOM, alt gegen neu**: UI von Phase 2 (`c91c9f29`, temporärer Worktree) und vom neuen Stand ohne `STRATA_DEV` gebaut,
   beide Hosts ohne Dev, 7 Seiten in Chrome verglichen: Chat, Skills, Tasks, Serving, Settings gleich; Models und Monitor nur in
   Umgebungswerten verschieden (Checkout-Pfad in einer Fehlermeldung, schwankender GPU-Takt), wie in Phase 2. `/api/app/mode` gleich.

Live-App und `C:\Users\jonat\Documents\Strata` nicht verändert; alle Läufe mit Scratch-Datenordnern und eigenen Ports, kein Modell geladen.

## Testergebnisse

- `npm test` (apps/host): **1292 Tests, 1189 bestanden, 0 fehlgeschlagen, 103 übersprungen** (Phase 2: 1281/1178/0/103; neu 11 Tests
  in `dev-skills.test.mjs`). Ein erster Lauf hatte 18 Fehler: Tests bündeln `NativeModels` mit esbuild, das das virtuelle Vite-Modul der
  Extensions nicht kennt. Behoben, indem `DevModels`/`DevSkills` nur noch `dev-state.ts` importieren.
- `npm run typecheck`: sauber.
- `python -m pytest tests/terminal` (temporäre venv im Scratch-Ordner wie in Phase 2): **2043 bestanden, 6 fehlgeschlagen,
  62 übersprungen** (Phase 2: 2038/6/62; neu 5 Tests in `test_dev_models.py`). Die 6 Fehler sind dieselben wie vor Phase 2
  (`test_bundle_imports` 1, `test_launch_dsv41` 4, `test_launch_tui::FolderStep` 1; hängen an der Hardware dieses PCs).

## Prüfung im Dev-Modus

Host mit `STRATA_DEV=1` und Dev-UI in Chrome (headless), Scratch-Datenordner, `STRATA_DEV_MODELS` auf einen Scratch-Ordner:
Skills-Seite zeigt den Dev-Bereich mit den drei Beispiel-Skills (alle "ok", `csv-stats` mit Tool und `compatibility`), je eine Version;
`skills-live` enthält die ganzen Ordner (`cloud-deploy/references/*.md`, `csv-stats/scripts`), das Frontmatter von `cloud-deploy` behält
`metadata`; die Tool-Datei enthält `skill-csv-stats-csv-stats`. Die Chat-Toolbar hat den Skill-Knopf. Model-Menü zeigt "Strata Dev · models";
ein Overlay über die API angelegt und auf "Coder" gestellt erscheint mit Profil. Keine Seitenfehler.

**Dev-App gebaut** mit `dev-build.ps1 -NoStart` (Rust-Build 3 min 15 s), Installer
`apps\desktop\src-tauri\target\release\bundle\nsis\Strata Dev_0.2.9_x64-setup.exe`, nicht installiert. Die gebaute `strata-dev.exe` kurz
gestartet: `/api/dev/skills` liefert die drei Beispiel-Skills ohne Fehler (`csv-stats` mit Tool), in
`%LOCALAPPDATA%\StrataDev\home\data\agent\skills-live\cloud-deploy\references` liegen die Varianten; `/api/dev/models` antwortet (Ordner
`dev\models` im Repo, kein aktives Overlay, weil das Beispiel mit `_` beginnt). Im Dev-Datenordner war kein Modell installiert und die
Hardware-Angabe in den ersten Sekunden noch leer, daher keine Speicherwarnung in diesem Lauf. Danach wieder beendet; Live nicht berührt.

## Offene Punkte

- **llama.cpp-Runner**: umgesetzt nach deiner Freigabe (Abschnitt unten). Mit einem echten `llama-server` noch nicht gelaufen, weil auf
  diesem PC keiner liegt.
- **Skill-Wahl pro Chat im Agenten**: Die Harness hat eine Skill-Liste für alle Chats; Dev nennt die erlaubten Skills in der Nachricht
  (das Modell hält sich daran, erzwungen ist es nicht). Erzwingen ginge mit einem eigenen Skill-Ordner pro Chat (Harness-Konfiguration pro
  Sitzung), das wäre ein größerer Umbau.
- **Hugging Face in "Add model"**: legt ein Overlay mit Quelle an, lädt aber nichts herunter. Modelle der unterstützten Familien lädt
  weiter das vorhandene Model-Menü; andere bräuchten den llama.cpp-Runner.
- **Profile** wirken ab dem nächsten Start des Modells (Control wendet sie beim Start an), nicht auf ein laufendes.
- **DeepSeek-Profil**: Ob die Engine die Profildatei wirklich schreibt, habe ich nicht gemessen (kein Lasttest); Dev geht auf Nummer sicher
  (Kopie oder `--profile-readonly`). Die Korrektur der vollen Modell-ID bei `loadAfter` ist ein Live-Fehler, der nur in Dev behoben ist;
  Kandidat für die Übernahme in Phase 4.
- Der Skills-Import nimmt einen getippten Pfad (Ordner oder `.zip`); einen Dateiauswahl-Dialog für `.zip` gibt es noch nicht.

## Nachtrag: llama.cpp-Runner (nach Freigabe des Vorschlags)

Basis `34161535`, eigener Commit. Nur mit `STRATA_DEV=1`; `SUPPORTED_SELECTIONS`, `src`, `serve` und die GPU-Backends unverändert.
Nichts heruntergeladen oder installiert.

**Konfiguration** (`dev.config.json`):

```json
"llamacpp": { "server": "C:\llama.cpp\llama-server.exe", "port": 8095, "args": ["-ngl", "99"], "context": 8192 }
```

`port` Standard 8095; die Ports von Strata (8080, 8081, 8090, 8791–8794, 47831, 47832) werden abgelehnt. `args` sind zusätzliche
llama-server-Optionen; `-m`, `--host 127.0.0.1`, `--port`, `-c` und `--alias` setzt der Runner. Optional `healthTimeoutMs`
(Standard 180 s, so lange darf das Laden dauern).

**Woher `llama-server`**: offizielle Releases von llama.cpp, https://github.com/ggml-org/llama.cpp/releases. Für die RTX 5070 das
Windows-Paket mit `cuda` im Namen (dazu das passende `cudart`-Paket aus demselben Release), sonst `vulkan` oder `cpu`. In einen eigenen
Ordner entpacken (nicht in einen Strata-Ordner) und den Pfad zu `llama-server.exe` eintragen. Danach im Dev-Panel "Reload".

**Bedienung**: Model-Menü → "Add model" → "A local .gguf file" → Pfad → "Check" → "Add". Eine Datei, die die Strata-Engine nicht kennt,
wird als Overlay mit `"runner": "llama.cpp"` gespeichert und steht in der Liste mit dem Hinweis "läuft über llama.cpp" und einem Knopf
"Start". Der Runner-Kasten darüber zeigt den Zustand (bereit / startet / läuft auf Port … / Fehler mit den letzten Zeilen des Servers) und
"Stop". Läuft der Server (`/health` = 200), ist er als API-Verbindung "llama.cpp (Strata Dev)" mit dem Modell "<Name> · läuft über llama.cpp"
eingetragen und im Modell-Picker des Chats wählbar; Chat, Agent, Skills und Hooks nutzen ihn wie jede API-Verbindung.

**Fehler und Beenden**: fehlender/falscher Pfad, Ordner statt Programm, Strata-Port, belegter Port, fehlende `.gguf`, Absturz beim Laden
oder später, keine Antwort in der Zeit: jeweils ein Zustand `error` mit Klartext im Model-Menü und im Dev-Panel (Logs, Quelle `llama`);
der Host läuft weiter. Ein Modell zur Zeit; ein neuer Start beendet den alten Server. Beim Beenden des Hosts (`shutdown`) wird der Server
gestoppt, bei jedem anderen Ende des Host-Prozesses sofort beendet (`process.on('exit')`).

**Dateien**: neu `apps/host/server/dev-llama.mjs`, `apps/host/tests/dev-llama.test.mjs`, `apps/host/tests/fixtures/mock-llama-server.mjs`;
geändert `apps/host/server/index.mjs` (Runner, Route `/api/dev/llama`, Stopp beim Beenden), `apps/host/server/dev-api.mjs` und
`apps/host/src/dev/DevPanel.tsx` (Log-Quelle `llama`), `apps/host/src/dev/DevModels.tsx` (Runner-Kasten, Start/Stop, Kennzeichnung),
`apps/host/src/dev/dev.css`, `docs/DEV_BUILD.md` (Abschnitt "llama.cpp-Runner"), dieser Bericht.

**Geprüft** (ohne echten llama-server): 6 Tests gegen einen Mock-Server, der wie `llama-server` aufgerufen wird und echt auf einem Port
lauscht (`/health` erst 503, dann 200, `/v1/models`): Start bis "running" mit genau einer API-Verbindung (Basis-URL, Modell-ID = Alias,
Name mit "läuft über llama.cpp"), zweiter Start ersetzt das Modell und behält die Verbindung, Stop beendet den Prozess; Absturz beim Laden
→ Fehler mit der letzten Serverzeile, keine Verbindung; keine Antwort in der Zeit → Prozess beendet, Fehler; belegter Port, fehlende
`.gguf`, kein `llama-server` → Klartext-Fehler; Routen. Dazu ein Host-Smoke-Test mit falschem Pfad: `/api/dev/llama` meldet
"llama-server is not at …", Start liefert den Fehler, die Zeile steht in den Dev-Logs, der Host läuft weiter.
`npm test` (apps/host): **1298 Tests, 1195 bestanden, 0 fehlgeschlagen, 103 übersprungen** (6 neu); `npm run typecheck` sauber.
Die Python-Suite ist von diesem Nachtrag nicht betroffen (kein Python-Code geändert) und wurde nicht erneut ausgeführt.
