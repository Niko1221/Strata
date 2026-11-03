# Phase 4: Prüfen, Bauen und Übernahme in Live (Bericht)

Stand: 03.11.2026, Branch `dev-sandbox`, Basis `0d025134`. Ausführlich: `docs/DEV_BUILD.md`, Abschnitt "Von Dev nach Live".

**Wichtig vorweg:** Nichts in dieser Phase spielt etwas auf die installierte Live-App. "Übernahme in Live" bereitet nur einen lokalen
Git-Branch `promote/<name>` vor. Eine echte Übernahme habe ich **nicht** ausgeführt; die Werkzeuge sind nur in einem Wegwerf-Klon und
mit Probeläufen geprüft. `main` gibt es lokal nicht (nur `origin/main`) und es wurde nicht verändert. Kein Push, kein Tag, kein Release.

## Was gebaut wurde

| Datei | Zweck |
| --- | --- |
| `dev-build.ps1` (erweitert) | baut Dev-UI, Dev-App und NSIS-Installer wie bisher; am Ende eine Übersicht: Installer, App, Ports, Datenordner, wie starten (Doppelklick, `-Install`, Browser-URL), was als Nächstes (`dev-check`, `dev-promote`) |
| `dev-check.ps1` + `.bat` | Validator, Typecheck, `npm test`, `pytest tests/terminal` (bekannte Fehler markiert), Smoketest; Bericht, Logs und Fortschritt in `Documents\Strata-Dev-Logs\dev-check\`; bricht ab, wenn Live läuft oder schon ein dev-check läuft |
| `apps/host/scripts/dev-validate.mjs` | Validator für `dev.config.json`, UI-Extensions, Backend-Hooks, Skills (inkl. `tools.json`), Modell-Overlays, `llamacpp`-Pfad; nutzt dieselben Prüfungen wie der Host zur Laufzeit |
| `dev-promote.ps1` + `.bat` | Überblick Kern vs. Dev-only, Probelauf, Übernahme auf `promote/<name>` mit Backup-Branch, Befehle für Push und PR |
| `dev-rollback.ps1` + `.bat` | Backups und promote-Branches ansehen, promote-Branch löschen, lokalen `main` auf ein Backup setzen, Backup löschen |
| `dev/claude-commands/dev-check.md`, `dev-promote.md`, `dev-rollback.md` | Vorlagen der Slash-Commands (neben `dev-debug.md`) |
| `apps/host/tests/dev-validate.test.mjs` | Test des Validators |
| `dev/tools/rescue-chats.mjs`, `rescue-one-chat.mjs`, `restore-live-chat.mjs` | Chat-Rettung aus den Agent-Verläufen (siehe unten), nur Dev-Werkzeuge |
| `apps/host/server/dev-rescue.mjs`, Tab "Rescue" im Dev-Panel | zeigt gerettete Chats und lädt sie in die Chatliste von Strata Dev (nur mit `STRATA_DEV=1`) |
| `docs/DEV_BUILD.md`, `dev/README.md` | Ablauf bauen → prüfen → ansehen → übernehmen → zurückrollen |

### dev-check im Detail

- **Vorher**: Live-App ganz schließen, nur ein Lauf zur Zeit. Das Skript prüft beides (Port 47831 bzw. ein Prozess aus `Programs\Strata`;
  benannter Mutex) und bricht sonst mit Exit-Code 2 und einem Hinweis ab. `-AllowLive` erlaubt den Lauf neben Live, dann ohne pytest.
- **Logs**: `Documents\Strata-Dev-Logs\dev-check\`: `report-<Zeit>.txt`, je Schritt ein Log und `<Zeit>-progress.log`, das während des
  Laufs Zeile für Zeile wächst. Die Konsole steht auf UTF-8; die Logs haben keine kaputten Umlaute mehr.
- **pytest**: Python-Wahl in dieser Reihenfolge: `-Python`, die eigene venv `%LOCALAPPDATA%\StrataDev\test-venv`, `STRATA_DEV_PYTHON`,
  `python`; genommen wird die erste mit rich, textual, jinja2, psutil, pyyaml und pytest. Fehlt eine, ist der Schritt eine Warnung mit
  dem Hinweis `.\dev-check.bat -SetupVenv` (legt die venv an, aktualisiert pip, lädt die Pakete aus PyPI und prüft danach, dass alle
  sich laden lassen; nur auf diesen ausdrücklichen Schalter; das System-Python bleibt). Die bekannten Fehler (sechs hardwarebedingte,
  siehe PHASE2_BERICHT.md, und ein unter Last wackelnder Zeitmessungs-Test) stehen als Liste im Skript und zählen als "bekannt"; jeder
  andere Fehler ist "NEU" und macht den Schritt rot.
- **Smoketest**: Läuft auf 47832 keine Dev-App, startet das Skript die neueste gebaute (`%LOCALAPPDATA%\Programs\Strata Dev` oder
  `target\release`) und beendet sie danach wieder. Geprüft: `/api/app/mode` meldet Dev, `GET /` liefert die UI, `/v1/models` des Hosts,
  `/api/dev/state`; `/v1/models` auf 8081 nur, wenn dort schon eine Engine lauscht (es wird kein Modell geladen, kein Lasttest); auf
  47831/8791/8080 darf kein Prozess aus einem Dev-Ordner lauschen.

### dev-promote im Detail

- **Dev-only** (nie übernommen, außer `-IncludeDevOnly`): `dev/`, `dev.config.json`, `dev-*.ps1/.bat`, `docs/DEV_BUILD.md`, Berichte,
  `PROMPTS.md`, `ANALYSE.md`, `skill-struktur.md`, `.claude/`, `tauri.dev-build.conf.json`, `devbuild.rs`, `apps/host/server/dev-*.mjs`,
  `strata-dev-hooks.mjs`, `apps/host/src/dev/`, `DevBadge.tsx`, Dev-Tests und -Fixtures, `dev-validate.mjs`, `strata_control/dev_models.py`,
  `tests/terminal/test_dev_models.py`. Alles andere ist **Kern**; eine Kern-Datei, deren neue Zeilen Dev-Code erwähnen, ist als
  **Dev-Haken** markiert (nur mit `STRATA_DEV=1` aktiv, braucht die Dev-only-Module).
- **Probelauf** (`-DryRun`): Dateiliste, weggelassene Dev-only-Dateien und eine Warnung für jeden Import einer übernommenen Datei, dessen
  Ziel auf `main` fehlt und nicht mitkommt (sonst bricht der Build).
- **Übernahme** (`-Yes`): Backup-Branch `backup/live-<Datum>` vom aktuellen `main` (lokal, sonst `origin/main`), dann `promote/<name>` in
  einem temporären Worktree unter `%LOCALAPPDATA%\StrataDev\pw` (der Checkout von `dev-sandbox` bleibt, wie er ist; lange Pfade erlaubt),
  `cherry-pick -x` der Commits ohne Dev-only-Dateien bzw. die gewählten Dateien. Konflikte nur in Dev-only-Dateien löst das Skript;
  einer in einer Kern-Datei bricht ab, nennt die früheren Commits, auf denen die Datei aufbaut, und entfernt Branch und Backup dieses
  Laufs wieder. Am Ende die Befehle `git push -u origin promote/<name>` und `gh pr create --base main … --title "[<Bereich>] …"`
  (Bereich aus den Dateien: GUI, TUI, Engine, Docs, Benchmark, Other) und eine PR-Text-Vorlage nach CONTRIBUTING.md.

## Testergebnisse

Vollständiger `dev-check` am 03.11.2026 (Lauf `20261103-015952`, Live-App geschlossen, ein Lauf), Bericht und Logs unter
`Documents\Strata-Dev-Logs\dev-check\`:

| Schritt | Ergebnis |
| --- | --- |
| Validator | OK: 0 Fehler, 0 Warnungen, 6 OK (Config, UI-Extension, Backend-Hook, 3 Skills) |
| Typecheck | OK: `tsc --noEmit` ohne Fehler |
| npm test | OK: 1192 bestanden, 0 fehlgeschlagen, 107 übersprungen |
| pytest | OK: 2043 bestanden, 62 übersprungen, 6 fehlgeschlagen; alle 6 bekannt und hardwarebedingt (`test_bundle_imports` ohne llama.cpp-`gguf-py`, 4 × `test_launch_dsv41` mit Plan für 4 Karten/251 GB RAM, `test_launch_tui::FolderStep` mit Plattenplatz), 25 min |
| Smoketest | OK: Dev-Host auf 47832 als Strata Dev, UI lädt, `/v1/models` (Host), `/api/dev/state`; 8081 übersprungen (kein Modell geladen); 47831 und 8791 frei, 8080 belegt von `NVIDIA Broadcast.exe`, kein Dev-Prozess auf Live-Ports |
| Gesamt | 0 Fehler, 0 Warnungen |

Der Lauf davor (`20261103-012850`) hatte 11 weitere pytest-Fehler. Ursache war die Umgebung, nicht der Code (kein Python-File dieser
Tests unterscheidet sich von `main`): gestartet aus einer PowerShell, in der `bash` der WSL-Platzhalter ohne Distribution war,
`NO_COLOR=1` gesetzt war (aus meiner Werkzeug-Umgebung, nicht deinem System) und `TERM` fehlte. Einzeln mit korrigierter Umgebung liefen
alle 11 durch. Seitdem gibt `dev-check` pytest immer dieselbe Umgebung (Git-Bash vor dem WSL-Platzhalter, `TERM=xterm-256color`, ohne
`NO_COLOR`) und stellt sie danach wieder her.

### Testlauf von dev-promote / dev-rollback (Wegwerf-Klon)

Klon des Repos im Scratch-Ordner, darin `main` aus `origin/main` (c62d5e3a) angelegt:

1. Überblick im echten Repo (nur lesen): 7 Commits voraus; 41 Kern-Dateien (davon 30 mit Dev-Haken) und 52 Dev-only-Dateien (Stand vor dem Commit dieser Phase).
2. `-Name mtp-fix -Commits e26737c8 -DryRun`: 2 Dateien (`tools/mtp_fetch.py`, Test), keine Warnung.
3. `-Name skills -Commits 34161535 -DryRun`: übernimmt nur die Kern-Dateien und warnt mit 16 Verweisen auf fehlende Dev-Module
   (z. B. `index.mjs -> dev-skills.mjs`, `presets.py -> dev_models.py`).
4. `-Name mtp-fix -Commits e26737c8 -Yes`: `backup/live-20261102` = c62d5e3a, `promote/mtp-fix` mit einem Commit (`cherry picked from
   commit e26737c8… on dev-sandbox`), `main` unverändert, kein Worktree übrig, Push-/PR-Befehle ausgegeben (Titel `[Other] MTP check: …`).
5. `-Commits 8199b7b7 -Yes`: Konflikt in `app.css` (baut auf `abe93a81` auf) → Abbruch mit dem Hinweis "Wähle sie mit: abe93a81",
   Branch und Backup wieder entfernt. Mit `-Commits abe93a81,8199b7b7 -Yes`: zwei Commits, 16 Dev-only-Dateien weggelassen
   (der Konflikt in `devbuild.rs` wurde als Dev-only aufgelöst).
6. `dev-rollback`: Liste zeigt Backups und promote-Branches; `-DeletePromote` ohne `-Yes` ändert nichts; `-Backup … -Yes` setzt einen
   vorher verschobenen lokalen `main` zurück auf das Backup; `-DeleteBackup … -Yes` löscht es.

Der Klon wurde danach gelöscht; im echten Repo gibt es keine neuen Branches.

## Bedienung (kurz)

In PowerShell gehört `.\` davor (sonst "nicht erkannt"); in `cmd` oder per Doppelklick geht es ohne.

```bat
.\dev-build.bat -NoStart          :: bauen (oder ohne -NoStart: bauen und starten)
.\dev-check.bat                   :: alles prüfen; vorher Live schließen, nur ein Lauf (-SkipPytest: schnell; einmal -SetupVenv)
.\dev-promote.bat                 :: ansehen, was Kern und was Dev-only ist
.\dev-promote.bat -Name <n> -Commits <hash> -DryRun   :: Probelauf
.\dev-promote.bat -Name <n> -Commits <hash> -Yes      :: übernehmen (deine Freigabe)
.\dev-rollback.bat                :: zurück
```

### Slash-Commands einrichten

Vorlagen für `/dev-check`, `/dev-promote`, `/dev-rollback` und `/dev-debug` liegen in `dev/claude-commands/`. Einmal im Repo-Ordner
in `cmd` kopieren (Claude Code darf `.claude\commands` nicht selbst anlegen):

```bat
mkdir .claude\commands
copy dev\claude-commands\*.md .claude\commands\
```

Danach kennt Claude Code in diesem Ordner `/dev-check` (prüft vorher, dass Live zu ist), `/dev-promote` (Überblick und Probelauf,
übernimmt nur mit deiner Freigabe), `/dev-rollback` und `/dev-debug`.

## Chat-Rettung (Zwischenfall vom 02.11.2026)

Am Abend des 02.11. war die Chatliste der Live-App bis auf 8 Chats leer. Ursache (aus dem Code gelesen, nicht gemessen): Die Chatliste
liegt im localStorage des Live-Fensters (`strata.sessions.v1`) und ist auf 3,5 Mio. Zeichen begrenzt (`MAX_TOTAL_CHARS` in
`apps/host/src/lib/session-persist.ts`). Wird sie größer, lässt die App beim Speichern die ältesten Chats weg. Lange Chats mit
angehängten Dateien füllen die Grenze schnell. Dass meine Prozesse die Live-Daten geschrieben haben, ist auszuschließen: Windows leitet
alle AppData-Schreibzugriffe aus der Claude-App (Store-App) in deren Paketordner um. Nicht auszuschließen ist, dass ein pytest-Lauf die
laufende Live-App beendet hat; deshalb bricht `dev-check` jetzt ab, wenn Live läuft.

Die Agent-Verläufe der Chats (`~\.strata\data\agent\home\sessions`) waren noch vollständig. Daraus bauen die Werkzeuge die Chats nach:

| Werkzeug | Was |
| --- | --- |
| `dev/tools/rescue-chats.mjs` | alle Hauptchats aus den Agent-Verläufen (Helfer-Sitzungen übersprungen), abgeglichen mit den noch vorhandenen Chats; schreibt `rescued-chats-full-*.json` und eine gekürzte `rescued-chats-import-*.json` für den Rescue-Tab |
| `dev/tools/rescue-one-chat.mjs` | ein Chat so vollständig wie möglich: lesbares `.md` und Import-Datei mit Werkzeug-Karten und Reasoning |
| `dev/tools/restore-live-chat.mjs` | bereitet in einer **Kopie** der Live-Local-Storage einen Chat zum Zurückspielen vor (Chrome im Hintergrund, kein Server); den Ordner tauscht der Nutzer selbst |
| Rescue-Tab (`dev-rescue.mjs`) | listet Import-Dateien aus `STRATA_DEV_RESCUE`, `%LOCALAPPDATA%\StrataDev\rescue` oder `Documents\Strata-Rescue-Backup` und lädt sie in die Chatliste von Strata Dev |

Ergebnis: 82 Hauptverläufe → 8 Chats waren noch in Live, 72 nachgebaut, 2 leer. Der wichtige Chat ist zurück in Live (vom Nutzer
selbst eingespielt und geprüft). Im Repo liegen nur die Werkzeuge, keine Chat-Inhalte und keine persönlichen Pfade; die Sicherungen
liegen außerhalb des Repos unter `Documents\Strata-Rescue-Backup`.

### Vorschlag Kern-Fix (nicht gebaut, braucht deine Freigabe)

Die Chatliste zusätzlich als Dateien auf der Platte speichern, über den Host (etwa `<STRATA_HOME>\chats\<id>.json` plus Index), und
localStorage nur noch als schnellen Zwischenspeicher nutzen. Beim Start liest die UI die Liste vom Host; fehlt dort ein Chat, der im
localStorage steht, wird er nachgetragen. Die Grenze von 3,5 Mio. Zeichen gilt dann nicht mehr für die Dauerhaftigkeit.

- **Aufwand**: etwa 1 bis 2 Tage. Zwei Host-Routen (Liste lesen, Chat schreiben/löschen), `session-persist.ts` schreibt je geändertem
  Chat statt der ganzen Liste, einmalige Übernahme der localStorage-Liste beim ersten Start, Tests in `apps/host/tests`.
- **Risiko**: mittel. Kern-Änderung (ändert Live) an jedem Chat. Fehlerquellen: zwei Speicher laufen auseinander, mehrere Fenster,
  Schreiben beim Beenden, Platz bei sehr großen Anhängen. Abmilderung: atomar schreiben (Temp-Datei, dann umbenennen), erst lesen und
  vergleichen, localStorage bleibt als Rückfall.
- **Kleinere Alternative** (wenige Stunden, geringes Risiko): Chats, die beim Kürzen wegfallen, nicht verwerfen, sondern über den Host
  in eine Datei sichern und in der UI einen Hinweis zeigen. Verhindert stillen Verlust, ändert das Laden aber nicht.

## Kandidaten für Kern-Änderungen

- **`e26737c8` MTP-Check** (`tools/mtp_fetch.py`): reiner Kern-Fix ohne Dev-Haken, im Klon sauber übernommen.
- **DeepSeek: volle Modell-ID nach "Use a local GGUF"** (Phase 3): In Live wird nach dem Import `deepseek41-<größe>` geladen, die
  Run-Config heißt aber `deepseek-v4.1-flash-<größe>`. In Dev behoben (`dev-native.mjs`, nur mit `STRATA_DEV=1`). Für Live ist es
  **kein** Cherry-Pick-Kandidat, weil es an den Dev-Modulen hängt: Es bräuchte einen eigenen kleinen Kern-Commit in
  `native-control.mjs` ohne Dev-Schalter (`devLoadAfterModel` als normale Funktion). Das kann ich auf Wunsch als eigenen Commit
  vorbereiten, den `dev-promote` dann ohne Warnung übernimmt.
- Alles andere aus den Phasen 1 bis 3 ist Dev-Infrastruktur (Kern-Dateien mit Dev-Haken plus Dev-only-Module). Übernimmt man es, kommt
  Strata Dev mit nach `main`, wirkt dort aber nur mit `STRATA_DEV=1`.

## Offene Punkte

- **Slash-Commands**: einmal kopieren (siehe "Slash-Commands einrichten").
- **pytest-venv**: Für deine eigenen Läufe einmal `.\dev-check.bat -SetupVenv` (lädt Pakete aus PyPI) oder `-Python <exe>`. Die venv,
  die ich angelegt habe, siehst du nicht (Store-Umleitung, nächster Punkt).
- **Store-Umleitung**: Was Claude Code aus der Claude-App heraus nach `%LOCALAPPDATA%` oder `%APPDATA%` schreibt, landet in
  `%LOCALAPPDATA%\Packages\Claude_pzs8sxrjxfjjc\LocalCache\…` und ist für dich und für Strata unsichtbar. Dateien, die du lesen
  sollst, liegen deshalb unter Dokumente (`Strata-Dev-Logs`, `Strata-Rescue-Backup`). Das Repo ist nicht betroffen.
- **Hardware-Fehler in pytest**: sechs Tests scheitern auf diesem PC schon ohne Strata Dev (Platz- und GPU-Annahmen, siehe
  PHASE2_BERICHT.md), dazu ein Zeitmessungs-Test, der unter Last wackelt. Sie stehen als "bekannt" in `dev-check.ps1`.
- **Kern-Fix Chatliste**: oben beschrieben, nicht gebaut.
- **Smoketest 8081**: ohne geladenes Modell übersprungen (Absicht: kein Lasttest).
- **Port 8080 belegt von NVIDIA Broadcast**: Das ist der Engine-Port von Live. Nicht geprüft, wie Live damit umgeht (ausweichen oder
  Fehler beim Laden). Falls Live beim Laden eines Modells hängt oder einen anderen Port meldet, ist das ein Kandidat.
- Die generierte Datei `apps/desktop/src-tauri/gen/schemas/windows-schema.json` liegt weiter unversioniert im Ordner.
