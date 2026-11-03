---
description: Strata Dev - Dev-Logs und den letzten Fehler einsammeln und mit der Fehlersuche beginnen
allowed-tools: Bash(curl:*), Read, Grep, Glob
---

# /dev-debug

<!-- Vorlage fuer den Slash-Command. Einmal nach .claude/commands/dev-debug.md kopieren (siehe dev/README.md). -->

Sammle den aktuellen Zustand von Strata Dev ein und beginne direkt mit der Fehlersuche. Antworte auf Deutsch.
Nur lesen: nichts an Live (`%LOCALAPPDATA%\Strata`, `Documents\Strata`, Ports 47831/8791/8080) ändern oder starten.

1. **Bericht vom laufenden Dev-Host holen.** Der Host antwortet nur mit `STRATA_DEV=1` auf `/api/dev/...`.
   Probiere die Dev-Ports der Reihe nach (installierte Dev-App: 47832; `dev-build.ps1 -HostOnly` / `npm run dev`: 8792):

   ```bash
   for p in 47832 8792; do curl -sf --max-time 5 "http://127.0.0.1:$p/api/dev/debug" && break; done
   ```

   Der Bericht enthält: den letzten Fehler mit Stack, Hooks (an/aus, Fehler), fehlgeschlagene Requests, Fehler und Warnungen
   aus allen Quellen (host, engine, harness, hooks, ui) und die letzten Logzeilen.
   Mehr Zeilen einer Quelle: `curl -s "http://127.0.0.1:<port>/api/dev/logs?sources=engine&limit=2000"`.
   Requests mit Dauer, Tokens und Hook-Zeiten: `curl -s "http://127.0.0.1:<port>/api/dev/requests"`.

2. **Läuft kein Dev-Host**, lies stattdessen die Dateien von Strata Dev (nur lesen):
   - `%LOCALAPPDATA%\StrataDev\desktop\` (Desktop-Log der Dev-App, neueste Datei)
   - `%LOCALAPPDATA%\StrataDev\home\agent\harness.log` (Agent/Harness, wird beim Beenden geschrieben)
   - `dev.config.json` im Repo (Warnungen stehen im Bericht unter "config warning")

3. **Auswerten.** Nenne kurz: den letzten Fehler, wo er herkommt (Quelle, Datei, Hook oder Extension), und die nächsten Schritte.
   Fundstellen: Host `apps/host/server/` (Dev-Teile: `dev-mode.mjs`, `dev-hooks.mjs`, `dev-api.mjs`, `dev-log.mjs`),
   UI `apps/host/src/` (Dev-Teile: `src/dev/`), Extensions `dev/extensions/ui/`, Hooks `dev/extensions/backend/`.
   Ein Hook, der unter "Hooks" als `OFF` mit `error=` steht, wurde von der Fehlergrenze abgeschaltet; nach dem Fix im Dev-Panel
   "Reload" drücken oder `curl -s -X POST http://127.0.0.1:<port>/api/dev/reload`.

4. Halte dich an `CLAUDE.md` (nie `src`, `include`, `serve`, `sycl`, `metal`, `third_party` ändern; kein Push).

$ARGUMENTS
