---
description: Strata Dev - alles pruefen (Validator, Typecheck, npm test, pytest, Smoketest) und kurz berichten
allowed-tools: Bash(powershell:*), Read, Grep, Glob
---

# /dev-check

<!-- Vorlage fuer den Slash-Command. Einmal nach .claude/commands/dev-check.md kopieren (siehe dev/README.md). -->

Pruefe Strata Dev vollstaendig und berichte auf Deutsch. Halte dich an `CLAUDE.md` (nichts in `src`, `include`, `serve`, `sycl`,
`metal`, `third_party` aendern; kein Push; Live und `Documents\Strata` nicht veraendern).

0. Vorher pruefen, dass die Live-App zu ist (Port 47831 frei, kein Prozess aus `Programs\Strata`) und kein anderer dev-check
   laeuft. Laeuft Live, NICHT starten, sondern den Nutzer bitten, Strata ganz zu schliessen (auch im Tray). Das Skript bricht in
   beiden Faellen selbst ab (Exit-Code 2). Nie zwei Laeufe parallel.
1. Fuehre im Repo-Ordner aus (dauert mit pytest etwa 40 Minuten; fuer einen schnellen Lauf `-SkipPytest` anhaengen):

   ```
   powershell -NoProfile -ExecutionPolicy Bypass -File dev-check.ps1 $ARGUMENTS
   ```

   Fehlen dem Python die Pakete (rich, textual, jinja2 ...), meldet der Lauf das als Warnung. Dann den Nutzer fragen, ob
   `.\dev-check.ps1 -SetupVenv` eine eigene venv unter `%LOCALAPPDATA%\StrataDev\test-venv` anlegen darf (laedt Pakete aus PyPI).
   Nie ins System-Python installieren.
2. Lies den Bericht, dessen Pfad die letzte Zeile nennt (`Documents\Strata-Dev-Logs\dev-check\report-*.txt` in den Dokumenten; der Fortschritt steht
   waehrenddessen in `<Zeit>-progress.log` daneben), und bei `FAIL` das
   passende Log daneben.
3. Antworte kurz: Ergebnis je Schritt (Validator, Typecheck, npm test, pytest, Dev-Host, UI, /v1/models, Live-Ports), bei Fehlern die
   Ursache aus dem Log und den naechsten Schritt. Die bekannten pytest-Fehler (hardwarebedingt oder wackelig, Liste in `dev-check.ps1`) zaehlen nicht als neu.
