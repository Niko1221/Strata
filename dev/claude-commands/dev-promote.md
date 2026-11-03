---
description: Strata Dev - Uebernahme in Live vorbereiten (Kern vs. Dev-only zeigen, nach Freigabe auf promote/<name> uebernehmen)
allowed-tools: Bash(powershell:*), Bash(git log:*), Bash(git diff:*), Bash(git show:*), Read
---

# /dev-promote

<!-- Vorlage fuer den Slash-Command. Einmal nach .claude/commands/dev-promote.md kopieren (siehe dev/README.md). -->

Bereite eine Uebernahme von `dev-sandbox` nach Live vor. Antworte auf Deutsch. **Nichts uebernehmen ohne die ausdrueckliche Auswahl
und Freigabe des Nutzers in diesem Gespraech.** Nie `git push`, `git tag`, `gh release` oder `gh pr merge`.

1. Ueberblick (aendert nichts):

   ```
   powershell -NoProfile -ExecutionPolicy Bypass -File dev-promote.ps1
   ```

   Zeige dem Nutzer die Commits und Dateien, getrennt nach **Kern-Aenderungen** und **Dev-only**. Erklaere, dass Kern-Dateien mit
   "Dev-Haken" die Dev-only-Module brauchen und ohne sie auf main nicht bauen (das Skript warnt bei `-DryRun`). Ein reiner Kern-Fix
   ohne Dev-Haken ist z. B. `e26737c8` (MTP-Check). Die Korrektur "volle Modell-ID nach Use a local GGUF" (DeepSeek) steckt heute in
   `dev-native.mjs` und ist nur in Dev aktiv: fuer Live muesste sie als eigener kleiner Kern-Commit ohne Dev-Schalter geschrieben werden.
2. Lass den Nutzer waehlen (Commits oder Dateien, ob Dev-only mit soll, einen Namen). Zeige dann den Probelauf:

   ```
   powershell -NoProfile -ExecutionPolicy Bypass -File dev-promote.ps1 -Name <name> -Commits <hash,...> -DryRun
   ```

3. Erst nach seinem ausdruecklichen "ja": derselbe Befehl mit `-Yes` statt `-DryRun`. Das legt `backup/live-<Datum>` und
   `promote/<name>` an (lokal, in einem temporaeren Worktree; der Checkout bleibt).
4. Gib die Befehle fuer Push und Pull Request weiter, die das Skript ausgibt (Titel mit Bereich nach CONTRIBUTING.md), und die Datei
   mit dem PR-Text zum Ausfuellen. Ausfuehren tut sie der Nutzer selbst.

$ARGUMENTS
