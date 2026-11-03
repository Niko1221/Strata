---
description: Strata Dev - eine Uebernahme zuruecknehmen (promote-Branch loeschen, lokalen main auf das Backup setzen)
allowed-tools: Bash(powershell:*), Bash(git log:*), Bash(git branch:*), Read
---

# /dev-rollback

<!-- Vorlage fuer den Slash-Command. Einmal nach .claude/commands/dev-rollback.md kopieren (siehe dev/README.md). -->

Nimm eine Uebernahme zurueck. Antworte auf Deutsch. **Nichts loeschen oder zuruecksetzen ohne die ausdrueckliche Freigabe des
Nutzers in diesem Gespraech.** Nie pushen.

1. Zeige, was da ist (aendert nichts):

   ```
   powershell -NoProfile -ExecutionPolicy Bypass -File dev-rollback.ps1
   ```

   Backups `backup/live-*` (Stand von main vor einer Uebernahme) und Branches `promote/*` (mit Upstream = schon gepusht).
2. Frage, was zurueck soll, und zeige zuerst ohne `-Yes`, was passieren wuerde:
   - `-DeletePromote <name>`: den lokalen Branch `promote/<name>` loeschen.
   - `-Backup backup/live-<Datum>`: den lokalen `main` auf das Backup setzen (nicht, wenn `main` ausgecheckt ist).
   - `-DeleteBackup backup/live-<Datum>`: ein Backup loeschen, das nicht mehr gebraucht wird.
3. Nach dem "ja" denselben Befehl mit `-Yes`. Was schon auf GitHub ist, nimmt der Nutzer selbst zurueck; gib die Befehle weiter, die
   das Skript ausgibt (z. B. Revert-PR oder `git push origin --delete promote/<name>`).

$ARGUMENTS
