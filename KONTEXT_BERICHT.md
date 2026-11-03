# Bericht: contextGuard (Dev)

Stand 03.11.2026, Branch `dev-sandbox`, noch nicht committet.

## Geändert
- `apps/host/server/dev-mode.mjs`: `contextGuardOf` prüft den Abschnitt (0.3 bis 0.9, sonst aus plus Warnung); `devSettings().contextGuard`.
- `apps/host/server/agent/profile.mjs`: `compactionHeadroom(context, out, midTaskAt = null)`. Nur für lokale Modelle (inkl. Shaped-Kopien) gesetzt.
- `apps/host/server/agent/context-fit.mjs`: `guardSummarize`, `guardCompactAfterTurn` (reine Entscheidungen).
- `apps/host/server/agent/agent-bridge.mjs`: `fitCheck` (kurze Nachrichten, Schwelle) und `finishTurn` (Nachlauf nur wenn der bestehende Zweig nicht schon zusammenfasst).
- `apps/host/scripts/dev-validate.mjs`: `contextGuard` als bekannter Schlüssel.
- `dev.config.json`: Schalter an mit 0.55 / 0.55 / 0.60.
- `apps/host/tests/context-guard.test.mjs` (neu), `docs/DEV_BUILD.md`.

## Live unverändert
- Ohne Schalter ist `midTaskAt` null; die Formel `room * 0.125` läuft unverändert (Test: 5734 Token Headroom, Schwelle 40142 bei 64K).
- Das Profil (`patch`) ist mit `enabled: false` identisch zum Profil ohne Dev-Einstellungen (Test).
- `fitCheck` und `finishTurn` lesen `activeDevSettings()`, das außerhalb von Dev null ist.
- Engine und Server nicht angefasst.

## Geprüft
- `npm test` in `apps/host`: 1209 bestanden, 0 fehlgeschlagen, 103 übersprungen.
- Mit Schalter an bei 64K: Headroom 9832, Harness-Schwelle 36044 (55 %).

## Nicht geprüft
- Die Anbindung in `fitCheck` und `finishTurn` läuft nicht im Test: Die Harness-Tests sind übersprungen (`STRATA_HARNESS_DIR` fehlt). Getestet sind die Entscheidungsfunktionen. Es lief kein Modell.
- `python -m pytest tests/terminal` nicht ausgeführt (Python-Seite nicht berührt).
