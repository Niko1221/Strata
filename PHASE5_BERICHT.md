# Phase 5: Chat-Liste, nichts mehr still verlieren (Bericht)

Stand: 03.11.2026, Branch `dev-sandbox`. Freigegeben: die **kleine Variante** des Kern-Fixes aus PHASE4_BERICHT.md. Ausführlich und mit
Anleitung zum Ausprobieren: `docs/DEV_BUILD.md`, Abschnitt "Chat-Liste: nichts mehr still verlieren".

## Fundstelle

- `apps/host/src/lib/session-persist.ts`, `createSessionWriter` → `flush()`: Ist die Chatliste größer als `MAX_TOTAL_CHARS`
  (3,5 Mio. Zeichen), lässt sie die ältesten nicht angehefteten Chats aus der gespeicherten Kopie weg; der neueste bleibt immer.
- `apps/host/src/lib/store.ts`: `sessionWriter` schreibt die Liste nach `localStorage["strata.sessions.v1"]` (`persistSessions`,
  `flushSessions`).
- Nach dem nächsten Neuladen fehlen die weggelassenen Chats. Das war die Ursache des Verlusts vom 02.11.

## Ablauf

1. Muss ein Chat aus der gespeicherten Liste, fragt der Writer den Wächter (`src/lib/chat-overflow.ts`): Ist dieser Chat in genau
   dieser Version (`id` + `updatedAt`) schon gesichert?
2. Nein: Der Chat **bleibt** in der Liste (sie ist kurz größer als die Grenze), und der Wächter schickt ihn vollständig an den Host,
   einen Chat pro Anfrage, nacheinander.
3. Der Host (`server/chat-overflow.mjs`, `POST /api/chat-overflow`) schreibt `[chat]` nach
   `<STRATA_HOME>\chat-overflow\overflow-<letzte Änderung>-<Chat-ID>.json`: Temp-Datei, dann Umbenennen; keine Titel im Dateinamen;
   dieselbe Version nur einmal; bis 64 MB pro Chat.
4. Erfolg: Hinweis "n ältere Chats wurden … ausgelagert und vollständig gesichert" mit Ordner (bleibt, bis man ihn schließt), und die
   Liste wird neu geschrieben, jetzt ohne die gesicherten Chats.
5. Fehler: Warnung, der Chat bleibt in der Liste, neuer Versuch frühestens nach 5 Minuten. Es wird nie ein Chat ohne Sicherung
   weggelassen.
6. Zurückholen: Dev-Panel → Rescue listet auch die `overflow-*.json`; "Laden" holt einen Chat als neuesten zurück.

Unter der Grenze ändert sich nichts.

## Schalter und Live

- `dev.config.json`: `"chatOverflowGuard": { "enabled": true }` (Standard an; `false` schaltet aus). `maxTotalChars` (10.000 bis
  3.500.000) setzt die Grenze nur zum Ausprobieren herunter.
- Ohne `STRATA_DEV=1` gibt es den Wächter nicht (`configureChatOverflow` wird nur in `dev.tsx` aufgerufen) und die Route ist nicht da
  (`index.mjs` nur im Dev-Block). Der Writer arbeitet dann exakt wie vorher. Der erste Test vergleicht die geschriebene Liste
  byte-genau mit und ohne ausgeschalteten Wächter.
- **Für eine spätere Übernahme in Live** (mit `dev-promote`): Kern-Dateien sind `session-persist.ts`, `chat-overflow.ts`, `store.ts`,
  `ui.tsx`, `server/chat-overflow.mjs`, die neun `fix-*.json` und der Test. Damit der Schutz in Live **wirkt**, braucht es noch eine
  kleine Entscheidung: Route und `configureChatOverflow({ enabled: true })` auch ohne Dev einschalten. Das ist bewusst nicht
  gemacht, weil Live sonst nicht mehr byte-gleich wäre.

## Geänderte Dateien

| Datei | Was |
| --- | --- |
| `apps/host/src/lib/session-persist.ts` | optionaler Wächter `overflow` im Writer; ohne ihn unverändert |
| `apps/host/src/lib/chat-overflow.ts` (neu) | Wächter: gesicherte Versionen, Sicherung über den Host, Fehler und Wiederholung |
| `apps/host/src/lib/store.ts` | Wächter am Writer, Hinweis-Zustand `chatOverflowNotice`, Warnung, Neuschreiben nach Sicherung |
| `apps/host/src/components/ui.tsx` | Hinweis bei den Toasts, bleibt bis "Schließen" |
| `apps/host/src/locales/fix-*.json` | drei Texte in allen neun Sprachen |
| `apps/host/server/chat-overflow.mjs` (neu) | `GET`/`POST /api/chat-overflow`, Datei schreiben und auflisten |
| `apps/host/server/index.mjs` | Route im Dev-Block, nur wenn `chatOverflowGuard` an ist; Rescue liest auch den Overflow-Ordner |
| `apps/host/server/dev-mode.mjs`, `scripts/dev-validate.mjs`, `dev.config.json` | Schalter `chatOverflowGuard` |
| `apps/host/server/dev-rescue.mjs`, `src/dev/DevPanel.tsx` | Rescue-Tab zeigt und lädt `overflow-*.json` (als neuester Chat) |
| `apps/host/src/dev/dev.tsx`, `src/lib/app-mode.ts` | Wächter in Dev einschalten |
| `apps/host/tests/chat-overflow.test.mjs` (neu) | 8 Tests |

## Testergebnisse

- `node --test tests/chat-overflow.test.mjs`: 8 bestanden. Geprüft wird:
  - Ohne Wächter ist das Ergebnis byte-gleich, und die Grenze ist unverändert.
  - Ein Chat ohne Sicherung bleibt in der Liste und wird vollständig gesendet, ältester zuerst.
  - Nach der Sicherung fällt er heraus und wird nicht doppelt gesendet.
  - Ein geänderter Chat wird neu gesichert.
  - Bei einer fehlgeschlagenen Sicherung bleibt der Chat, es gibt eine Warnung, und der nächste Versuch kommt erst nach der Wartezeit.
  - Angeheftete Chats gehen zuletzt, und der neueste bleibt immer.
  - Der Host schreibt die Datei atomar, ohne Titel im Namen und nur einmal pro Version, und Rescue findet sie.
  - Die Config-Werte werden korrekt gelesen.
- `npm run typecheck` (apps/host): ohne Fehler.
- `npm test` (apps/host): 1200 bestanden, 0 fehlgeschlagen, 107 übersprungen (vorher 1192; dazu kommen die 8 neuen Tests).
  Ein erster Lauf fand einen Fehler in meinem französischen Text (gerades Apostroph, Leerzeichen vor ";"); behoben.
- Validator: 0 Fehler, 0 Warnungen.
- **Nicht gelaufen**: `dev-check`, pytest (der Python-Teil ist nicht betroffen; den vollständigen Lauf startest du selbst mit
  geschlossener Live-App). Nicht von Hand im Fenster durchgespielt, weil dafür ein neuer Build nötig ist (siehe unten).

## Ausprobieren

1. Einmal `.\dev-build.bat -Install` (die installierte Dev-App bekommt den Schutz erst mit einem neuen Build).
2. In `dev.config.json` `"chatOverflowGuard": { "enabled": true, "maxTotalChars": 30000 }` setzen, Dev-App neu starten.
3. In Strata Dev drei, vier Test-Chats mit großem Anhang oder langem eingefügtem Text anlegen (zusammen deutlich über 30.000 Zeichen).
4. Der Hinweis erscheint; die Dateien liegen in `%LOCALAPPDATA%\StrataDev\home\chat-overflow`. Nach Strg+R fehlen die ausgelagerten
   Chats in der Seitenleiste.
5. Dev-Panel → Rescue → `overflow-*.json` → "Laden": Der Chat ist zurück.
6. Danach `maxTotalChars` wieder entfernen.
