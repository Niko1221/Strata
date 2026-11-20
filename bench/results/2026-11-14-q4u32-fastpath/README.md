# q4u32 als int8-Ersatz: der Fast-Path und die Laufzeitmessung (Vorregistrierung, 2026-11-14)

Diese Datei legt fest, **was gebaut, was gemessen und was behauptet wird**, bevor der erste Bau
startet. Sie deckt zwei Dinge ab, die bisher getrennt liefen:

* **Serie #1 — die offene Laufzeitkontrolle.** Für `q4u32` gibt es bis heute **keine** Laufzeitmessung.
  Die 2,25× aus Tabelle 2 des Papiers gehören dem Level-Tabellen-Arm, nicht `q4u32`.
* **Der Fast-Path.** `q4u32` ist affine und byte-identisch zum `q4_0`-Layout, wird vom Prompt-Pfad
  aber abgelehnt (`engine/src/kernels/cuda/qsa_prompt_attn.cu:2526`). Auf HIP nimmt heute **kein** q4-Codec den
  Matrix-Core-Pfad.

**Branch:** `feature_q4u32_as_an_int8_replacement_for_all`, Basis `ac33b0c4b` (von
`daves_kv_q4_quant_with_int8_quality`, Paper 0.24). Der Branch ist angelegt und gepusht.

**Die zwei Entscheidungen vom 2026-11-14:**

1. **B3 ist entschieden: das Fenster kommt auf den Fast-Path.** Der Produktarm (`q4u32` + Fenster) darf
   den Fast-Path nicht verlieren. B3 ist damit im Umfang; die Wahl des Weges (drei Varianten in B3)
   fällt mit B2.
2. **`STRATA_HIP_WMMA=1` wird in Serie #1 gesetzt** (G-R6). Nach der eigenen A/B-Messung vom 08.11. ist
   der Schalter auf **dieser** Karte allerdings ein Decode-Hebel und **kein** Prompt-Hebel (0.3) — er
   wird gesetzt, damit jeder Arm sagt, welchen Pfad er lief, aber er ist nicht das, worauf wir warten.
   Für die **Qualitätsketten** bleibt er aus (er ändert die Ausgabebits).

**Reihenfolge:** Teil A misst den **Zustand heute** (Baseline, Teil B ist noch nicht gebaut). Teil B
baut den Fast-Path. Teil A2 misst denselben Harness **nach** dem Bau. Der Vergleich A1 gegen A2 ist
die Messung der Engine-Änderung. Teil C ist die Optimierung und beginnt erst, wenn A2 vorliegt.

---

## Teil 0 — Ausgangslage

### 0.1 Gemessen (heute, diese Maschine, ein Test, keine Vorregistrierung nötig)

`build-hip-port\hip_prompt_attn_wmma.exe` — der HIP-Build von `engine/src/kernels/qsa_prompt_attn_parity.cpp`,
künstliche Pools, keine Modelllast:

| Kontext / Queries | FP32-Prompt-Kern | Matrix-Core-Kern | Faktor |
|---|---:|---:|---:|
| 32.768 / 2.048 | 94,593 ms | 8,901 ms | **10,63×** |
| 1.500 / 1.500 | 32,353 ms | 3,388 ms | 9,55× |
| 2.100 / 256 | 14,663 ms | 1,683 ms | 8,71× |

je Chunk. Parität gegen FP64: 3,17e-06 (Matrix-Core) gegen 2,51e-06 (FP32); **0 Fehler**. Das ist der
Preis, den ein q4-Codec zahlt, wenn er den Matrix-Core-Pfad nicht bekommt — aber Achtung: dieser
Vergleich ist **Prompt-Kern gegen Prompt-Kern**, nicht der Decode-Fallback aus Tabelle 2. Die beiden
Zahlen sind nicht dasselbe und werden nicht vermischt.

### 0.2 Gelesen (statische Analyse, keine Messung)

* `q4u32` = `Q4Mode::kUniformLs`: der Block ist `block_q4_0` (18 Byte je 32 Werte: fp16-Skala + 16 Byte
  Nibble-Codes), die Rekonstruktion `v = (code−8)·s` — affine, **keine Tabelle, keine Konstante**
  (`engine/include/strata/kernels/kv_q4.hpp:28-34`, `engine/src/kernels/cuda/qsa_decode_attn.cu:105-117`).
* Der CUDA-Prompt-Pfad liest genau dieses Layout bereits: `KV_MODE 4` = q4_0, „mode 1 with a scale per
  32 values — each q4_0 block's codes enter as exact int8 (code−8) and its fp16 scale is applied in
  FP32" (`qsa_prompt_attn.cu:93-95`).
* Der Dispatch lehnt `kUniformLs` ab (`qsa_prompt_attn.cu:2526`). Die Begründung im Kommentar — die
  Level-Tabellen brauchen einen anderen Leser, der Shear trägt eine Konstante je Block — trifft den
  Uniform-Grid **nicht**. Der Header sagt das Gegenteil: *„kUniformLs is affine, so it rides the
  prompt fast path"* (`kv_q4.hpp:112-113`).
* Auf HIP gilt der ganze `KV_MODE`-Zweig nicht (`qsa_prompt_attn.cu:2505-2514`): dort wird **nur** der
  int8-Weg (Matrix-Core) angeboten, jeder q4-Modus fällt in den Decode-Kernel zurück.
* Auf Windows ist der Matrix-Core-Pfad **aus** (Opt-in): „the prompt attention on matrix cores is off by
  default on Windows (opt-in until tested on Windows); STRATA_HIP_WMMA=1 turns it on (gfx1200)" — so
  steht es in unseren eigenen 128K-Logs vom 13.11.

### 0.3 Der WMMA-Schalter: unsere eigene Messung (2026-11-08) und was daraus folgt

Der Schalter ist eine **Laufzeitvariable**, kein Build-Flag, und er ist **opt-in** (auf Windows aus, auf
gfx12-Linux an). Er ist unbedenklich einzusetzen — mit der Einschränkung, die die Dokumentation nennt:
er ändert die Ausgabebits (nicht bitweise; der Greedy-Text weicht ab etwa Token 50 ab). Für die
**Qualitätsketten** wird er deshalb **nicht** gesetzt (Regel vom 08.11.: die Kette soll mit P1
vergleichbar bleiben), für **Laufzeitläufe** schon.

Unsere eigene A/B-Messung vom 08.11. (Suite `standard-v1`, Swift IQ3_XXS, `--kv int8`, Kontext 32.768,
ROCm 7.2, Engine 0.1.47, RX 9060 XT gfx1200; Rohdaten unter
`C:\Users\krone\git\Strata-data\bench\2026-11-08-wmma-int8\`), Prompt-Raten in tok/s:

| Prompt | WMMA aus | WMMA an |
|---|---:|---:|
| 4k (r1/r2/r3) | 613,08 / 479,40 / 478,94 | 478,73 / 476,73 / 478,01 |
| 16k (r1/r2/r3) | 608,53 / 564,33 / 565,36 | 565,45 / 566,11 / 565,43 |

**Die Prompt-Raten sind gleich** (4k: 479,40 gegen 478,01; 16k: 565,36 gegen 565,45). Die erste
Wiederholung jedes Arms ist ein Ausreißer (613 bzw. 608) und trägt keinen Schaltereffekt; sie ist die
Quelle des scheinbaren „×1,12" im Plan vom 08.11. Im Decode: +6,9 % (4k), ≈0 (16k), −3,4 % (short) —
mit **überlappenden Spannen**.

**Die Lesart: der WMMA-Schalter ist auf dieser Karte ein Decode-Hebel, kein Prompt-Hebel.** Unser Preis
(die 2,25×) ist aber ein **Prompt**-Preis. Das ist genau die Aussage in
`docs/research/KV_FASTPATH_LITERATURE.md`, Abschnitt 4a: *„Der Fast-Path ist die Struktur (Abschnitt 1),
nicht die Matrix-Cores."*

**Folge für diesen Plan:** der Schalter wird in Serie #1 gesetzt (G-R6, damit jeder Arm sagt, welchen
Pfad er lief), aber er ist **nicht** das, worauf wir warten. Erwartet wird die Wirkung der **Struktur**
(B1/B2/B3), und die wird an **zwei** Stellen gemessen (G-R7).

---

## Teil A — Serie #1: die Laufzeitmessung

### A.1 Die Frage

**Wie schnell liest `q4u32` den Prompt, und läuft er dabei auf dem Fast-Path?** Heute ist beides
unbekannt: der 128K-Lauf hat nur *eine* Prompt-Rate je Arm geliefert (26,0 gegen 28,0 tok/s), und die
trennt die Arme nicht.

### A.2 Was gemessen wird

* **Metrik 1 (das, was der Benutzer sieht):** die Prompt-Read-Rate aus dem Engine-Log —
  `strata serve: prompt N tokens = 0 reused + N read in X ms (Y tok/s)`. Das ist die Prompt-Lesedauer
  ohne Decode-Anteil (der Decode ist bei 32K neun bis zehn Token und vernachlässigbar).
* **Metrik 2 (das, was die Arbeit betrifft):** die **Kernel-Zeit der Prompt-Attention** selbst. Auf
  dieser 16-GB-Karte kommt ein großer Teil der Prompt-Zeit vom Nachladen der Experten (Plan vom
  08.11.), deshalb kann die End-to-End-Rate flach bleiben, während die Attention-Kernel-Zeit fällt.
  Die Kernel-Zeit kommt aus dem Paritätstest (`hip_prompt_attn_wmma.exe`, bzw. `qsa_prompt_attn_parity`),
  der sie je Chunk selbst timt.
* **Harness:** `tools/needle_bench.py --lengths 32k --depths 10,50,90` gegen einen laufenden Server.
  Der Heuhaufen ist der Text dieses Repositories in fester Reihenfolge, die Codewörter kommen aus
  `random.Random(7)` — **jeder Arm sieht dieselben Prompts**.
* **Protokoll:** ein Arm, eine Engine, frischer Server, 30 s zwischen den Engine-Starts, Maschine
  sonst idle. Drei Tiefen je Arm; berichtet werden Median und Spanne.
* **Nicht die Metrik:** die `seconds` aus `needles.json` sind die einmalige Wandzeit einer Anfrage. Sie
  sind ein Recall-Hinweis, **kein Durchsatz** (so steht es auch im Protokoll vom 24.10.).

### A.3 Die Arme

Alle am selben Tag, ein Build, eine Engine nach der anderen:

| Arm | `--kv` | Fenster | Zweck |
|---|---|---|---|
| `int8` | `int8` | aus | der Anker: der Fast-Path in Reinform |
| `q4_0` | `q4_0` | aus (das Fenster wird laut abgelehnt) | der Stock-Gitter-Vergleich |
| `q4lm8v` | `q4lm8v` | aus | **Kontrollarm**: der Level-Tabellen-Arm, dessen 2,25× der Harness sehen muss |
| `q4u32` | `q4u32` | aus | der Uniform-Grid **ohne** Fenster (Bulk, byte-gleich zum q4_0-Layout) |
| `q4u32`+Fenster | `q4u32` | S 256 / R 64 | die Produktkonfiguration; **vor B3 ohne Fast-Path** |
| `q4lm32w` | `q4lm32w` | S 256 / R 64 | der Produktarm des Papiers (16/16 + Fenster) |

Der Fenster-Arm trägt `STRATA_KV_WINDOW=1`, `STRATA_KV_WINDOW_S=256`, `STRATA_KV_WINDOW_R=64`;
der Fenster-Arm muss die Logzeile „ring sized" zeigen, die anderen nicht (das prüft der Treiber).

### A.4 Die Gatter (vorab fixiert)

* **G-R1 (Struktur — der Pfad muss sichtbar sein).** Für jede Behauptung „der Arm lief auf dem
  Fast-Path" braucht es eine **Logzeile**, die das belegt. **Die gibt es heute nicht.** Deshalb wird
  sie Teil von B1/B2: der Prompt-Pfad meldet sich, wenn er einen q4-Uniform-Arm bedient. **Ohne diese
  Zeile ist jede Zeitangabe in diesem Plan strukturell nicht zuzuordnen** — sie wird dann als
  „Zeit des Fallbacks" berichtet und nicht als Fast-Path-Zeit.
* **G-R2 (Leistung — das eigentliche Ziel).** `q4u32` ohne Fenster ist am selben Tag **nicht langsamer**
  als `int8` (Median der drei Tiefen). Die Byte-Hypothese sagt mehr: der 4-Bit-Bulk liest die Hälfte
  der KV-Bytes, also *schneller*. Verfehlt der Arm G-R2, ist die Behauptung „q4u32 ist ein int8-Ersatz
  auch in der Laufzeit" **widerlegt** und wird so berichtet.
* **G-R3 (Kontrolle — der Harness muss sensibel sein).** `q4lm8v` muss am selben Tag **deutlich
  langsamer** sein als `int8` (sein Prompt läuft in den Decode-Kernel). Ist er das nicht, sieht der
  Harness den Unterschied nicht und **alle Zeiten in diesem Plan sind wertlos** — dann wird die Serie
  abgebrochen und der Fehler gesucht, nicht die Arme gerechtfertigt.
* **G-R4 (Streuung).** Drei Tiefen je Arm. Ein Unterschied unter 3 % wird **kein** Unterschied
  genannt. Jede Zahl bekommt Median und Spanne.
* **G-R5 (Fenster).** Der Fenster-Arm wird **getrennt** berichtet: vor B3 läuft er per Definition über
  den Fallback. Die „ring sized"-Zeile muss da sein, sonst ist der Arm kein Fenster-Arm.
* **G-R6 (Matrix-Core-Pfad).** Auf Windows ist der Matrix-Core-Pfad Opt-in. Jeder Arm setzt
  `STRATA_HIP_WMMA=1`, und die Logzeile („the prompt attention on matrix cores on (gfx1200)") wird mit
  protokolliert. **Ohne diese Zeile gilt der Lauf als FP32-Lauf** und wird nicht mit dem
  Matrix-Core-Pfad verglichen.
* **G-R7 (zwei Zahlen, oder keine).** Jeder Arm bekommt **beide** Metriken: die End-to-End-Prompt-Rate
  **und** die Kernel-Zeit der Prompt-Attention. Bleibt die End-to-End-Rate flach, während die
  Kernel-Zeit fällt, wird **genau das** berichtet — nicht „kein Unterschied". Wird nur eine der beiden
  gemessen, gilt die Arm-Aussage als unvollständig. (Grund: 0.3 — auf dieser Karte ist der WMMA-Schalter
  kein Prompt-Hebel; der Fast-Path ist die Struktur, und die zeigt sich in der Kernel-Zeit.)
* **G-R8 (die Karte muss frei sein).** Der VRAM-Budget ist die Fallgrube dieser Serie: die Engine
  bekommt von Windows nur so viel VRAM, wie kein fremder Prozess belegt. Am 14.11. um 13:13 hat **Unsloth
  Studio** der Karte rund 7,5 GB genommen — die Engine sah 8.766 von 16.304 MiB statt 15.419, der
  Expert-Cache bekam **0 Slots**, und der Prompt lief über die SSD statt über die VRAM: 36,6 tok/s
  statt rund 557 (A.9). **Gate:** jeder Arm protokolliert die Budget-Zeile und die Slot-Zahl; unter
  14.000 MiB Budget oder unter 1.000 Expert-Slots ist der Arm **ungültig** (nicht „langsam") und wird
  wiederholt.
  **Ein Fehler, den das Gate selbst gemacht hat (und der behoben ist):** das Engine-Log wird
  **angehängt** — jeder Arm eines Laufs landet in derselben Datei, mit einer „engine started"-Marke pro
  Lauf. Der Wächter las die *erste* Budgetzeile und damit die Zeile des **verworfenen** Laufs vom 13:13
  (8.766 MiB): der `int8`-Arm sah sich dadurch kaputt an, obwohl sein eigener Lauf 15.419 MiB und 2.611
  Slots hatte. Der Wächter liest jetzt den Text **nach der letzten** „engine started"-Marke — und der
  Treiber startet vorher einen kurzen **Probe-Start**, der das Budget liest, damit der erste echte Arm
  nicht auf einem Zustand landet, der dem Lauf davor gehört.

### A.5 Die Befehle

Der Treiber ist eine Abwandlung von `bench/results/2026-11-10-merge-reeval/run-b2.py` (derselbe
Harness, dieselben 30 s, dieselbe Ein-GPU-Job-Sperre) und liegt als
`bench/results/2026-11-14-q4u32-fastpath/run-runtime.py` bei:

```
python -m serve.server --engine strata --config bench/configs/needle-...-<arm>.json --port 7603 --no-api-key
python tools/needle_bench.py --url http://127.0.0.1:7603 --lengths 32k --depths 10,50,90 --out needles-<arm>.json
```

Der Treiber druckt vor jedem Arm die SHA-256 des Engines, prüft die Ein-GPU-Job-Sperre
(`tasklist`), wartet, bis der Engine weg ist, und liest danach die Prompt-Rate aus dem Log.

### A.6 Die Fallen

1. **Der Matrix-Core-Pfad ist auf Windows aus.** Ohne `STRATA_HIP_WMMA=1` messen wir den FP32-Kern.
   Das ist der wichtigste Fang: die 128K-Läufe vom 13.11. liefen **ohne** Matrix-Core.
2. **Heute läuft der Prompt jedes q4-Arms über den Decode-Kernel.** Vor B2 ist die `q4u32`-Zeit also
   die Fallback-Zeit. Sie wird gemessen und so benannt — sie ist die Zahl, die ein Benutzer heute sieht.
3. **Die Fenster-Armen sind nicht mit den Bulk-Armen vergleichbar.** Der Overlay kostet einen
   Side-Pool und einen Zweig je Zelle; er wird separat berichtet.
4. **VRAM.** Der 4-Bit-Bulk braucht die Hälfte, das Fenster einen Side-Pool daneben. Die VRAM-Budget-
   Zeile wird mitprotokolliert (sie steht im Log).
5. **Ein Build, ein Tag.** Ein Arm, der auf einem anderen Build läuft, wird verworfen, nicht
   nachgerechnet.
6. **Die Karte muss frei sein** (G-R8). Ein fremder Prozess auf der GPU macht den Prompt zum
   SSD-Lauf; die Rate misst dann die SSD. Das ist am 14.11. um 13:13 passiert (A.9).
7. **Das Engine-Log wird angehängt.** Jeder Arm eines Laufs landet in derselben Datei. Wer daraus
   liest, muss den Text nach der letzten „engine started"-Marke nehmen — sonst liest man den Lauf
   davor (genau das hat der G-R8-Wächter getroffen).

### A.7 Die Rohdaten

Nicht im Repo: `C:\Users\krone\git\Strata-data\bench\2026-11-14-q4u32-fastpath\` — je Arm das
`needles-<arm>.json`, das `needle-<arm>.engine.log` (mit der SHA-256-Zeile), die Server-Ausgabe und die
SHA-256 der Rohdateien. Im Repo bleibt **dieses README** plus der Treiber.

### A.8 Die beiden Läufe

* **A1 — Baseline, vor dem Build.** Alle sechs Arme, ein Build (0.1.59.1, unverändert). Ergebnis: die
  heutigen Zeiten, mit dem Kontrollarm G-R3.
* **A2 — nach Teil B.** Dieselben sechs Arme, der neue Build, derselbe Tag, wenn möglich. Der
  Vergleich A1/A2 ist die Messung der Engine-Änderung; die Zahlen von A1 und A2 stehen **nebeneinander**,
  nie einzeln.

### A.9 Der verworfene erste Lauf (14.11.) — und die Referenz, die er brauchte

Der erste A1-Versuch am 14.11. ist **verworfen**, und der Grund steht im Engine-Log, nicht im
Rauschen:

| Zeile im Log | Referenzlauf (b2, 10./11.11.) | der erste Versuch (14.11.) |
|---|---|---|
| `Windows budgets … of this card's 16.304 MiB` | 15.419 MiB | **8.766 MiB** |
| `expert cache … slots` | 2.664 Slots (int8), 3.239 (Fenster-Arm) | **0 Slots** |
| Prompt-Read-Rate (32K, drei Tiefen) | int8 **557,2 / 578,1 / 361,7** tok/s; Fenster-Arm (q4lm8v) **264,2 / 260,0 / 265,0** tok/s | int8 **≈ 37 tok/s** (11.520 Tokens in 315 s) |

Die 7.538 MiB Differenz sind „the desktop's and other programs' share" (so der Kommentar in
`engine/src/core/device.cu:209`) — ein fremder Prozess auf der Karte, zur Stunde der Messung **Unsloth
Studio**. Folge: 0 Expert-Slots, der Output-Kopf wanderte in den geheften Hostspeicher, und der Prompt
las die Experten über die SSD statt über die VRAM. Gemessen wurde damit die SSD, nicht der KV-Pfad.

**Zwei Dinge folgen daraus.** Erstens: die Referenzzahlen von Tabelle 2 des Papiers sind der
**b2-Lauf vom 10./11.11.** auf dieser Maschine mit diesem Harness — int8 rund 557–578 tok/s, der
Fenster-Arm des Level-Tabellen-Codecs rund 260–265 tok/s, also ein Faktor **2,11×**. Das ist der Wert,
gegen den A1 und A2 geprüft werden, und er ist **heute nicht** zu wiederholen, solange ein fremder
Prozess die Karte belegt. Zweitens: G-R8 existiert erst wegen dieses Laufs, und der Treiber prüft es
jetzt von sich aus (Budget ≥ 14.000 MiB, ≥ 1.000 Expert-Slots; sonst Arm ungültig).

### A.10 A1 — die Ergebnisse (14.11.2026, Build 0.1.59.1, SHA `fdd3823e…`)

Ein Arm, eine Engine, 30 s dazwischen. Die Metrik ist die Prompt-Read-Rate aus dem Engine-Log
(`prompt 30782 tokens = 0 reused + 30782 read in … ms (… tok/s)`), 32K-Nadel, Tiefen 10/50/90 %, in
allen Armen 3 von 3 Nadeln gefunden. VRAM-Budget und Expert-Cache-Slots stehen je Arm daneben (G-R8).

| Arm | 10 % | 50 % | 90 % | Median | Fenster | VRAM-Budget | Expert-Slots | Recall |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `int8` (Anker) | — | — | — | **ungültig** | nein | 8.766 MiB | 0 | — |
| `q4_0` (ausgeliefert) | 606,1 | 609,5 | 607,5 | **607,5** | nein | 15.419 | 3.015 | 3/3 |
| `q4u32` (uniform, ohne Fenster) | 263,1 | 264,4 | 263,7 | **263,7** | nein | 15.419 | 3.015 | 3/3 |
| `q4lm8v` (Level-Tabelle 16/8) | 256,9 | 258,2 | 257,9 | **257,9** | nein | 15.419 | 3.019 | 3/3 |
| `q4lm32` (Level-Tabelle 16/16) | 257,2 | 268,5 | 268,4 | **268,4** | nein | 15.419 | 3.015 | 3/3 |
| `q4u32` + Fenster | 256,1 | 270,2 | 260,3 | **260,3** | ja | 15.419 | 2.947 | 3/3 |
| `q4lm32w` (Produktarm) | 266,0 | 256,6 | 256,6 | **256,6** | ja | 15.419 | 2.947 | 3/3 |

**Der `int8`-Anker ist ungültig** (G-R8: Budget 8.766 MiB, 0 Expert-Slots) und wird wiederholt — der
Wächter hat den Arm vor den Nadeln abgebrochen, das ist richtig so, aber der Anker fehlt (A.11).

**Der Befund, der die bisherige Erklärung kippt.** Der ausgelieferte Stock-Gitter-Arm `q4_0` liest mit
**607,5 tok/s**, jeder andere 4-Bit-Arm mit **256,6 bis 263,7 tok/s** — ein Faktor **2,32**. Der
Uniform-Grid `q4u32` **ohne** Level-Tabelle liegt bei **263,7 tok/s**, also praktisch auf dem Niveau des
Level-Tabellen-Arms (257,9) und **2,31× unter dem Stock-Arm**. Die Messung reproduziert Tabelle 2 des
Papiers (dort: `q4_0` 616 gegen `q4lm8v` 265 tok/s).

Damit steht die Erklärung aus §5 des Papiers infrage: dort wird die 2,25× **allein** der Level-Tabelle
zugeschrieben („the prompt fast path declines the Lloyd modes structurally … so the decode kernel runs
instead"). **Gemessen ist: den Preis zahlt jeder Modus, der nicht der ausgelieferte `kUniform` ist** —
auch der tabellenfreie Uniform-Grid. Zwischen `kUniform` (kv_mode 2) und `kUniformLs` (kv_mode 6) gibt
es im Code genau **einen** Unterschied: die fensterfähigen Instantiationen (KV_MODE ≥ 4) tragen den
Overlay-Zweig, der ausgelieferte nicht (statische Analyse,
`engine/src/kernels/cuda/qsa_decode_attn.cu:213`). Das Fenster selbst kostet **nichts messbares**: `q4u32` mit
Fenster (260,3) liegt auf dem Niveau von `q4u32` ohne Fenster (263,7) — der Unterschied liegt im
Streubereich. Der Preis sitzt also an der **Instantiation**, nicht am Codec und nicht am Fenster.

**Das gibt Teil B ein messbares Ziel:** wenn der `q4u32`-Arm seinen Prompt auf dem Matrix-Core-Pfad
bedient (B2), statt in der Decode-Instantiation mit dem Overlay-Zweig, muss er auf das Niveau des
Stock-Arms (≈ 600 tok/s) kommen. Das ist die Hypothese, die A2 prüft.

**Der Mechanik-Arm `q4lm32` entscheidet, wo der Preis sitzt** (der Lloyd-Arm mit der vollen
16-Level-Tabelle, ohne Fenster): **268,4 tok/s** — also **auf dem Niveau des tabellenfreien
Uniform-Grids** (263,7) und **2,27× unter dem Stock-Arm** (607,5). Damit ist die Level-Tabelle als
Ursache **ausgeschlossen**: ein Arm mit Tabelle und ein Arm ohne Tabelle kosten dasselbe. Übrig bleibt
der Unterschied zwischen den Instantiationen — `kUniform` (kv_mode 2, ohne Overlay-Zweig) gegen die
fensterfähigen (KV_MODE ≥ 4, mit Overlay-Zweig). Das ist genau der Zweig, den B2 umgeht: der
`q4u32`-Arm bedient seinen Prompt dann auf dem Matrix-Core-Pfad statt in der Decode-Instantiation.

### A.11 A2 — nach dem Bau (B2), und was der Build bringt

Der neue Build (SHA `497df90d…`) mit dem 4-Bit-Pfad im Matrix-Core-Prompt-Kern. Die Arme, die der
Build nicht ändert, laufen wie gemessen:

| Arm | A1 (Build `fdd3823e…`) | A2 (Build `497df90d…`) | Differenz |
|---|---:|---:|---|
| `int8` | 634,1 | 644,0 | +1,6 % (im Streubereich) |
| `q4_0` | 607,5 | 623,7 | +2,7 % (im Streubereich) |
| `q4u32` | 263,7 | **ungültig** | der Engine hängt (siehe unten) |

**Der `int8`-Pfad ist bitgleich geblieben:** die drei Prüfsummen des Paritätstests
(`84c178024d79e375`, `dc3b41638fab8f77`, `d791e4bfb2fd064d`) sind vor und nach dem Build gleich, und
der Paritätstest meldet **0 Fehler**. Das ist das Gate G-R7 für den Default-Pfad ✓.

**Der 4-Bit-Pfad besteht sein Paritätsgate** (der neue Fall `fmt 3` im Paritätstest, dieselben Blöcke,
der Modus „uniformes Gitter mit gefitteter Stufe"):

```
PASS q4u32 ctx 32768, 2048 queries: vs FP64 old 4e-06 new 9e-06 (output scale 7.61);
     new vs old 1.46e-05 (1.8e-06 of scale); 97.312 -> 19.005 ms per chunk (5.12x)
```

**5,12×** gegenüber dem Decode-Kern im selben Harness — die Wirkung des B2 auf die *Kernel-Zeit* ist
damit gemessen ✓.

**Im Engine hängt der Arm.** Das Symptom, wörtlich aus dem Engine-Log:

```
strata serve: no progress for 60 s during a request (reading the prompt (verify windows), from token 30776)
              - stopping the engine so the server starts it again
verify: an earlier window never finished on the GPU (#267); restart the engine
```

Der Server antwortet danach mit HTTP 503, der Needle-Harness findet 0 von 3 Nadeln. Der Aufruf hängt
**im Kernel**: die GPU war nach dem Abbruch noch 124 s busy („released the verify window's GPU waits:
the GPU finished in 124 ms"). **Die Reproduktion ist billig**: 4k-Nadel gegen den `q4u32`-Arm ≈ 70 s
pro Anfrage (der 32k-Lauf braucht 391 s und drei Neustarts).

**Was ausgeschlossen ist (jedes mit einer Messung):**

| Hypothese | Test | Ergebnis |
|---|---|---|
| der Build hat allgemein etwas kaputtgemacht | die int8-Prüfsummen | **unverändert** ✓ |
| der Hänger ist nicht der 4-Bit-Pfad | `q4u32` mit `STRATA_HIP_WMMA=0` (Decode-Pfad) | **läuft**, 3/3 Nadeln ✓ |
| der ungated Prompt-Pfad ist auf HIP kaputt | `int8` mit `STRATA_QSA_FUSE_GATE=0` | **läuft**, 3/3 ✓ |
| die Rotation ist es | `int8` mit `STRATA_KV_ROT=1` | **läuft**, 3/3 ✓ |
| der Kernel ist falsch | der Paritätstest mit dem q4-Pool (32.768, 512, 64 Kontext) | **besteht alle drei** ✓ |

Übrig bleibt der Unterschied zwischen dem **synthetischen** Pool des Paritätsharness (zufällige Codes,
Skalen in [0,01; 1] mit Zufallszeichen) und dem **echten** Pool des Engines (die LS-gefitten Stufen, die
echten Codes, die rotierte Query). Der Kernel ist also mit synthetischen Daten richtig und hängt mit den
echten — die Ursache ist **nicht isoliert**.

**Deshalb ist der Pfad opt-in** (`STRATA_KV_Q4_PROMPT=1`), und der Default-Pfad bleibt byte-identisch
zum letzten Release: der `q4u32`-Arm läuft ohne die Variable über den Decode-Kern (13 s je 4k-Prompt,
3/3 Nadeln ✓). Das ist die Zwischenstand-Regel des Repos („a new feature is opt-in, and the default
path stays byte-identical to the last release") ✓.

**Nebenbefund mit eigenem Wert:** der Decode-Kern kannte die Modi 5 (`q4u32z`) und 6 (`q4u32`) im
Warp-Scoring-Pfad **nicht** — sie fielen auf den fp16-Kern zurück und lasen Pools, die ein 4-Bit-Zustand
nicht hat (`k_pool`/`v_pool` sind null): der Start schlug mit „unspecified launch failure" fehl. Nur die
Lane-Cell-Instantiation (`STRATA_ATTN_LANECELL=1`, auf gfx12 per Default an) kannte sie. Das ist ein
vorbestehender Bug, unabhängig von B2, und jetzt behoben ✓ — er fällt nur auf Karten oder
Konfigurationen auf, die den Lane-Cell-Pfad nicht nehmen.

### A.12 — Der Hänger braucht **keine** Länge (15.11.2026)

Drei Läufe, alle mit `STRATA_KV_Q4_PROMPT=1` und `needle-fastpath-q4u32.json`:

1. **4k-Nadel: hängt auch.** Bei **3.321 von 3.328** Token, 77 s, alle drei Tiefen HTTP 503. Die Länge
   ist damit **nicht** die Bedingung, und die Reproduktion kostet 77 s statt 391 s.
2. **4k mit `QA_Q4TRACE=1`: der Bulk läuft, danach nichts mehr.** Die zwölf letzten
   `[q4u32] launch`-Zeilen der Session sagen alle „cells 1 chunks 1" — das ist die **erste** Query
   eines Prompt-Chunks (Position 0, eine Zelle), also der Bulk-Prompt (zwölf Aufrufe = die
   QSA-Schichten). Danach kommt **keine** Zeile. Der Trace druckt **vor** dem Launch: der Host hat den
   nächsten `launch_wmma` nie erreicht. Dazwischen liegt nur das 6-Token-Verify-Fenster
   (`strata verify: captured the 6-token window … upload no error, sync no error`, 117,8 ms) und die
   Stall-Meldung „reading the prompt (verify windows), from token 3321".
3. **Der isolierte Kern mit derselben Form läuft.** 3.328 Zellen, 6 Queries, mit Gate:
   `PASS … 0,502 → 0,488 ms per chunk (1,03×)`. Dazu die beiden kleinen Fälle unter dem Gate
   (64/64: 0,116 → 0,055, 2,10×; 512/512: 4,737 → 1,171, 4,04×), **FAILURES: 0**.

**Werkzeugfehler, der hätte täuschen können:** ohne `STRATA_KV_Q4_PROMPT=1` startet der Harness für
fmt 3 den **Decode**-Kern („the q4 arm runs the prefill in the decode kernel … opt-in") und druckt
trotzdem `PASS`. Seit dem Gate ist ein grüner q4-Fall also **kein** Beleg für den Matrix-Kern. Die drei
Zahlen oben sind **mit** Gate gefahren; die B2-Zahlen (32.768/512/64) stammen von **vor** dem Gate und
bleiben gültig — ein erneuter Lauf ohne Gate wäre aber ein falsches Grün. (Für fmt 1 setzt der Harness
`STRATA_HIP_WMMA` selbst; für fmt 3 fehlt das Gegenstück.)

**Was damit ausgeschlossen ist:** die Länge, die Grid-Größe, der Kernel auf synthetischen Daten und der
Bulk-Prompt selbst. Was bleibt, ist der **Übergang** vom Bulk-Prompt zum Verify-Fenster — die Stelle,
an der echter Pool, echte Selektions-IDs und Fenster-Pools zum ersten Mal zusammentreffen.

**Die nächsten drei Schritte** (billig nach teuer):
`STRATA_QSA_SELECT_DUMP=1` (kompiliert, nicht an) schreibt die echten Selektionen des hängenden Laufs —
damit lässt sich die Form im Harness nachspielen; ein Lauf mit `--prefill 512` zeigt, ob es an *jedem*
Chunk-Ende hängt oder nur am letzten; der Host-Stack liegt in `strata-stall-*.dmp` (auf dieser Maschine
ist kein `cdb`, also nicht lesbar).

### A.13 — Der Trace sagt jetzt die Form, und der Verdacht verschiebt sich (15.11.2026)

B6 hat gezeigt, dass der Trace `steps[0]` las (die Selektion der *ersten* Query) und damit „cells 1
chunks 1" für jeden Aufruf druckte — Bulk und Fenster gleich. Der Trace nennt jetzt `n_q`, `cap`, die
Route und die Zellen der ersten **und** letzten Query. Derselbe 4K-Lauf (Gate an, `QA_Q4TRACE=1`,
`STRATA_KV_Q4_PROMPT=1`):

```
[q4u32] launch: n_q 3321 cap 2051 ungated, cells 1..2051 chunks 1..65     (×12, eine je QSA-Schicht)
strata verify: captured the 6-token window (upload no error, sync no error) in 118.4 ms
strata serve: no progress for 60 s during a request (reading the prompt (verify windows), from token 3321)
```

Zwei Dinge stehen damit fest:

1. **Der Bulk-Prompt ist der Matrix-Pfad, vollständig** — 3.321 Queries, `cap` 2.051, 12 Aufrufe (eine je
   QSA-Schicht), keine Fehlermeldung. Die 12 Zeilen aus A.12 waren also wirklich der Bulk.
2. **Das Fenster ruft den 4-Bit-Kern überhaupt nicht.** Nach dem Capture kommt **keine** Trace-Zeile —
   und das ist jetzt eindeutig, weil jede q4-Instantiierung (auch die gated, B6) eine druckt. Der Grund
   steht im Code (statische Analyse): das Verify-Fenster geht über `qsa_decode_attn_batch` / `_rows` /
   `_slots` / `_split` (`verify.cpp` 2227–2330), nie über den Prompt-Pfad. **Die Aussage aus A.12, der
   Host stehe „vor dem nächsten `launch_wmma`", ist damit widerlegt** — es gibt keinen solchen Aufruf.
   Der Hänger liegt im **Decode-Kern des Fensters**, nachdem der Bulk über den Matrix-Pfad lief.

Damit ist die entscheidende Kontrolle die, die wir schon haben: `q4u32` mit `STRATA_HIP_WMMA=0` (Bulk
**und** Fenster über den Decode-Kern) läuft 3/3. Der Unterschied ist also nicht der Decode-Kern und
nicht das Fenster allein, sondern **der Matrix-Pfad davor**. Zwei Möglichkeiten bleiben, beide billig:

* der 4-Bit-Kern **schreibt** irgendwohin, wo das Fenster später liest (Out-of-Bounds an `cap`/`ids`);
* oder der letzte Bulk-Launch **endet nie**, und das „sync no error" des Captures gilt einer anderen
  Wartestelle.

Der Test, der das trennt, ist `HIP_LAUNCH_BLOCKING=1`: dann ist jeder Launch synchron, und die letzte
Zeile vor dem Stillstand ist der Launch, der nicht zurückkommt.

**Nebenbefund — und eine Korrektur an mir selbst:** ich hatte notiert, `cap` sei im Motor 2.051 und im
Harness die volle Kontextlänge. Das ist **falsch**: der Harness nimmt `qsa_selection_width(ctx)` =
`min(ctx, idx_top_k + idx_block − 1)` = `min(ctx, 2051)` (`engine/include/strata/kernels/qsa.hpp` 131–140),
also genau dieselbe 2.051, und die letzten 1.277 Queries eines 3.328er Laufs sättigen sie. Was der
Harness **nicht** prüft, ist ein Aufruf, in dem die Queries **verschiedene** Weiten haben (Position 0
mit `n = 1` bis Position 3.320 mit `n = 2.051` in einem Grid) — der Motor macht genau das (A.13, elf
Bulk-Aufrufe `n_q 3321`) — und den gated-Zweig (B6).

### A.14 — Der Hänger ist ein Doorbell-Deadlock im Fenster-Launch, und der Fast Path läuft (15.11.2026)

Vier Läufe, 4K und 32K gegen `needle-fastpath-q4u32.json`, alle mit `STRATA_KV_Q4_PROMPT=1`.

**1. `HIP_LAUNCH_BLOCKING=1` änderte nichts** (76 s statt 72 s, gleicher Stall) — der Schalter greift in
diesem HIP-Build also **nicht**; der Lauf ist damit **kein** Beleg gegen einen asynchronen Rest des
Bulks. Als Befund gestrichen.

**2. `STRATA_VERIFY_DEBUG=1` nennt den Ort** (Laufzeitvariable, kein Build):

```
verify dbg: captured; staging
verify dbg: staged; launching
   ... 60 s nichts ...
verify dbg: launched                 (erst nach dem Abbruch)
verify dbg: layer 0 rang … layer 15 served      (16 Zeilen in 35 ms)
```

Der Host steht **im Launch-Aufruf** des Fensters — nicht in einer Warteschleife danach, nicht in der
Attention.

**3. `STRATA_LAUNCH_PROBE=1` nennt den Grund:**

```
strata tdr-probe: the 6-token window graph has 1876 nodes, 1876 of them kernels
strata tdr-probe: window 0 T=6 pos=3321: launching
strata tdr-probe: window 0 T=6: the launch call has not returned after 50000 ms;
                  GPU rang 1, flags served 0 A 0 B 0       (bei 50 / 250 / … / 30000 ms unverändert)
strata tdr-probe: window 0 T=6: launch call returned in 62964.00 ms;
                  the GPU had rung 4 layer(s) and the host served 4294967295 at the return
```

Der Launch blockiert **in der Submission**, während die GPU auf die **Host-Doorbell** wartet (`A 0 B 0` =
Plan und Kopien sind noch nicht veröffentlicht). Der aufrufende Thread kann sie nicht bedienen, weil er
im Launch steht — ein Deadlock. Genau dafür gibt es `STRATA_VERIFY_ASYNC_LAUNCH=1` (der Graph startet auf
einem Helfer-Thread, der Aufrufer bedient die Doorbells; `verify.cpp` 3551: „a blocking instrumented
launch cannot also serve the graph's CPU doorbells"). Dass es normalerweise gutgeht, ist ein
**Wettlauf**: der Launch kehrt zurück, *bevor* die erste Doorbell fällig wird. Der 4-Bit-Pfad verliert
ihn.

**4. Mit dem Helfer-Thread läuft es — und zwar richtig.** Dieselbe Nadel, derselbe Build
(`1a63531b8` + Review), derselbe Schalter, nur der KV-Modus verschieden, drei Tiefen je Arm:

| Tiefe | `q4u32` (Matrix-Pfad) | `int8` | Verhältnis |
|---|---|---|---|
| 10 % (~30.782 Token) | **785,1 tok/s** | 626,6 tok/s | 1,25× |
| 50 % | 617,3 tok/s | 626,7 tok/s | 0,985× |
| 90 % | 619,4 tok/s | 620,8 tok/s | 0,998× |
| Nadeln | 3 von 3 | 3 von 3 | |

Alles Prompt-Lesen (30.782 / 30.783 Token); Generierung 27,5–27,6 gegen 35,0–45,3 tok/s (der
4-Bit-Pfad ist ein **Prompt**-Pfad, er ändert die Decode-Kerne nicht — die Differenz dort ist Rauschen
der Draft-Annahme: 6/6, 5/8, 6/11 gegen 6/6, 6/6, 6/9).

**Also: `q4u32` erreicht int8.** Zwei der drei Läufe liegen innerhalb von 0,4 % von int8, einer (der
erste der Serie) ist **25 % schneller** — und das ist **nicht** erklärt; ein Wiederholungslauf gehört
dazu, bevor daraus ein Satz wird. Gegen A.1 (263,7 tok/s, Build `fdd3823e…`, ohne den Pfad) sind es
**2,35×**.

Der Gewinn, der bleibt, ist der **Speicher**: `q4u32` braucht je Kopf 2 × (128/32) × 18 = 144 Byte
gegen int8's 2 × (128 + 2) = 264 Byte — **54,5 %** (aus den Layouts gerechnet: `kv_q4_bytes_per_head` in
`kv_q4.hpp` 127–130, `kv_q8_bytes_per_cell` in `kv_q8.hpp` 25–27; nicht gemessen). Gleiche Prompt-Rate,
gut halb so viel KV.

**Was das für den Default heißt:** der Hänger ist **nicht** der 4-Bit-Kern (das Fenster ruft ihn nie,
A.13) und kein Rechenfehler — die Nadeln werden mit dem Fast Path an allen drei Tiefen gefunden, die
Attention liest den Kontext also richtig. Es ist ein **vorbestehender Wettlauf im Fenster-Launch**, den
der schnellere Pfad verliert. Zwei Wege, und der erste ist der richtige:

1. den Launch reparieren (den Helfer-Thread zum Default machen, wo Doorbells bedient werden müssen, oder
   die Doorbell vor dem Submit veröffentlichen);
2. `q4u32` als Default erst danach — ein Umweg über eine Diagnose-Variable ist kein Default.

Die 619,4 tok/s gelten **mit** `STRATA_VERIFY_ASYNC_LAUNCH=1`; ohne sie hängt der Lauf bei 3.321 von
3.328 Token. Das gehört in jede Tabelle, die diese Zahl zitiert. Auch die 4K-Nadel lief so (3 von 3,
8 s je Anfrage).

### A.15 — Der Fix: der Launch gehört aus dem bedienenden Thread (15.11.2026)

Die Ursache aus A.14 stand schon im Header, nur nicht als Voreinstellung. `host_graph_launch.hpp` 11–14:
*„Instrumented cudaGraphLaunch may block until device work completes. A graph which waits for CPU
doorbells must therefore launch outside its service thread."* Genau das war der Default nicht: der
Helfer-Thread lief nur mit `STRATA_VERIFY_ASYNC_LAUNCH=1`.

**Die Änderung** (`verify.cpp`, eine Voreinstellung, kein neuer Pfad): der Helfer-Thread ist auf dem
**Windows-HIP-Build** der Default — dort ist der blockierende Submit gemessen —, auf allen anderen Builds
bleibt der Inline-Launch der Default (unverändert, CUDA also byte-gleich). `=0` nimmt den Inline-Launch
als A/B-Arm, `=1` den Helfer-Thread auch anderswo. Die Bedingung ist nicht „q4": **jedes** Fenster-Graph
wartet auf die Host-Doorbells, also galt der Wettlauf für jeden KV-Modus.

**Die Messung.** 32K-Nadel, `needle-fastpath-*.json`, dieser Build, **keine** Variable gesetzt (nur
`STRATA_KV_Q4_PROMPT=1` für den noch opt-in q4-Pfad), drei Tiefen je Arm, Prompt-Rate in tok/s:

| Arm | 10 % | 50 % | 90 % | Mittel | Nadeln |
|---|---|---|---|---|---|
| `q4u32` (Matrix-Pfad) | 640,4 | 638,2 | 649,9 | **642,8** | 3/3 |
| `int8` | 644,4 | 651,8 | 650,9 | **649,0** | 3/3 |

**`q4u32` liegt bei 99,1 % von int8.** Der Abstand ist kleiner als die Streuung: dieselbe int8-Konfiguration
las in A.14 (mit dem Schalter, gleicher Build-Vorgänger) 626,6 / 626,7 / 620,8 — **4 % langsamer** als
heute, ohne dass sich an int8 etwas geändert hätte. Zwei Sitzungen, die identisch sein sollten, streuen
also ±4 %; die 1 % zwischen den Armen sind darin nicht auflösbar. (Der 785,1-Ausreißer aus A.14 bleibt
damit auch unerklärt — er ist als Streuung zu lesen, nicht als Vorsprung.)

**4K** mit demselben Fix: 3 von 3 Nadeln, 8 s je Anfrage, ohne Schalter.

**Was der Fix *nicht* ist:** er macht `q4u32` nicht zum Default — der Pfad bleibt hinter
`STRATA_KV_Q4_PROMPT=1`. Er nimmt nur den Hänger weg, und zwar für alle KV-Modi. Die Zahlen oben sind
der Beleg, dass der Default-Pfad darunter nicht leidet (int8 3/3, 649,0 tok/s).

**Offen bleibt:** die Streuung von ±4 % zwischen Sitzungen (zwei Wiederholungen desselben Arms in einer
Sitzung wären billig, um sie zu beziffern), und die Wiederholung der Arme `q4_0` und `q4u32win`, die in
dieser Serie noch fehlen.

### A.16 — Das Gate für B3 steht, und es ist rot (15.11.2026)

**Vorregistriert:** `hip_prompt_attn_wmma.exe` bekommt einen Fensterfall (S 8/R 128, Ring), Referenz ist
der **Decode-Kern** (er liest das Fenster schon) **und** FP64 mit derselben `qsa_win_slot`-Regel. Der Fall
muss **rot** sein, solange der Prompt-Pfad den Seitenpool nicht liest, und darf **nicht** grün werden,
indem er heimlich über den Decode-Kern läuft. Erst danach wird der Pass gebaut.

**Der Harness (`qsa_prompt_attn_parity.cpp`)** hat drei Fenstermodi, alle auf `fmt 3` (`q4u32`):

| Modus | Ring | Was er prüft |
|---|---|---|
| 1 | S + n_q (der ganze Batch passt hinein) | jeder Recent-Cell ist ein Tag-Treffer; Sinks immer |
| 2 | nur S (Tags veraltet) | Tag-Fehltreffer lesen **quantisiert** — beide Quellen in einem Aufruf, wie im Motor mitten im Chunk |
| 3 | nur S, `win_all = 1` | nur der Tag entscheidet (die Horizont-Prüfung fällt weg) |

Die Sinks (Zellen 0..7) werden in jede Selektion gesetzt, damit der sink-Pfad (kein Tag nötig) auch im
sparse Fall läuft. Die Zellzahlen druckt der Harness selbst, damit ein Fensterfall nicht grün werden kann,
ohne eine einzige exakte Zelle zu lesen.

**Der Lauf** (`hip_prompt_attn_wmma.exe 32768 1024 3`, gfx1200, HIP-Build):

```
window1: 139264 of 2100224 cells exact (6.6 %; 8192 sinks, 6.2 % recent), 1960960 read quantized
FAIL q4u32+win ctx 32768, 1024 queries: the prompt path refused the pools (no matrix-core route);
     the decode kernel, the reference, reads this window at 4.72e-06 vs FP64 (output scale 7.42)
window2: 10304 of 503735 cells exact (2.0 %; 2048 sinks, 1.6 % recent), 493431 read quantized
FAIL q4u32+win ctx 2100, 256 queries: ... 3.6e-06 vs FP64 (output scale 7.62)
window3: 10304 of 351360 cells exact (2.9 %; 2048 sinks, 2.3 % recent), 341056 read quantized
FAIL q4u32+win ctx 1500, 256 queries: ... 7.15e-06 vs FP64 (output scale 7.87)
FAILURES: 3
```

**Was daran zählt:**

* **Rot, aus dem richtigen Grund:** `qsa_prompt_attn_batch` lehnt ab (`return false`, 2694–2696), und der
  Harness druckt dafür `FAIL` statt durch den Decode-Kern zu laufen und `PASS` zu sagen (die Falle aus
  B6). Der Abbruch zählt als Fehler (`FAILURES: 3`, Exit 3) — die frühere Fassung hätte hier `exit(2)`
  gemacht und die anderen Fälle verschluckt.
* **Die Referenz ist belegt, nicht angenommen:** der Decode-Kern trifft die FP64-Fensterreferenz auf
  **3,6e-06 … 7,2e-06** bei einer Ausgabenskala von ~7,5 — **~1e-6 relativ**, dieselbe Größenordnung wie
  ohne Fenster. Damit stimmen Zeilenlayout (`[(row, kv_head, head_dim)]`), Ringzeilen, Tags und die
  `qsa_win_slot`-Regel im Harness. Ohne diese Zahl wäre jede spätere Abweichung zwischen Kern und
  Referenz zweideutig.
* **Die Abdeckung ist beziffert:** **6,6 %** exakte Zellen im vollen Fenster (128 recent / 2.051 + 8
  Sinks), **2,0 %** bzw. **2,9 %** in den Tag-Fehltreffer-Fällen. Das ist die Zellrate, die der
  zweite WMMA-Satz zusätzlich rechnen muss — die *Zeit* ist damit noch nicht gemessen.

**Zwei Fehler in meinem eigenen Harness, beide vor dem ersten Befund:** `pl.win_tag` zeigt auf
**Device**-Speicher; die Host-Referenz hat ihn dereferenziert und ist mit einer Zugriffsverletzung
abgestürzt (exit `0xC0000005`). Und `pl.win_b` war auf `n_q` gesetzt, während der Ring auf `ctx`
geklemmt wurde — `qsa_win_slot` rechnet den Ring aber aus `win_s + win_b`, also las es hinter dem
Tag-Array. Beides behoben (`ph.win_tag = wtag.data()`, `win_s + win_b == wring` durch Konstruktion).

### A.17 — Der Fenster-Pass steht, das Gate ist grün (15.11.2026)

**Der Pass, wie er gebaut ist** (`engine/src/kernels/cuda/qsa_prompt_attn.cu`, nur die 4-Bit-Instantiierung
`Q4 = true`):

| Baustein | Wo | Was |
|---|---|---|
| `win_row(p, cell, n, nkv, kvh)` | 1966–1976 | `qsa_win_slot` (die Regel des Decode-Kerns) × `nkv + kvh`; `-1` ohne Fenster |
| `h8_win(base)` | 1959–1965 | `PA_KL` Halves aus dem Seitenpool als B-Fragment (16 B auf gfx12, 32 B auf gfx11) |
| `win_of(ci)` | 2099–2108 | die Fensterzeile der Zelle **dieser Lane** im Chunk `ci`, unabhängig vom Page-Table |
| `Chunk::wrow` / `wr[2]` | 2103–2110 | die Zeile der Zelle bzw. der Zelle der beiden 16er-Kacheln (per `__shfl`) |
| V-Staging `row >= 0 && wrow < 0` | 2132, 2151 | eine Fensterzelle legt **nichts** quantisiert ab (V-Codes 0, `vs` 0) |
| `ksc = 1.0f` bei `wr[nt] >= 0` | 2161–2165 | die exakten Halves sind K selbst, es gibt keinen Blockschritt |
| K-Operand | 2238–2246 | `h8_win(...)` statt `i8_to_h(...)`, sobald die Zelle im Fenster liegt |
| `S.valid = row >= 0 \|\| wrow >= 0` | 2226 | die Maske des Decode-Kerns (`srow < 0 && swin < 0`) |
| zweiter V-Satz `tmpw[4]` | 2308–2356 | `A = p` (ohne `vdown`), `B = V_exakt`, warp-uniform übersprungen ohne Fensterzelle |
| Freigabe | 2659–2662, 2694, 2751 | `q4u_wmma_ok` verlangt keinen leeren Pool mehr; die **int8**-Routen fordern ihn weiter |

**Das Gate** (`hip_prompt_attn_wmma.exe 32768 1024 3`, gfx1200), vorher rot (A.16), jetzt:

```
window1: 139264 of 2100224 cells exact (6.6 %; 8192 sinks, 6.2 % recent), 1960960 read quantized
PASS q4u32+win ctx 32768, 1024: vs FP64 old 4.72e-06 new 5.66e-06 (scale 7.42); new vs old 1.29e-05 (1.6e-06 of scale)
window2: 10304 of 503735 cells exact (2.0 %; 2048 sinks, 1.6 % recent), 493431 read quantized
PASS q4u32+win ctx 2100, 256:    vs FP64 old 3.6e-06 new 8.25e-06 (scale 7.62); new vs old 1.37e-05 (1.7e-06 of scale)
window3: 10304 of 351360 cells exact (2.9 %; 2048 sinks, 2.3 % recent), 341056 read quantized
PASS q4u32+win ctx 1500, 256:    vs FP64 old 7.15e-06 new 7.46e-06 (scale 7.87); new vs old 1.72e-05 (2.2e-06 of scale)
FAILURES: 0
```

* **Die Fensterzellen werden wirklich gelesen.** Der Harness legt in den Seitenpool Werte, die mit den
  Codes nichts zu tun haben. Ein Kern, der das Fenster ignoriert, müsste gegen FP64 um Größenordnungen
  danebenliegen; er liegt bei **5,7e-06 … 8,3e-06** auf einer Skala von ~7,5 — so gut wie der
  Decode-Kern selbst (4,7e-06 … 7,2e-06). Das ist der Beweis, den ein Abbruch-Gate nicht liefern konnte.
* **Kein Rückfall:** die Nicht-Fenster-Fälle sind unverändert grün und **bitgleich** zu vorher
  (`470428b775efed56` int8, `0f694f06948998d5` q4u32, `82ef1dc3fa5dd4e7`/`6c01e0a92922cca5` kurz).

**Ein gemessener Fehler auf dem Weg, und er war nicht klein.** Der zweite V-Satz stand zuerst
**unbewacht** in der int8-Instantiierung: ein `tmpw[4]` = 32 VGPR, das nur `if (anywin)` schreibt, mit
`anywin` als Laufzeitwert. Damit fiel der int8-Prompt-Kern von **8,07 auf 17,4 ms je Chunk** (32K,
1.024 Queries) — Faktor 2,16, und die Ausgabe blieb bitgleich. Mit `if constexpr (Q4)` um die ganze
Fenster-Mechanik (und `Q4 ? tmpw : 0` in der Akkumulation) ist er wieder bei **8,26 ms**. Das ist die
Lehre: ein 32-Register-Akkumulator, der nie benutzt wird, ist nicht gratis — die Occupancy entscheidet
hier über den Faktor 2, nicht die Rechenarbeit.

**Was das Gate *nicht* sagt:**

* **Die Kosten des zweiten Satzes waren in diesem Lauf nicht gemessen.** Im selben Lauf: `q4u32`
  11,33 ms/Chunk, `q4u32+win` 12,34 ms/Chunk (32K, 1.024 Queries) — **aber das sind zwei verschiedene
  Selektionen** (der Fensterfall hat Sinks und einen anderen Horizont), also kein kontrollierter A/B.
  Der kontrollierte A/B steht in A.18: **+6,2 … 13,4 %** der Attention-Kernelzeit.
* **gfx11 ist ungetestet** (`PA_KL = 16`, `PA_KOFF(half) = 0`, `PA_ROW = 2i + half`): `h8_win` spiegelt
  die Adressierung des int8-Pfads, aber auf einer gfx12-Karte ist das statische Analyse.
* **Die Engine ist gelaufen** — A.18: Route-Zeile in allen drei q4-Läufen, 16/16 Nadeln, der Fensterarm
  auf dem Matrix-Pfad. Die Rate dort ist allerdings nicht auflösbar (Zweimoduseffekt).
* **CUDA bleibt zurück** (`qsa_prompt_attn.cu:2793`, `if (pools.k_win != nullptr …) return false;`): die
  Tensor-Core-Route (mode 4) liest kein Fenster, ein Overlay-Arm nimmt dort weiter den Decode-Kern. Auf
  dieser Maschine nicht baubar — als offener Punkt notiert, nicht als erledigt.

### A.18 — vorregistriert: das Fenster in der Engine (15.11.2026, vor dem ersten Lauf)

**Frage:** was kostet das Fenster auf dem Matrix-Pfad in der Engine, und bleibt die Nadelgüte?

**Arme** (je ein Engine-Start, 30 s dazwischen, dieselbe Sitzung, dieser Build):

| Arm | Config | Variablen |
|---|---|---|
| `int8` | `needle-fastpath-int8.json` | `STRATA_HIP_WMMA=1` |
| `q4u32` | `needle-fastpath-q4u32.json` | + `STRATA_KV_Q4_PROMPT=1` |
| `q4u32win` | `needle-fastpath-q4u32win.json` | + `STRATA_KV_Q4_PROMPT=1` (S 256, R 64) |

4K **und** 32K in einem Lauf, Nadeln bei 10/50/90 %.

**Kriterien, vorher festgelegt:**

* **Der Lauf zählt nur mit der Route im Log**: `the matrix-core prompt path serves the 4-bit uniform
  grid` muss für jeden q4-Arm stehen. Steht dort die Rückfall-Zeile („runs the prefill in the decode
  kernel"), ist der Lauf kein Beleg für B3 — er wird wiederholt, nicht umgedeutet.
* **Erwartung 1:** `q4u32win` ≥ 90 % der Prompt-Rate von `q4u32` (Fenster kostet ≤ 10 %).
* **Erwartung 2:** `q4u32win` ≥ `q4u32` in der Nadelgüte — die exakten Zellen sind genauer als die
  quantisierten. Der Unterschied kann unter der Auflösung der Nadel liegen; dann ist er **nicht** belegt.
* **Erwartung 3:** 3/3 Nadeln bei 4K und 32K, kein Hänger (der Launch-Fix aus A.15 ist Default).
* **Streuung:** ±4 % zwischen Sitzungen. Der Fensterarm läuft **zweimal** (zwei Starts); nur ein
  Unterschied > 5 % gilt als Befund. Bei < 90 % wird zuerst gemessen, *welcher* Anteil (Chunks mit
  Fensterzellen), nicht umgebaut.

### A.18 — Das Ergebnis: die Route stimmt, die Prompt-Rate ist auf dieser Maschine nicht auflösbar (15.11.2026)

**Die Route ist belegt.** Alle drei q4-Logs tragen die Zeile

```
strata: prompt attention: the matrix-core prompt path serves the 4-bit uniform grid (q4u32), scale per 32 values
```

und in keinem steht die Rückfall-Zeile. Der Fensterarm schreibt zusätzlich
`KV window ring sized for the prompt chunk (6144 cells)`. **B3 ist damit im Motor angekommen** — vorher lief
dieser Arm über den Decode-Kern.

**Die Nadeln: 16 von 16.** 4K 3/3 und 32K 3/3 in jedem der vier Läufe, kein Hänger. Erwartung 3 ist erfüllt.

**Die Prompt-Raten** (wie in A.15: Prompt-Tokens ÷ Wanduhr der Anfrage; der Decode von 9–10 Token steckt
mit drin, ~1 s), in tok/s:

| Arm | 4K 10/50/90 % | 4K Ø | 32K 10/50/90 % | 32K Ø | Nadeln |
|---|---|---|---|---|---|
| `int8` | 382,4 / 386,9 / 405,9 | **391,7** | 657,8 / 656,3 / 656,3 | **656,8** | 6/6 |
| `q4u32` | 496,6 / 528,1 / 512,0 | **512,2** | 874,5 / 882,0 / 879,5 | **878,7** | 6/6 |
| `q4u32win` (1. Start) | 382,4 / 405,7 / 416,0 | **401,4** | 656,4 / 659,1 / 657,7 | **657,7** | 6/6 |
| `q4u32win` (2. Start) | 378,1 / 405,7 / 416,0 | **399,9** | 663,4 / **872,0 / 869,5** | **801,6** | 6/6 |

**Der Befund ist nicht das Fenster, sondern ein Zweimoduseffekt — und er ist nicht dem Arm zuzuordnen.**
Die Fortschrittszeilen des Motors (`reading the prompt: 6.144 of 30.781 tokens, N s so far`) zeigen
**zwei diskrete Niveaus** für den 6.144-Token-Chunk: **~9 s (langsam)** und **~6,5 s (schnell)**, Faktor
**1,38**. Prompt-Lesezeit je 32K-Anfrage, direkt aus dem Log:

| Arm | 10 % | 50 % | 90 % | Niveau |
|---|---|---|---|---|
| `int8` | 46 s | 46 s | 46 s | langsam |
| `q4u32` | 34 s | 34 s | 34 s | schnell |
| `q4u32win` 1 | 46 s | 46 s | 46 s | langsam |
| `q4u32win` 2 | **46 s** | **35 s** | **35 s** | langsam → **schnell** |

* **Das Niveau ist keine Eigenschaft des Arms.** Der zweite Fensterlauf hat **dieselbe Config** wie der
  erste (gleiche JSON, gleiche Variablen) und liefert beide Niveaus — die erste 32K-Anfrage in der
  langsamen Stufe, die beiden folgenden in der schnellen, **innerhalb eines Prozesses**, ohne Neustart.
  Damit ist „Arm X ist langsam" als Erklärung ausgeschlossen.
* **Erwartung 1 (Fenster ≥ 90 % von `q4u32`) ist nicht auswertbar.** 657,7 gegen 878,7 (75 %) und 801,6
  gegen 878,7 (91 %) sind zwei verschiedene Niveaus, nicht zwei Arme. Das Kriterium wird **nicht**
  umgedeutet; es ist mit dieser Maschine in dieser Form nicht messbar.
* **Was innerhalb eines Niveaus gilt:** im langsamen Niveau liegen `int8` (656,8) und `q4u32win`
  (657,7) **0,1 %** auseinander — nicht unterscheidbar. Im schnellen Niveau liegen `q4u32` (878,7) und
  die beiden schnellen Fensteranfragen (872,0 / 869,5) **0,7–1 %** auseinander — ebenfalls nicht
  unterscheidbar. In beiden Niveaus ist also kein Fenster-Aufpreis sichtbar, aber das ist eine
  **Nicht-Ablehnung**, kein Beleg.
* **Erwartung 2 (Nadelgüte) ist nicht belegt:** 16 von 16 Nadeln in allen Armen. Die Nadel hat für diesen
  Unterschied keine Auflösung; die exakten Zellen sind damit **nicht** als besser belegt.
* **Die Ursache des Niveaus ist nicht belegt.** Kandidaten, alle unbewiesen: WDDM-Seitenwechsel (jeder
  Lauf endet unter seiner eigenen Reserve: „only 1793–1854 MiB free once the slots are written (reserve
  2560 MiB)" steht in **allen vier** Logs), Expert-Paging über PCIe, Takt/Power. Ich habe sie **nicht**
  gemessen und nenne deshalb keinen.

**Was A.15 damit nachträglich bedeutet — und warum es keine neue Zahl ist.** A.15 maß `q4u32` 642,8 gegen
`int8` 649,0 tok/s, also **47,4–48,2 s** bzw. **47,2–47,8 s** je 32K-Anfrage — beide im **langsamen
Niveau** (heute 46,8 s). Die Aussage „99,1 %" war also ein Vergleich *innerhalb* eines Niveaus und bleibt
gültig. Sie sagt aber **nicht**, dass `q4u32` im Prompt gleich schnell ist wie int8 — im schnellen Niveau
steht `q4u32` bei 878,7, und für `int8` gibt es in diesem Niveau **keine** Messung. Der Unterschied (1,34×)
ist damit **offen**, nicht gefunden.

**Die kontrollierte Zahl kommt aus dem Harness, nicht aus dem Motor.** `qsa_prompt_attn_parity.cpp` hat
einen A/B bekommen: derselbe Aufruf, dieselben Selektionen, dieselben quantisierten Pools, nur der
Seitenpool abgehängt (`k_win = v_win = win_tag = nullptr`, `win_* = 0`, also `qsa_win_on` falsch). Zwei
vollständige Läufe (`hip_prompt_attn_wmma.exe 32768 512 3`, gfx1200):

| Fall | exakte Zellen | mit Fenster | ohne | Aufpreis (Lauf 1 / Lauf 2) |
|---|---|---|---|---|
| 1 (S 4096, Ring = S + n_q) | 6,6 % | 6,06 / 6,15 ms | 5,48 / 5,43 ms | **+11,8 % / +13,4 %** |
| 2 (Ring = S, Tags veraltet) | 2,0 % | 2,98 / 3,19 ms | 2,79 / 2,91 ms | **+6,6 % / +9,7 %** |
| 3 (`win_all = 1`) | 2,9 % | 2,40 / 2,38 ms | 2,13 / 2,12 ms | **+13,0 % / +12,3 %** |

Das Fenster kostet also **6,2 % bis 13,4 % der Attention-Kernelzeit** (drei Läufe: der dritte Kontrolllauf
derselben Bauart gab +12,7 % / +6,2 % / +13,4 %, `FAILURES: 0`, Exit 0), und zwar **chunk-granular, nicht
zellproportional**: der zweite V-Satz fällt je Chunk an, nicht je Zelle.

> **Korrektur (17.11.2026, gemessen).** Der Satz, der hier stand — „in jeder Selektion liegen Sinks,
> damit hat *jeder* Chunk Fensterzellen" — ist **falsch**: die 8 Sinks sind die Zellen 0..7 und landen
> in **Chunk 0**, die 128 Recent-Zellen am Ende in den letzten vier Chunks. Bei 32K/512 sind das 5 von
> 65 Chunks und 9 von 130 Kachel-Durchläufen. Die 6,2–13,4 % sind deshalb auch **nicht** der zweite
> V-Satz, sondern das divergente Lesen des Seitenpools in genau diesen Chunks; ein A/B mit halbierter
> Matrixarbeit (`SPLIT` aus) senkt den Aufpreis nicht (17.11., C.0.1).

**Warum der Motor das nie zeigen kann (Hochrechnung, keine Messung am Motor).** Der Harness mißt für
`q4u32` bei 32K **11,2 ms** je Aufruf mit 1.024 Queries. Der Motor liest 6.144 Token je Chunk, also grob
das Sechsfache ≈ **67 ms** — rund **1 %** der 6,5 s, die ein Chunk im schnellen Niveau kostet. 13 % davon
sind **0,1 %** der Prompt-Zeit. Ein Motor-A/B auf das Fenster wäre damit selbst ohne den Zweimoduseffekt
nicht messbar; die belastbare Zahl ist die aus dem Harness.

**Was der Motor-Unterschied *nicht* sein kann: der Attention-Kern.** Der Harness misst im selben Aufbau
(32K, 1.024 Queries, dieselben Selektionen, dieselben Pools) den 4-Bit-Kern gegen den int8-Kern:

| Kern | Laufzeit je Chunk | gegen FP32 |
|---|---:|---:|
| int8 (v2) | **8,26 ms** | 55,40 → 8,26 ms (6,7×) |
| int8, opt-in IMMA | 7,80 ms | 7,94 → 7,80 ms (1,02×) |
| `q4u32` (4-Bit, B3-Apparat an Bord) | **11,33 ms** | 56,74 → 11,33 ms (5,3×) |

Der 4-Bit-Kern ist also **37 % langsamer als der int8-Kern** (11,33 gegen 8,26 ms) — er muss Nibbles
entpacken und je Gruppe eine affine Skala anwenden, wo int8 seine Codes als Bytes in Registern hat. Der
Motor-Vorsprung von `q4u32` (878,7 gegen 656,8 tok/s) kann damit **nicht** aus der Attention kommen; er
muss aus etwas kommen, das die Arme sonst unterscheidet. Der einzige Kandidat, den die Logs beziffern:
die **Größe des KV-Pools** (int8 braucht doppelt so viel) und damit der Expert-Cache — **2967 Slots** im
`int8`-Lauf gegen **3510** im `q4u32`-Lauf (`expert cache auto: … MiB free … -> N slots`). Das ist ein
**Kandidat, keine Messung**: die beiden Fensterläufe mit gleicher Config haben 3411 und 3421 Slots und
trotzdem verschiedene Niveaus, der Cache allein erklärt also auch nicht alles. Bis zu einer eigenen
Messung wird der Motor-Unterschied **nicht** als „q4u32 ist schneller" berichtet.

**Decode, nur zur Vollständigkeit (mit Vorbehalt):** `int8` 38,0 / 36,6 / 33,3 tok/s, `q4u32`
31,2 / 30,6 / 28,2, `q4u32win` 31,4 / 30,7 / 28,2 bzw. 32,3 / 31,2 / 28,3 (32K, 10/50/90 %). Der
Unterschied ist stabil über die Niveaus hinweg — aber **kein sauberer Decode-Vergleich**: die
Spekulationsannahme unterscheidet sich (nur der `q4u32`-Lauf protokolliert `suffix drafts: 1 windows, 2 of
5 drafts accepted`), und die Tokenzahl je Anfrage ist 9–10. Als Hinweis notiert, nicht als Befund.

**Was daraus für A.19 folgt:** ein Motor-A/B „Fenster gegen kein Fenster" ist auf dieser Maschine nicht
auflösbar (0,1 % Effekt gegen 38 % Niveau-Streuung) und wird **nicht** gefahren — dafür ist der
Harness-A/B da. Offen ist die Frage des Zweigs: **warum ist `q4u32` im Prompt schneller als `int8`**, wenn
sein Attention-Kern 37 % langsamer ist?

### A.19 — vorregistriert: ist der Prompt-Vorsprung die Cache-Größe? (15.11.2026, vor dem ersten Lauf)

**Hintergrund.** A.18 lässt zwei Erklärungen für 878,7 gegen 656,8 tok/s übrig, und die Messung kann nur
eine davon ausschließen:

1. **Der Expert-Cache.** `int8` braucht den doppelten KV-Pool, also bleiben weniger Slots für die
   Experten: in A.18 **2967** gegen **3510** Slots (`expert cache N slots` im Log). Weniger Slots heißt
   mehr Experten über PCIe je Prompt-Chunk. Das ist der Kandidat aus A.18.
2. **Etwas am 4-Bit-Pfad selbst** (Writer, Layout, Paging). Der Attention-Kern ist es **nicht** — er ist
   im Harness 37 % **langsamer** (A.18).

**Die Frage:** verschwindet der Prompt-Vorsprung von `q4u32` gegen `int8`, wenn beide **dieselbe Anzahl
Expert-Slots** haben?

**Der Hebel, und was er wirklich tut (zwei Rauchtests à 1k, vor der Serie).** `--expert-cache N` ist
**kein** Slot-Zähler: es ist ein Budget in Einheiten des größten Blobs, und die tatsächliche Slotzahl
steht im Log. **`--expert-cache 2967` ergibt in beiden Armen exakt `expert cache 3950 slots`** (int8 und
q4u32, verifiziert) — damit ist die Slotzahl gleich, und genau das ist der gesuchte Zustand. Der
Nebeneffekt steht ebenfalls schon im Log: mit gleichem Cache meldet **`int8`** `only 587 MiB of VRAM free
at the start of this request (below 1024 MiB)` und die Warnung vor Driver-Paging — der doppelte KV-Pool
lässt sich nicht wegzaubern. **Cache-Größe und freier VRAM sind über die Pool-Größe gekoppelt; beide
zugleich gleichzumachen ist auf dieser Karte unmöglich.** Der Lauf kann also nur zeigen, ob die
Cache-Größe *hinreicht*, den Unterschied zu erklären, nicht ob sie ihn *allein* erklärt.

**Arme** (drei Configs, je zwei Starts, 32K, Nadeln 10/50/90 %, 30 s zwischen den Starts, dieser Build):

| Arm | KV | Cache | freier VRAM (erwartet) | Rolle |
|---|---|---|---|---|
| `int8` | int8 | `--expert-cache 2967` → **3950 Slots** | **~0,6 GiB** (Paging-Warnung) | gleich viel Cache, weniger Platz |
| `q4u32` gleich | q4u32 | `--expert-cache 2967` → **3950 Slots** | ~1,8 GiB | gleich viel Cache, mehr Platz |
| `q4u32` natürlich | q4u32 | auto (A.18: **3510 Slots**) | ~1,8 GiB | wie A.18 (Kontrolle: Satz stabil?) |

Der eigentliche Test ist **`q4u32` gleich gegen `int8`** (gleicher Cache, beide 3950 Slots). Der dritte Arm
prüft, ob der Vorsprung aus A.18 in dieser Sitzung überhaupt wieder auftritt.

**Reihenfolge:** `int8`, `q4u32` gleich, `q4u32` natürlich, `q4u32` natürlich, `q4u32` gleich, `int8`
(symmetrisch, damit eine Zeitdrift beide Enden gleich trifft).

**Kriterien, vorher festgelegt:**

* **Je Anfrage wird das Niveau aus dem Log klassifiziert** (`reading the prompt: … N s so far`, letzte
  Zeile): **schnell < 40 s**, **langsam ≥ 40 s** (A.18: 34–35 s bzw. 46–47 s). Nur Vergleiche
  **innerhalb** eines Niveaus zählen; Anfragen verschiedenen Niveaus werden nicht verglichen.
* **Der Lauf zählt nur mit der Route-Zeile** (wie A.18) und mit der **erwarteten Slotzahl** im Log
  (`expert cache 3950 slots` bzw. `3510 slots`, ±1 %). Weicht die Slotzahl ab, ist der Lauf kein Beleg für
  die gesetzte Zahl; er wird wiederholt, nicht umgedeutet.
* **Der Chunk muss `6144` bleiben** (`prompt chunk auto: 6144 tokens`): ein kleinerer Chunk wäre ein
  zweiter Unterschied und würde den Vergleich tragen statt der Cache.
* **Ergebnis A (Cache-Größe reicht):** in einem Niveau liegt `q4u32` gleich bei `int8` (±5 %).
* **Ergebnis B (Cache-Größe reicht nicht):** in einem Niveau liegt `q4u32` gleich ≥ 15 % über `int8`.
* **Ergebnis C (nicht auflösbar):** in keinem der beiden Niveaus kommen zwei der Arme zusammen vor.
  Dann wird das so berichtet; **kein** weiterer Start derselben Bauart.
* **Kein Decode-Vergleich** in diesem Lauf (die Spekulationsannahme ist nicht gleich; A.18).
* **Was dieser Lauf nicht kann:** die Rolle des freien VRAM von der der Cache-Größe trennen (s. o.). Wird
  Ergebnis B gemessen, ist „der 4-Bit-Pfad selbst ist schneller" **nicht** belegt — es bleibt bei „nicht
  die Cache-Größe" plus offenem Rest.

### A.19 — Das Ergebnis: die Cache-Größe erklärt nichts, und im Motor sind die Arme gleich (15.11.2026)

**Die Vorbedingungen sind alle erfüllt** (die Prüfungen der Vorregistrierung, aus den Logs):

| Lauf | Slots | Route | Chunk | freier VRAM |
|---|---|---|---|---|
| `int8-a` | **3950** | Matrix-Pfad (int8) | 6144 | 590 MiB (**Paging-Warnung**) |
| `int8-b` | **3950** | Matrix-Pfad (int8) | 6144 | 664 MiB (**Paging-Warnung**) |
| `q4u32-eq-a` | **3950** | 4-Bit-Gitter | 6144 | 1415 MiB |
| `q4u32-eq-b` | **3950** | 4-Bit-Gitter | 6144 | 1412 MiB |
| `q4u32-nat-a` | **3510** | 4-Bit-Gitter | 6144 | 2131 MiB |
| `q4u32-nat-b` | **3506** | 4-Bit-Gitter | 6144 | 2139 MiB |

Keine Rückfall-Zeile, überall 6.144-Token-Chunks, die Slotzahl wie gesetzt (3506 statt 3510 = 0,1 %).

**Die Prompt-Raten** (32K, tok/s; **langsam** = 45–47 s Prompt, **schnell** = 35–37 s):

| Arm | 1. Start | 2. Start | Niveau |
|---|---|---|---|
| `int8` @3950 | 659,1 · 667,7 · 672,1 | 656,3 · 663,4 · 652,2 | langsam (6/6) |
| `q4u32` gleich @3950 | 660,6 · 669,2 · 667,7 | 660,6 · **823,0 · 832,0** | 4 langsam, 2 schnell |
| `q4u32` natürlich @3510/3506 | **859,8 · 874,5 · 869,6** | 653,5 · 659,1 · 659,2 | 3 schnell, 3 langsam |

**Ergebnis A (das Kriterium der Vorregistrierung) ist erfüllt — im langsamen Niveau.** Dort liegen alle
**13** Anfragen aller drei Arme zwischen **652,2 und 672,1 tok/s** (Mittel 661,6, Spannweite 3,0 %):
`int8` 661,8 (n=6), `q4u32` gleich 664,5 (n=4), `q4u32` natürlich 657,3 (n=3). **Die Arme sind nicht
unterscheidbar** — bei gleichem Cache (0,2 % auseinander) wie bei natürlichem Cache (0,7 %).

**Die Cache-Größe erklärt den Niveauunterschied also nicht:**

* Im langsamen Niveau macht die Cache-Größe **nichts** (3950 gegen 3506 Slots: 666 gegen 657 tok/s, 1,4 %).
* Im schnellen Niveau ist es **umgekehrt wie erwartet**: der Arm mit dem **größeren** Cache (3950 Slots) ist
  **4,7 % langsamer** (827,5 gegen 868,0 tok/s) — n=2 gegen n=3, ein Prozess je Arm, also eine Beobachtung
  mit dünnem Beleg, aber sie spricht **gegen** die Cache-Erklärung.
* **Der freie VRAM erklärt es auch nicht:** `int8` läuft mit 590 MiB und der Paging-Warnung **genauso schnell
  wie** `q4u32` mit 1415 MiB ohne Warnung (661,8 gegen 664,5).

**Der Fass, der übrig bleibt: das Niveau selbst.** Es trat in **10 von 21** q4-Anfragen auf (A.18 + A.19) und
in **0 von 9** int8-Anfragen — und **zweimal innerhalb eines Prozesses** (A.18 `win-2`: Anfrage 1 langsam,
2 und 3 schnell; A.19 `eq-b`: Anfrage 1 langsam, 2 und 3 schnell). Das Niveau ist also **kein** Arm, aber es
ist auch **nicht zufällig**: die q4-Arme erreichten es zehnmal, int8 nie. **Ich habe die Ursache nicht
gemessen** — der Host-Commit war in allen Läufen unkritisch (75,0–76,8 GiB von 150,4 GiB frei), der
freie VRAM, die Cache-Größe und der Chunk sind ausgeschlossen. Als Kandidaten bleiben: der
PCIe-/Experten-Nachschub vom Host, die WDDM-Seitenwechsel, der Takt. Keiner ist belegt.

**Was daraus für die Branch folgt — die wichtigste Zeile dieses Laufs:**

1. **Im Motor gibt es keinen Prompt-Vorsprung von `q4u32` gegen `int8`.** Der Vorsprung aus A.18
   (878,7 gegen 656,8 tok/s) war ein Vergleich **über zwei Niveaus** und damit nicht zulässig. Innerhalb
   eines Niveaus: **gleichauf**.
2. **Der Kernel ist trotzdem 37 % langsamer** als der int8-Kern (A.18, Harness, gleiche Selektionen) — er
   fällt im Motor nur nicht auf, weil die Attention dort ~1 % des Prompt-Chunks ausmacht (A.18,
   Hochrechnung). Die Optimierung (Teil C) ist damit **kein Prompt-Gewinn**, sondern eine Rettung des
   Anteils, der bei langen Kontexten wächst.
3. **Die Kernaussage des Zweigs verschiebt sich damit:** `q4u32` ist der int8-Ersatz **bei gleicher
   Prompt-Zeit und kleinerem KV-Speicher** — 144 Byte je Zelle und Kopf (`block_q4_0`: 8 Blöcke × 18 Byte)
   gegen 264 Byte (256 int8-Werte + 4 fp16-Skalen), also **45 % weniger** — nicht durch Geschwindigkeit.
   Das ist eine echte, belegte Eigenschaft und eine andere als die, die A.18 nahlegte. **Die
   Prompt-Gleichheit gilt mit `STRATA_KV_Q4_PROMPT=1`** (A.20/A.21); ohne den Schalter ist der Prompt
   **2,5×** langsamer (A.21).

**Offen bleibt:** ein `int8`-Lauf im schnellen Niveau (0 von 9 Anfragen) — ohne ihn bleibt offen, ob `int8`
dort langsamer wäre als `q4u32` (868,0 tok/s). Das ist die einzige Zahl, die die alte Behauptung noch
stützen oder widerlegen könnte; sie ist **nicht** gemessen.

### A.20 — vorregistriert: die verschobenen Vergleiche `q4_0` und `int8` (17.11.2026, vor dem ersten Lauf)

**Warum dieser Lauf.** A.19 hat den Expert-Cache als Erklärung ausgeschlossen und den Satz geschärft: im
langsamen Niveau sind `q4u32` und `int8` nicht unterscheidbar (661,8 gegen 664,5 tok/s), und `q4u32` ist
der int8-Ersatz **wegen des KV-Speichers**, nicht wegen der Prompt-Zeit. Zwei Arme dieser Reihe fehlen
aber noch, und die Default-Entscheidung (KV-Typ) hängt an ihnen:

1. **`q4_0` im Motor** — `q4_0` steht in der Config-Liste, ist in der Motor-Serie A.18/A.19 aber nie
   gelaufen. Es ist **derselbe Speicher** wie `q4u32` (beide `block_q4_0`: 18 Byte je 32 Werte = 144 B je
   Zelle und Kopf), aber ein anderer Codec: `q4_0` schreibt den Block-Maximalwert als Schritt, `q4u32`
   den Least-Squares-Fit nach der Hadamard-Rotation.
2. **`int8` im schnellen Niveau** — A.19 hat es in 0 von 9 Anfragen erreicht; die Frage „ist `int8` dort
   langsamer als `q4u32`?" ist offen.

**Statischer Befund vor dem Lauf (Code gelesen, nicht gemessen):** Auf HIP nimmt **nur das uniforme
Gitter** den Matrix-Core-Prompt-Pfad (`q4u_wmma_ok()` verlangt `q4_uniform_ls(pools.q4_mode)`,
`qsa_prompt_attn.cu:2711`). Für `q4_0` fällt die Prompt-Attention auf den Decode-Kernel zurück, und der
Code sagt das selbst: „the q4 arm runs the prefill in the decode kernel (no prompt fast path on this
backend)" (`:2819-2820`). **Die Erwartung daraus:** `q4_0` ist im Prompt deutlich langsamer als `q4u32`
und `int8` — nicht wegen des Codecs, sondern weil ihm der Kern fehlt. Liegt `q4_0` trotzdem gleichauf, ist
diese Aussage falsch, und die Route-Zeile des Laufs muss zeigen, warum.

**Arme** (drei Configs, je **drei** Starts à drei Nadeln 10/50/90 %, 32K, 30 s zwischen den Starts,
dieser Build):

| Arm | KV | Config | Cache | erwartete Slots | Route (erwartet) |
|---|---|---|---|---|---|
| `int8` | int8 | `needle-fastpath-int8-c2967.json` | `--expert-cache 2967` | 3950 | int8-Matrix-Pfad (keine Route-Zeile, wie A.19) |
| `q4_0` | q4_0 | `needle-fastpath-q4_0-c2967.json` (neu) | `--expert-cache 2967` | 3950 | **Decode-Kernel** (statisch) |
| `q4u32` | q4u32 | `needle-fastpath-q4u32-c2967.json` | `--expert-cache 2967` | 3950 | 4-Bit-Gitter, Matrix-Core |

Alle drei mit demselben Cache-Budget, damit die Slotzahl (3950) gleich ist. Die 4-Bit-Arme laufen mit
`STRATA_KV_Q4_PROMPT=1` (ohne das läuft auch `q4u32` im Decode-Kernel), `q4_0` mit demselben Envs.

**Reihenfolge (verschränkt, kein Arm immer zuerst):** Runde 1 `int8`, `q4_0`, `q4u32`; Runde 2
`q4u32`, `q4_0`, `int8`; Runde 3 `int8`, `q4_0`, `q4u32`. Damit hat jeder Arm je einen Start am Anfang, in
der Mitte und am Ende.

**Kriterien, vorher festgelegt:**

* Je Anfrage das Niveau aus dem Log (`reading the prompt: … N s so far`, letzte Zeile): **schnell < 40 s**,
  **langsam ≥ 40 s**. Verglichen wird nur **innerhalb** eines Niveaus; Anfragen verschiedenen Niveaus
  werden nicht verglichen.
* Ein Lauf zählt nur mit **3950 Slots ±1 %**, **Chunk 6144** und — für `q4u32` — der Route-Zeile „the
  matrix-core prompt path serves the 4-bit uniform grid". Für `q4_0` wird die Zeile „… decode kernel (no
  prompt fast path on this backend)" **erwartet**; fehlt sie und ist `q4_0` schnell, ist die Messung
  ungültig — dann lief ein anderer Pfad als angenommen.
* **Ergebnis A (erwartet):** `q4_0` liegt im Prompt ≥ 15 % über `q4u32`/`int8`, und `q4u32` liegt bei
  `int8` (±5 %). → `q4u32` bleibt der Kandidat, `q4_0` scheidet als Default aus, solange es keinen
  Prompt-Kern hat.
* **Ergebnis B (überraschend):** `q4_0` liegt innerhalb 5 % von `q4u32`. Dann trägt der Prompt-Kern bei 32K
  nichts, und die Entscheidung fällt über Qualität und Speicher; die Route-Zeile muss dann erklären, warum.
* **Ergebnis C (nicht auflösbar):** in keinem Niveau kommen zwei Arme zusammen vor, oder `q4_0` läuft in
  ein anderes Niveau als die anderen. Dann wird das berichtet; **kein** weiterer Start derselben Bauart.
* **`int8` im schnellen Niveau:** tritt es wieder nicht auf, bleibt die Frage offen und wird als offen
  berichtet — **nicht** durch zusätzliche Starts erzwungen, und keine Aussage über `int8` im schnellen
  Niveau.
* **Kein Decode-Vergleich** (die Spekulationsannahme ist nicht gleich; A.18).
* **Was dieser Lauf nicht kann:** den Codec von der Route trennen — `q4_0` hat den Prompt-Kern nicht, ein
  Unterschied ist also nicht dem 4-Bit-Codec zuzuschreiben. Er beantwortet nur „was kostet der fehlende
  Kern im Motor?".

### A.20 — Das Ergebnis: `q4_0` kostet 5–7 %, `int8` und `q4u32` sind gleichauf (17.11.2026)

**Die Vorbedingungen sind erfüllt, und die Route-Zeile bestätigt den statischen Befund 3 von 3:**

| Arm | Läufe | Slots | Chunk | Route-Zeile im Log | freier VRAM |
|---|---|---|---|---|---|
| `int8` | 3 | 3950 | 6144 | keine (wie A.19) | 602 / 775 / 759 MiB (**Paging-Warnung**) |
| `q4_0` | 3 | 3950 | 6144 | **„decode kernel (no prompt fast path on this backend)"** 3/3 | keine Warnung |
| `q4u32` | 3 | 3950 | 6144 | „matrix-core prompt path serves the 4-bit uniform grid" 3/3 | keine Warnung |

Alle neun Läufe: 3 von 3 Nadeln gefunden, 6144-Token-Chunks, 3950 Slots. **Alle 27 Anfragen liegen im
langsamen Niveau** (40–49 s Prompt) — das schnelle Niveau trat in dieser Sitzung **überhaupt nicht** auf.

**Die Prompt-Raten** (32K, tok/s, neun Anfragen je Arm):

| Arm | Mittel | Median | Spannweite | je Runde (1 / 2 / 3) |
|---|---:|---:|---|---|
| `int8` | **680,2** | 679,5 | 672,1–685,6 (2,0 %) | 676,0 / 682,0 / 682,6 |
| `q4u32` (Matrix-Kern) | **676,7** | 666,3 | 656,3–**749,0** | 703,9 / 662,5 / 663,9 |
| `q4_0` (Decode-Kern) | **631,3** | 636,0 | 614,4–638,7 (3,9 %) | 629,7 / 628,3 / 636,0 |

**Die drei Befunde:**

1. **`int8` und `q4u32` sind nicht unterscheidbar.** Mittel 680,2 gegen 676,7 (**+0,5 %** für `int8`),
   Median 679,5 gegen 666,3, je Runde +4,1 / −2,9 / −2,8 %. Das ist A.19 bestätigt, mit 9 statt 4
   Anfragen je Arm: **kein Prompt-Vorsprung des einen oder anderen**. Wieder auffällig: `int8` läuft mit
   der **Paging-Warnung** (602–775 MiB frei) und ist trotzdem nicht langsamer.
2. **`q4_0` ist 5–7 % langsamer als `q4u32`** — Mittel **−6,7 %**, Median −4,6 %, ohne den einen
   749,0-Ausreißer **−5,4 %**; gegen `int8` −7,2 %. Das liegt **an der 5-%-Schwelle**, nicht darüber, und
   **weit unter** dem in der Vorregistrierung erwarteten Abstand von ≥ 15 % (Ergebnis A). Der statische
   Grund ist trotzdem bestätigt: `q4_0` fährt den Prompt im **Decode-Kern**, die Route-Zeile sagt es in
   allen drei Läufen. Sein Preis ist also der fehlende Prompt-Kern — und der ist bei 32K **klein**:
   ~0,6–0,9 s je 6144-Token-Chunk.
3. **Das schnelle Niveau kam nicht vor** (0 von 27 Anfragen; A.18/A.19: 10 von 21 q4-Anfragen). Die
   einzige Spur ist **eine** `q4u32`-Anfrage mit 40,0 s und **749,0 tok/s** — die einzige Anfrage an der
   Niveaugrenze. Eine Anfrage ist kein Niveau; sie ist als Streuung zu lesen und fließt in keine Aussage
   ein. Damit bleibt **`int8` im schnellen Niveau ungemessen** (über A.19 und A.20 zusammen: 0 von 12).

**Was das für die Default-Entscheidung heißt (nur die gemessenen Sätze):**

* **`q4_0` bringt gegenüber `q4u32` nichts:** gleicher Speicher (144 B je Zelle und Kopf, beide
  `block_q4_0`), **5–7 % langsamerer Prompt**, und auf HIP **kein** Matrix-Core-Prompt-Kern. Als Default
  hat es damit keinen Vorteil, den es nicht schon über `q4u32` gäbe. (Ein Qualitätsvergleich der beiden
  Codecs — Block-Maximum gegen Least-Squares-Fit — ist **nicht** Teil dieser Serie und wird hier nicht
  behauptet.)
* **`int8` bringt gegenüber `q4u32` nichts an Prompt-Zeit** (±3 %, im Rauschen) und kostet **83 % mehr KV**
  (264 gegen 144 Byte je Zelle und Kopf). Der einzige offene Punkt bleibt das schnelle Niveau.
* **Der Prompt-Kern ist für `q4_0` ~5–7 % wert** — nicht 2,3×, wie A.1 nahelegte. ~~Das ist die Zahl, an
  der sich entscheidet, ob `STRATA_KV_Q4_PROMPT=1` Default werden soll.~~ **Korrigiert durch A.21:** die
  5–7 % gelten für `q4_0`, das im Decode-Kern schon die schnelle Instantiation nutzt. Für `q4u32` ist
  derselbe Kern **2,5×** wert (265 gegen 655 tok/s), weil sein Decode-Kern die fensterfähige
  Instantiation ist. Die Default-Frage hängt damit an `q4u32`, nicht an `q4_0`.
* **Nicht gemessen:** `q4u32` **ohne** den Schalter, also im Decode-Kern wie `q4_0`. Der `q4_0`-Arm ist
  dafür nur ein Stellvertreter (gleicher Kern, anderer Codec); die 631,3 tok/s sind **kein** Messwert für
  `q4u32` ohne Schalter. ~~Das wird in A.21 nachgeholt.~~ **A.21 hat es nachgeholt, und der Stellvertreter
  war falsch — siehe die Korrektur unten.**

### A.21 — Das Ergebnis: ohne den Schalter ist `q4u32` 2,5× langsamer (17.11.2026)

**Die Vorbedingungen sind erfüllt:** alle sechs Läufe 3950 Slots, Chunk 6144; die **Opt-in-Zeile**
(„the matrix-core prompt path for the 4-bit grid is opt-in") steht **3 von 3** `q4u32`-Läufen, die
Matrix-Kern-Zeile **nie**. Der Arm lief also wirklich im Decode-Kern, wie vorregistriert. 3 von 3 Nadeln
gefunden.

| Arm | Anfragen | Mittel | Median | Spannweite | Niveau |
|---|---:|---:|---:|---|---|
| `int8` | 12 | **654,6** (n=8 langsam) | 652,2 | 641,3–676,5 | 11 langsam, **1 schnell** (785,3) |
| `q4u32` ohne Schalter | 9 | **265,0** | 264,5 | 263,1–272,6 | **9 von 9 langsam** |

**Im langsamen Niveau: −59,5 %** (Mittel), −59,4 % (Median) — **2,47× langsamer** als `int8`.

**Die Vorregistrierung ist damit widerlegt, und zwar klar.** Erwartet war (Hypothese, aus dem `q4_0`-Arm
von A.20) ein Niveau um 631 tok/s, also 5–7 % unter `int8`. Gemessen: **265 tok/s**. Der `q4_0`-Arm ist
**kein** Stellvertreter für `q4u32` im Decode-Kern — er sieht nur so aus, weil beide dieselbe Zeile
„no prompt fast path on this backend" drucken. Der Unterschied liegt in der **Instantiation**, nicht im
Codec (statische Analyse, in A.1 schon so beschrieben): `q4_0` läuft als `kv_mode 2` (`kUniform`, ohne
Overlay-Zweig), `q4u32` als `kv_mode 6` (`kUniformLs`, fensterfähige Instantiation **mit** Overlay-Zweig).

**Und dieser Preis ist reproduzierbar:** 263,1–272,6 tok/s hier gegen **263,7 tok/s** in A.1 (14.11.,
anderer Tag, anderer Cache: 3015 gegen 3950 Slots). Zwei unabhängige Messungen, dieselbe Zahl.

**Der Aufbau der 116 s** (aus den Konsolenzeilen, Ursache **nicht** gemessen): je Anfrage ein
**Vorlauf über den ganzen Prompt** (~20 s; die Zeilen „30.782 tokens, 10 s / 20 s") und danach
**~13 s je 6144-Token-Chunk** — gegen 9 s bei `int8`. Zum Vergleich die Zeile aus dem `q4_0`-Arm in A.20:
9–10 s je Chunk, ohne Vorlauf. Der Vorlauf ist der größte Einzelposten und in dieser Serie nicht zerlegt.

**Was das ändert:**

1. **Die Kernaussage des Zweigs gilt nur mit dem Schalter.** `q4u32` ist der int8-Ersatz **bei gleicher
   Prompt-Zeit und 45 % kleinerem KV** — mit `STRATA_KV_Q4_PROMPT=1`. **Ohne** ihn ist der Prompt
   **2,5× langsamer** als `int8`; damit wäre `q4u32` kein Ersatz, sondern ein Rückschritt.
2. **Die 5–7 % aus A.20 gehören zum Stock-Codec, nicht zum uniformen Gitter.** Für `q4_0` (das den schnellen
   Decode-Kern hat) ist der Matrix-Kern 5–7 % wert; für `q4u32` (das den langsamen hat) ist er **2,5×**
   wert. Beides ist gemessen, beides gilt für seinen eigenen Vergleich.
3. **Der Schalter ist damit die eigentliche Default-Frage**, nicht der KV-Typ: mit ihm liegt `q4u32` bei
   `int8` (±3 %) und spart 83 % KV-Bytes; ohne ihn ist es 2,5× langsamer. Die Entscheidung ist eine
   Risiko-Entscheidung (der Hänger aus A.15, dort behoben, aber weiter opt-in) — keine
   Geschwindigkeitsfrage mehr.
4. **Nebenbeobachtung:** `int8` hat **einmal** das schnelle Niveau erreicht (38 s, **785,3 tok/s**,
   Anfrage 1 von `a21-int8-c`) — in A.18/A.19/A.20 in 0 von 21 Anfragen. Das Niveau ist also **nicht**
   arm-spezifisch; eine Anfrage ist aber keine Aussage über das Niveau, und `q4u32`/`q4_0` kamen in
   diesem Lauf nicht in seine Nähe.

### A.22 — vorregistriert: der gekoppelte Schalter (17.11.2026, vor dem ersten Lauf)

**Die Änderung (Code, statisch):** `q4_prompt_on()` liest `STRATA_KV_Q4_PROMPT` jetzt als **Aus**-Schalter:
unset oder `1` = an, `0` = aus. Die Funktion wird nur auf dem HIP-Zweig erreicht, und dort nur, wenn
`q4u_wmma_ok()` gilt — also **nur für das uniforme Gitter (`--kv q4u32`)**. int8, fp16, `q4_0` und alle
Lloyd-Modi laufen denselben Code wie vorher; ihr Pfad ist byte-identisch. Der CUDA-Zweig (mode 4) ist
unberührt, dort war der Schalter nie im Spiel.

**Das Gate (gelaufen, 17.11.):** `hip_prompt_attn_wmma.exe 32768 512 3` mit dem neuen Binary —
**`FAILURES: 0`**, und **alle neun `NEWHASH` identisch** zum Lauf vom 17.11. vor der Änderung
(`232d4781a2b59ccc`, `2455dbed6eba900f`, `9f901d5231c4013e`, `d791e4bfb2fd064d`, `82ef1dc3fa5dd4e7`,
`6c01e0a92922cca5`, `ad466135a4d13e70`, `06c917251ed1d5cf`, `1d3b59ff22ef3ea5`). Die Änderung fasst keine
Arithmetik an; das Gate prüft, dass das so bleibt.

**Die Frage.** Kommt `q4u32` **ohne jedes Env** im Motor auf das Niveau von `int8` — also auf dem Weg, den
ein Nutzer ohne Flags geht?

**Die Arme** (3 Starts je Arm, 3 Nadeln, 32K, 30 s, dieser Build):

| Arm | KV | Env | erwartete Route |
|---|---|---|---|
| `int8` | int8 | — | int8-Matrix-Pfad (keine Zeile, wie immer) |
| `q4u32 Default` | q4u32 | **kein Env** | **Matrix-Kern**, Zeile „serves the 4-bit uniform grid" |
| `q4u32 Aus` | q4u32 | `STRATA_KV_Q4_PROMPT=0` | Decode-Kern (Zeile „STRATA_KV_Q4_PROMPT=0: … decode kernel") |

**Reihenfolge:** `int8`, `q4u32 Default`, `q4u32 Default`, `int8`, `int8`, `q4u32 Default`, danach **ein**
Lauf `q4u32 Aus` als **Routing-Prüfung** (kein Vergleich — die 2,5× sind in A.21 mit neun Anfragen
gemessen).

**Kriterien, vorher festgelegt:**

* `q4u32 Default` zählt nur, wenn die Matrix-Kern-Zeile **3 von 3** im Log steht; erscheint die
  `=0`-Zeile, ist die Kopplung nicht wirksam und der Lauf ungültig.
* `q4u32 Aus` muss die `=0`-Zeile zeigen und ~265 tok/s liefern (A.21: 265,0) — sonst ist der Aus-Schalter
  nicht der alte Pfad.
* 3950 Slots ±1 %, Chunk 6144, Niveau je Anfrage (≥ 40 s langsam), Vergleich nur innerhalb eines Niveaus.
* **Erwartung:** `q4u32 Default` liegt innerhalb 5 % von `int8` (A.20: 676,7 gegen 680,2 tok/s).
* **Ergebnis A:** innerhalb 5 % → die Kopplung liefert den gemessenen A.20-Zustand ohne Flags.
* **Ergebnis B:** ≥ 5 % unter `int8` → die Kopplung ist nicht der gemessene Pfad; dann wird die
  Route-Zeile geprüft und **nicht** umgedeutet.
* **Ergebnis C:** kein Niveau mit mindestens zwei Armen → berichten, kein weiterer Start.

### A.22 — Das Ergebnis: die Kopplung trägt, ohne Flag (17.11.2026)

**Die Vorbedingungen sind erfüllt:** alle sieben Läufe **3950 Slots**, **Chunk 6144**. Die Matrix-Kern-Zeile
(„serves the 4-bit uniform grid") steht in **3 von 3** `q4u32 Default`-Läufen, die `=0`-Zeile in **keinem**;
der Aus-Lauf zeigt sie (**1 von 1**, einmal je Start, wie der Code sie druckt). Die Zeile „no prompt fast path"
kommt **nie**. **21 von 21 Nadeln** gefunden (7 Läufe × 3).

| Arm | Anfragen | Mittel | Median | Spannweite | Niveau |
|---|---:|---:|---:|---|---|
| `int8` (a+b) | 6 | **663,6** | 664,9 | 636,0–682,5 | 6 langsam |
| `int8` (c) | 3 | 860,9 | 862,2 | 841,0–879,5 | **3 schnell** |
| `q4u32 Default` (a+b+c) | 9 | **651,6** | 650,8 | 641,3–664,8 | 9 langsam |
| `q4u32 Aus` | 3 | 264,3 | 264,2 | 262,0–266,8 | 3 langsam |

**Im langsamen Niveau: −1,8 %** (Mittel), **−2,1 %** (Median) gegen `int8` — **Ergebnis A**: innerhalb der
vorab fixierten 5 %, also liegt der gekoppelte Schalter auf dem gemessenen A.20-Zustand (dort 676,7 gegen
680,2 tok/s, ebenfalls innerhalb 5 %). **Der Aus-Lauf bestätigt den Aus-Schalter:** 264,3 tok/s gegen
**265,0** in A.21 und **263,7** in A.1 — drei Messungen, dieselbe Zahl, und in dieser Serie −60,2 % gegen `int8`.

**Nebenbeobachtung zum Niveau:** `int8-c` lief **alle drei** Anfragen im schnellen Niveau (35,0 / 35,7 / 36,6 s,
841,0–879,5 tok/s) — in A.21 war es **eine** von zwölf `int8`-Anfragen, hier **drei von sechs**. `q4u32` kam in
dieser Serie **9 von 9** Anfragen nicht in seine Nähe (46,3–48,0 s). Der Vergleich oben wird deshalb **innerhalb
des Niveaus** geführt (9 gegen 6 langsame Anfragen); das Niveau selbst bleibt unerklärt und ist **nicht**
arm-spezifisch, wie A.21 schon zeigte.

**Damit ist `q4u32` ab jetzt „`q4u32` mit Prompt-Kern".** Die Kopplung braucht **kein** Env-Flag; `=0` ist nur
noch das A/B-Werkzeug, um den alten Decode-Kern-Pfad zu zeigen.

### A.23 — vorregistriert: der KV-Default im Motor (17.11.2026, vor dem ersten Lauf)

**Die Änderung (Code, statisch, zu diesem Zeitpunkt noch nicht gebaut):** Ohne `--kv` entscheidet die Karte.

* **gfx12** (gfx1200/gfx1201) **mit** dem Prompt-Default → **`q4u32`**, Log-Zeile
  `KV storage: q4u32 (the default on this part: …)`.
* **sonst** → **`fp16`**, Log-Zeile `KV storage: fp16 (the default here: …)`.

Dafür schaltet der **Windows**-HIP-Build die beiden freigegebenen gfx12-Prompt-Defaults ein
(`kGfx12PromptDefaultsOn`; vorher war Windows dort aus, A.17–A.22 sind der Test, der das ändert).
`STRATA_HIP_WMMA=0` hält den KV-Typ bei `fp16`, weil das der „kein Prompt-Pfad"-Zustand ist. **CUDA bleibt
`fp16`** (mode 4 existiert dort, wurde auf diesem Zweig aber weder gebaut noch gemessen).

**Die Frage.** Kommt ein Start **ohne `--kv` und ohne Env** auf den gemessenen Zustand — Matrix-Kern, `q4u32`
— und kommt der freigewordene KV-Platz mit `--expert-cache auto` **im Experten-Cache** an?

**Die Arme** (3 Nadeln, 32K, 30 s Abstand, dieser Build; Env nur `STRATA_PREFILL_CPU_SHARE=0`, **kein**
`STRATA_HIP_WMMA`):

| Arm | Flags | erwartete Route | Läufe |
|---|---|---|---|
| `default` | **kein `--kv`**, `--expert-cache auto` | Zeile `KV storage: q4u32` **und** Matrix-Kern-Zeile | 3 |
| `int8` | `--kv int8`, `--expert-cache auto` | int8-Matrix-Pfad (keine 4-Bit-Zeile) | 2 |
| `fp16` | `--kv fp16`, `--expert-cache auto` | Routen-Prüfung: **keine** 4-Bit-Gitter-Zeile | 1 |

**Reihenfolge:** `default`, `int8`, `default`, `int8`, `default`, `fp16`.

**Kriterien, vorher festgelegt:**

* `default` zählt nur, wenn **3 von 3** Läufen die Zeile `KV storage: q4u32` **und** die Matrix-Kern-Zeile
  zeigen. Erscheint `KV storage: fp16`, ist die Karte nicht als gfx12 erkannt → Lauf ungültig, Ursache prüfen.
* **Slots:** `default` ≥ `int8` + 1 % — der KV-Platz muss im Experten-Cache ankommen. Sind die Slots gleich,
  wird das **berichtet** (dann begrenzt etwas anderes den Cache), nicht umgedeutet.
* **Geschwindigkeit:** `default` nicht mehr als 5 % unter `int8` im selben Niveau. (Kein „schneller erwartet":
  mehr Slots ändern die Trefferquote, der Vergleich ist bewusst „nicht schlechter".)
* **Chunk:** wird berichtet, wie er kommt. Ändert er sich gegenüber 6144, ist das **kein** Fehler, sondern
  eine Folge des größeren freien VRAM — die tok/s-Zahl gilt dann für diesen Chunk.
* 3 von 3 Nadeln je Lauf.
* `fp16`-Lauf: keine 4-Bit-Gitter-Zeile; dient nur der Route, kein Vergleich.
* **Ergebnis A:** alle Punkte erfüllt → der Default bleibt.
* **Ergebnis B:** ein Punkt verletzt → berichten, Default überprüfen (nicht „weil erwartet" stehen lassen).

### A.23 — Das Ergebnis: der Motor wählt `q4u32`, und der KV-Platz kommt im Experten-Cache an (17.11.2026)

**Alle Vorbedingungen erfüllt.** In **3 von 3** `default`-Läufen steht `KV storage: q4u32 (the default on this
part: 54.5 % of int8's bytes, level with it at 32K; …)` (das ist die neue, motoreigene Zeile — kein `--kv` im
Aufruf), dazu die Matrix-Kern-Zeile. Die `int8`- und der `fp16`-Lauf zeigen **keine** 4-Bit-Gitter-Zeile, „no
prompt fast path" kommt **nie**. **18 von 18 Nadeln** gefunden (6 Läufe × 3).

| Arm | Slots (3 bzw. 2 Starts) | Mittel | Chunk | Anfragen | Mittel tok/s |
|---|---|---:|---:|---:|---:|
| `default` (kein `--kv`) | 3438 / 3414 / 3434 | **3428,7** | 6144 | 9 | **664,5** |
| `int8` | 2917 / 2860 | 2888,5 | 6144 | 6 | 669,0 |
| `fp16` (nur Route) | 1970 | — | **5888** | 3 | 559,4 |

**Slots: +18,7 %** (3428,7 gegen 2888,5) — der freigewordene KV-Platz kommt im Experten-Cache **an**, das
Kriterium war ≥ +1 %. **Geschwindigkeit: −0,7 %** im Mittel, **−0,1 %** im Median (664,5 gegen 669,0 tok/s) —
innerhalb der vorab fixierten 5 %. **Chunk** bleibt bei `default` und `int8` **6144**; der `fp16`-Lauf bekommt
**5888**, weil sein KV mehr Platz braucht — berichtet, kein Fehler, und dieser Lauf dient nur der Route.
`fp16` läuft mit 1970 Slots und 559,4 tok/s deutlich langsamer, was hier **nicht** als Vergleich zählt (anderer
KV-Typ **und** anderer Chunk).

Alle Punkte der Vorregistrierung erfüllt → **Ergebnis A**: **ohne `--kv` entscheidet die Karte, und auf gfx12
ist das `q4u32` mit dem Prompt-Kern.** `--kv int8` / `--kv fp16` bleiben der Weg, einen anderen Typ zu erzwingen.

**Einordnung.** A.22 maß `q4u32` gegen `int8` mit **festem** `--expert-cache 2967` (beide 3950 Slots, −1,8 %).
A.23 misst das, was ein Start **ohne Flags** tut (`--expert-cache auto`): der Motor legt 3438 statt 2917 Slots
an, und das kostet nichts an Prompt-Geschwindigkeit (innerhalb des Harness-Rauschens von 2–3 %).

**Die Versionsfrage dazu.** Der Motor dieses Zweigs ist **`0.1.62-q4u32`**: eine eigene Version über der
Upstream-Engine (0.1.63, v0.2.15), mit der Variante als Suffix, damit ein Zweig-Build nie für einen
Upstream-Build gehalten wird und der Updater die beiden unterscheiden kann. Das Terminal-Gate hängt an genau
dieser Nummer (`terminal/strata_control/kv_default.py`, `KV_DEFAULT_ENGINE = (0, 1, 62)`): nur eine Engine ab 0.1.62
**und** eine gfx12-Karte lassen den Format-Wahl dem Motor; überall sonst schreibt das Terminal das Format wie
vorher (`int8`), damit diese Starts byte-identisch zum Release bleiben.

### A.24 — vorregistriert: die gfx11-Familie (17.11.2026, vor dem ersten Lauf)

**Die Frage.** Der Matrix-Core-Prompt-Kern hat im Baum **schon eine gfx11-Variante**
(`engine/src/kernels/cuda/qsa_prompt_attn.cu:1897–1916` und `:1941–1950`: eigener Builtin, `PA_KL = 16`,
`PA_KOFF = 0`, `PA_ROW = 2·i + half`). Mit `STRATA_HIP_WMMA=1` läuft dort auch der **4-Bit-Pool**
(`hip_wmma_usable:2503–2513`, `q4_prompt_on:2723–2729`, `:2812–2826`); der integrierte gfx1103 (780M)
nimmt das per Default (`engine/src/core/arch_defaults.cpp:163`, **gemessen**: Prompt 1K 98,5 → 157,3 tok/s,
4K 115,1 → 233,3, `docs/RADEON_780M.md:82–105`). Was fehlt, ist nur die **Voreinstellung** des
KV-Formats: `kv_default_mode()` (`engine/src/program/generate.cpp:2547`) wählt auf gfx11 `fp16`.

**Der Plan (auf einer gfx1100/gfx1101/gfx1102-Karte):**

| Schritt | Was | Erwartung |
|---|---|---|
| 1 | `hip_prompt_attn_wmma.exe 32768 512 3` auf gfx11 | `FAILURES: 0`; der Harness akzeptiert gfx11 und fährt den 4-Bit-Fall (`engine/src/kernels/qsa_prompt_attn_parity.cpp:466–469`, `:487`) |
| 2 | Nadel-Lauf 32K, 3 Tiefen, drei Arme: `--kv q4u32` **mit** `STRATA_HIP_WMMA=1`, `--kv q4u32` **ohne** (also `STRATA_HIP_WMMA=0`), `--kv int8` mit Kern | der 4-Bit-Arm mit Kern nicht mehr als 5 % unter int8; der 4-Bit-Arm ohne Kern deutlich langsamer (auf gfx12 waren das 264 gegen 663 tok/s) |
| 3 | **Logit-Gate**: KL der Prefill-Logits des WMMA-Prefills gegen den FP32-Kern, gleiches Top-p | KL ≤ 0,005 und Top-p ≥ 98 % — die Schwelle aus dem gfx11-Fork (stew675 PR #199), dessen WMMA-Prefill die Logits messbar verschiebt (KLD 0,0293, Top-p 92,59 %) |

**Kriterien, vorher festgelegt:** alle drei Schritte grün → der Motor darf auf gfx12 **und** gfx11
selbst entscheiden (`q4u32`), das Terminal folgt (`terminal/strata_control/kv_default.py`, `setup.py:11177`).
Ein verletzter Punkt → berichten, die Voreinstellung bleibt auf gfx12 beschränkt.

**Stand in dieser Umgebung: offen.** Hier steht **keine gfx11-Karte** (der Testrechner ist eine
gfx1201). Schritt 1 lässt sich dort nicht fahren; die Schritte 2 und 3 brauchen die Karte erst recht.
Bis dahin bleibt alles gfx11-Spezifische **statische Analyse** (siehe
`docs/research/KV_Q4_MATRIX_PATH_RDNA4.md` §9) und die Voreinstellung gilt nur auf gfx12.

### A.25 — vorregistriert: der APU-Fall (Strix Halo gfx1151, 780M gfx1103) (17.11.2026, vor dem ersten Lauf)

**Die Frage.** Auf den APU-Karten läuft der Matrix-Core-Prompt-Pfad **schon per Default**
(`engine/src/core/arch_defaults.cpp:129–137`, `:172–191`; **gemessen** auf dem Strix-Box: 4K 690 → 1.090 und
8K 785 → 1.183 tok/s, ×1,58 / ×1,51, erster Token KL 0,0088; 780M: 109,5 → 214,7 tok/s). Offen ist dort
nur der **4-Bit-KV-Pool**. Das ist eine andere Frage als auf einer dGPU mit eigenem VRAM: auf dem APU
ist der Speicher **geteilt** (UMA), der KV **streamt ab 64K** (`setup_config.py:329–330`), und der APU
hat eigene gemessene Defaults (PF_FUSED, PF_F16, PF_DM16, SELECT_WMMA, Prompt-Chunk 4096).

**Die Arme** (Strix-Halo-Box, Windows, gfx1151; 3 Nadeln, 32K, 30 s Abstand, dieser Build):

| Arm | Flags | erwartete Route |
|---|---|---|
| `q4u32` | `--kv q4u32` (Matrix-Kern per Default an) | Matrix-Kern-Zeile, 4-Bit-Pool |
| `q4u32 ohne Kern` | `--kv q4u32`, `STRATA_HIP_WMMA=0` | Decode-Kern (der langsame Pfad) |
| `int8` | `--kv int8` | der Vergleichsarm |

**Kriterien, vorher festgelegt:**

1. **Parität:** `hip_prompt_attn_wmma.exe 32768 512 3` auf gfx1151 → `FAILURES: 0` (der Harness
   akzeptiert gfx11 und fährt den 4-Bit-Fall, `engine/src/kernels/qsa_prompt_attn_parity.cpp:466–469`, `:487`).
2. **Tempo:** der 4-Bit-Arm **mit** Kern nicht mehr als 5 % unter `int8` im selben Niveau; der Arm
   **ohne** Kern deutlich langsamer (auf gfx12 waren das 264 gegen 663 tok/s).
3. **Speicher:** die **Experten-Slots** des 4-Bit-Arms gegen `int8`, getrennt gemessen: unter 64K
   (ohne KV-Streaming, der KV liegt im Pool) und über 64K (mit Streaming, der KV liegt im RAM). Der
   Gewinn wird **pro Fall** berichtet, nicht verrechnet.
4. **Logit-Gate:** KL der Prefill-Logits des WMMA-Prefills gegen den FP32-Kern ≤ 0,005 und gleiches
   Top-p ≥ 98 % (die Schwelle aus dem gfx11-Fork, dessen WMMA-Prefill die Logits messbar verschiebt:
   KLD 0,0293, Top-p 92,59 %).
5. 3 von 3 Nadeln je Lauf.

**Ergebnis A:** alle Punkte erfüllt → der Motor darf auch auf gfx1151 selbst entscheiden; das Terminal
folgt (`terminal/strata_control/kv_default.py`, `setup.py:11177`).
**Ergebnis B:** ein Punkt verletzt → berichten; die Voreinstellung bleibt auf gfx12.
**Ergebnis C:** keine APU-Karte verfügbar → berichten, offen lassen.

**Stand in dieser Umgebung: offen** (kein Strix Halo, kein 780M hier). Alles APU-Spezifische in
diesem Abschnitt ist **statische Analyse** — siehe `docs/research/KV_Q4_MATRIX_PATH_RDNA4.md` §9a.

### A.26 — vorregistriert: die Skalierungsachse (per-Token-Gruppe gegen per-Kanal) (17.11.2026, vor dem ersten Lauf)

**Die Frage.** Unsere `q4u32`-Geometrie legt **eine Skala je 32 Werte entlang head_dim, je Token** (eine
„per-Token-Gruppe"): 4 Bit je Wert plus 0,5 Bit für die Skala = **4,5 bit/Wert**. Die Arbeit sagt das
andere: **K gehört per-Channel, V per-Token** (KIVI; zolotukhin.ai, 14.08.2026: unter 8 Bit bricht die
Qualität ein, und die Outlier-Kanäle vergiften ihre Nachbarn in der Gruppe). Der Vorschlag misst das auf
**unseren echten Zellen** — ohne Engine- und ohne Kernel-Änderung.

**Die Arme** (Zellen-Ebene, im rotierten Raum, in dem unser Format lebt; dieselbe Metrik wie der
Shear-Arm: MSE je Wert, in dB gegen den Anker):

| Arm | Skala | Bit je Wert |
|---|---|---:|
| `q4u32` (Anker) | je Token, je 32 Werte entlang head_dim (Schritt je Block gefittet) | 4,5 |
| `per-kanal` | je (Layer, Kopf, Kanal) **über alle Token** — die Tabelle ist token-unabhängig und liegt in der Modelldatei (1.024 fp16 je Layer und Seite) | 4,0 + ~0 |
| `hybrid` (KIVI) | K per-Kanal, V je Token-Gruppe | wie oben, je Seite |

**Kriterien, vorher festgelegt:**

1. **K:** der per-Kanal-Arm muss den Anker um **≥ 1,0 dB** schlagen. Sonst bleibt die per-Token-Gruppe
   (kein Kernel-Umbau, der Fast Path bleibt wie er ist).
2. **V:** per-Kanal-V wird **berichtet**, nicht angestrebt (KIVI sagt: V per-Token). Ein Gewinn dort ist
   eine Extra-Information.
3. **Struktur (Rechnung, kein Messwert):** eine Skala je Kanal lässt sich **nicht** je 32-Wert-Block
   anwenden — sie müsste auf den **Akkumulator** (die int8-Route), und die kostet dort Arbeit **je Ausgabe**
   statt je Eingabewert (Faktor 16 bei einem 16-Zeilen-Tile; §3 des Research-Dokuments). Ein KIVI-Gewinn ist
   also **kein** Format-Gewinn, sondern der Preis für den int8-Umstieg.
4. **Kill-Regel:** < 1,0 dB auf K ⇒ die Frage ist mit „nein" beantwortet; die per-Token-Gruppe bleibt, und
   das Thema wird nicht wieder aufgemacht (außer es taucht später als Nebenbefund auf).

**Die Messung:** offline auf dem Zellen-Dump des Centering-Projekts (`cells-cwe4k.bin`, die Zellen des
cwe-4k-Korpus, 12 QSA-Layer — dieselbe Datei, auf der der Offset-Arm gemessen wurde). Zusätzlich wird
berichtet, **wie viele Koordinaten** im Anker die Skala bestimmen (die „Outlier-Kanäle"): das erklärt,
**woher** ein Gewinn käme.

### A.26 — Das Ergebnis: die Skalierungsachse ist keine (17.11.2026)

**Für die Veröffentlichung:** `docs/paper/KV-q4u32-matrix-core.pdf` (q4u32: A 4-Bit KV Cache the Matrix Core
Can Read) trägt die Serie A.16–A.26 in der Form, in der sie erscheint; dieses README bleibt der
Arbeitsstand mit den offenen Armen (A.24/A.25) und den Vorregistrierungen.

**Die Messung** (offline, `tools/kv_scale_axis.py`, auf dem Zellen-Dump des Centering-Projekts: 768 Records,
**392.952 Token**, **201.191.424 Werte** je Seite, alle **12 QSA-Layer**; MSE je Wert, dB gegen den Anker):

| Arm | MSE K | dB K | MSE V | dB V |
|---|---:|---:|---:|---:|
| `q4u32` (Anker) | 1,493706e-02 | 0,000 | 2,461929e-03 | 0,000 |
| `per-kanal` | 1,997048e-02 | **+1,261** | 2,696579e-03 | **+0,395** |
| `hybrid` (KIVI: K per-Kanal, V je Token-Gruppe) | 1,997048e-02 | **+1,261** | 2,461929e-03 | 0,000 |

**Das Ergebnis ist das Gegenteil eines Gewinns: die per-Kanal-Skala ist schlechter.** Auf K ist sie in
**12 von 12 Layern** schlechter (+0,715 bis +1,646 dB, Median +1,29 dB); auf V in 10 von 12 (zwei Layer sind
um ~1 dB besser — aber V per-Kanal ist **kein** Ziel, KIVI sagt: V per-Token). Der Hybrid, also KIVIs
Rezept, erbt den K-Verlust ungefiltert.

**Warum — und das ist der eigentliche Befund.** Die Frage, wer im Anker die Skala setzt, beantwortet sie:
die besten **5 von 256** Koordinaten setzen nur **6,3 … 13,6 %** der Blöcke (zwischen den Layern), alle
übrigen sind unauffällig. **Es gibt keine Outlier-Kanäle**, die ihre Nachbarn in der Gruppe vergiften
könnten. Nach der Rotation ist der dynamische Bereich **je Token** verteilt (jede Zeile hat ihre eigene
Lage), und genau das ist unsere Geometrie: eine Skala je 32 Werte entlang head_dim, **je Token**. Die
per-Kanal-Skala muss dagegen den Extremwert über **alle** Token einer Koordinate decken — sie zahlt also
für das Schlimmste und ist für den typischen Token zu grob.

**Das Kriterium (K ≥ 1,0 dB Gewinn) ist nicht erfüllt, sondern umgekehrt: der Kill greift.** Die
per-Token-Gruppe bleibt, der Fast Path bleibt wie er ist (kein Kernel-Umbau), und das Thema wird nicht
wieder aufgemacht. Der Preis der Alternative wäre doppelt: sie ist **schlechter** (oben) **und** sie
lässt sich im Fast Path nicht anwenden (eine Skala je Kanal gehört auf den Akkumulator — die int8-Route,
Arbeit je Ausgabe statt je Eingabewert, Faktor 16 bei einem 16-Zeilen-Tile, §3 des Research-Dokuments).

**Was das für die Literatur bedeutet (Quelle, ausgewertet in `docs/research/KV_Q4_MATRIX_PATH_RDNA4.md` §8):**
die Warnung „unter 8 Bit bricht die Qualität ein, K gehört per-Channel" (zolotukhin.ai, 14.08.2026, nach
KIVI) **gilt für ein Format ohne Hadamard-Rotation**. In unserem Format (Rotation vor der Quantisierung)
ist die Achse per-Channel **geschlossen** — zweimal, zweimal gemessen: die S2-Sonde hatte die
per-Kanal-Skala **vor** der Rotation gekillt (die Vor-Rotation-Kanäle sind nicht der schwerere Schwanz),
und A.26 schließt sie **nach** der Rotation auf unseren Blöcken. Der überlebende Teil derselben Achse
ist der **per-Kanal-Offset** (1,70 dB K / 4,63 dB V), den das Paper bereits als Produktmerkmal führt.

**Die Werkzeug-Konvention:** `tools/kv_scale_axis.py` liest den Dump des Centering-Projekts, spiegelt die
Quantisierung des Motors (`tools/kv_codec_mse.py`) und misst die Achse ohne Engine- und ohne
Kernel-Änderung. Rohdaten: `a26-scale-axis.json` (im Bench-Ordner).

### A.27 — vorregistriert: die Gruppengröße des Blocks (16, 32, 64 Werte) (17.11.2026, vor dem ersten Lauf)

**Die Frage.** Die Gruppengröße ist der eine Knopf, den keine Messung dieser Linie verändert hat: die
Skala kostet **0,5 bit je Wert** bei 32 Werten je Block (4,5 bit/Zelle), **0,25** bei 64 (4,25 bit/Zelle) und
**1,0** bei 16 (5,0 bit/Zelle). Die Byteseite ist eine Rechnung; die **Qualitätsseite** ist offen: eine
größere Gruppe deckt einen weiteren Dynamikbereich, eine kleinere kostet Bytes.

**Die Arme** (Block-Ebene, dieselbe Metrik wie A.26: MSE je Wert, in dB gegen die 32er-Gruppe; dieselben
echten Zellen, derselbe Dump):

| Arm | Gruppe | Bytes je Wert | erwartete Route |
|---|---:|---:|---|
| `g32` (Anker) | 32 | 4,5 | die heute gebaute Geometrie |
| `g16` | 16 | 5,0 | zwei Skalen je 32er-Gruppe des Kerns |
| `g64` | 64 | 4,25 | eine Skala über zwei Kern-Gruppen |

**Kriterien, vorher festgelegt:**

1. **Qualität:** der Fehler je Gruppe gegen die 32er-Gruppe: die dB-Zahl je Arm, **kein** Schwellwert —
   berichtet wird, was der Handel kostet.
2. **Struktur (Rechnung, kein Messwert):** die Gruppengröße ist **nicht** ein freier Knopf, sondern an
   die Kerngeometrie gebunden: der Prompt-Kern wendet die Skala **je 32 Dimensionen** an, und ein Block ist
   eine Zeile entlang head_dim — eine 64er-Gruppe überspannt also **zwei** Kerngruppen und kann ihre eine
   Skala nicht je Gruppe anwenden. Eine 16er-Gruppe passt in eine Kerngruppe, kostet aber das Doppelte an
   Skalen-Bytes.
3. **Kill-Regel:** bleibt die 32er-Gruppe sowohl qualitäts- **als** strukturseitig die einzige mögliche
   Geometrie, ist der letzte offene Knopf des Formats geschlossen: das Format ist fest (4,5 bit/Zelle,
   eine Skala je 32 Werte, uniformes Gitter, kein Codebook, kein Nullpunkt, keine Kalibrierung).

**Die Messung:** offline auf demselben Dump (`tools/kv_scale_axis.py`, eine Minute CPU), mit der
Gruppengröße als Parameter; keine Engine- und keine Kernel-Änderung.

### A.27 — Das Ergebnis: die Gruppengröße ist an die Kerngeometrie gebunden (17.11.2026)

**Die Messung** (offline, `tools/kv_scale_axis.py --groups 16,32,64`, dieselben Zellen wie A.26: 768 Records,
392.952 Token, 201.191.424 Werte je Seite; dB gegen die 32er-Gruppe):

| Arm | Gruppe | Bit je Wert | dB K | dB V |
|---|---:|---:|---:|---:|
| `g16` | 16 | 5,000 | **−1,185** | −1,163 |
| `g32` (Anker) | 32 | 4,500 | 0,000 | 0,000 |
| `g64` | 64 | 4,250 | **+0,924** | +0,889 |

**Die Rechnung geht auf:** eine größere Gruppe deckt einen weiteren Dynamikbereich und verliert **0,92 dB**
dafür, dass sie **0,25 bit je Wert** einspart (4,5 → 4,25); eine kleinere Gruppe gewinnt **1,19 dB** und
kostet dafür **0,5 bit je Wert** mehr (4,5 → 5,0). Beides ist der Handel zwischen dem Dynamikbereich der
Gruppe und den Bytes ihrer Skala — die Skala ist bei 32 Werten **0,5 bit je Wert** und damit **ein Zehntel**
des Codes.

**Und die Kerngeometrie entscheidet, welche Seite baubar ist (statische Analyse, keine Messung — und
eine Korrektur der früheren Fassung dieser Notiz):** der Prompt-Kern wendet die Skala **je 32
Dimensionen** an, und ein Block ist eine Zeile entlang head_dim. Ein Block, der **größer** ist als die
Kerngruppe, wird von derselben Skala zweimal bedient: die Skala ist **innerhalb** jeder 32er-Gruppe
konstant, und der Kern faltet sie in den Operanden (`qsa_prompt_attn.cu`: die Stufe der Gruppe wird aus dem
Block gelesen und in die Codes gefaltet). Eine **64er**-Gruppe bräuchte also drei lokale Änderungen (den
Blockindex, den Lane-Offset im Block und die Nibble-Regel von zwei auf vier Quadranten), und die
Stufen-Abfrage läse jede zweite Gruppenstufe aus demselben Block — sie ist **baubar**. Ein Block, der
**kleiner** ist als die Kerngruppe, ist es nicht: zwei Blöcke fallen in eine Gruppe, und der Kern wendet
eine Stufe je Gruppe an — die **16er**-Gruppe fällt damit heraus.

**Der Handel ist also ein Handel, kein Abschluss:** die 64er-Gruppe spart **0,25 bit je Wert** (5,6 % vom
Bulk; auf der gemessenen Karte rund **40 zusätzliche Experten-Slots** bzw. +1,2 %) und kostet **0,92 dB**
Blockfehler — über die gemessene Block→nats-Brücke (0,00116–0,00167 nats je dB) rund **+0,0012 nats**.
Gegen ein Produktarm, das im Mittel auf 	exttt{int8} liegt und im p99 bereits dahinter, ist das die
falsche Richtung: es kauft Bytes, indem es die Marge verbraucht, wegen der das Format gewählt wurde.

**Damit ist die Kill-Regel erfüllt: die 32er-Gruppe ist die einzige mögliche Geometrie, und der letzte
offene Knopf des Formats ist geschlossen.** Das Format steht fest: **4,5 bit je Zelle, eine Skala je 32
Werte, uniformes Gitter, kein Codebook, kein Nullpunkt, keine Kalibrierung** — und es ist auf der
Kernel-Geometrie **angewiesen**, nicht nur mit ihr vereinbar.

**Was das für die Produktentscheidung heißt:** es gibt nichts mehr zu tun am Format. Die offenen Punkte der
Linie sind damit nicht mehr Formatfragen, sondern (a) die **Gruppengröße im Byte-Modell** — sie ist
geschlossen —, (b) die **Skalierungsachse** (A.26, geschlossen), (c) der **per-Block-Offset** (eine
Eigenschaft des Caches, kein Hebel auf das Format), (d) die **A.24/A.25** (gfx11 und APUs, Hardware fehlt)
und (e) der **int8-Umstieg** (die int8-Route, kein Formatproblem).

**Die Entscheidung (17.11.2026):** die Gruppe bleibt **32**. Die 64er-Gruppe ist baubar, bringt aber
**1,2 % Slots gegen 0,92 dB** — das ist die falsche Richtung für ein Format, dessen Marge gegenüber
\texttt{int8} der Grund der Wahl war; die 16er-Gruppe ist nicht baubar. Das Produktionsformat ist damit
**4,5 bit je Zelle, eine Skala je 32 Werte, uniformes Gitter, kein Codebook, kein Nullpunkt, keine
Kalibrierung** — und bleibt es.

### Korrektur zu A.20

A.20 schließt mit „Der Prompt-Kern ist ~5–7 % wert, nicht 2,3× — der Faktor aus A.1 war der ganz fehlende
Pfad". Der zweite Halbsatz ist **falsch** und wird hiermit zurückgenommen: A.21 misst für `q4u32` ohne
Kern **2,47×** (265 gegen 655 tok/s), und A.1s 2,3× war genau dieser Pfad, nicht ein Messfehler. Richtig
ist: **5–7 %** ist der Wert des Kerns für `q4_0` (das im Decode-Kern schon die schnelle Instantiation
nutzt); **2,5×** ist er für `q4u32` wert. Die A.20-Tabelle und der `q4_0`-Befund bleiben unberührt, nur
die Verallgemeinerung auf `q4u32` war falsch.

### A.21 — vorregistriert: `q4u32` ohne den Schalter, also der heutige Default-Zustand (17.11.2026, vor dem ersten Lauf)

**Die Frage.** A.20 misst `q4u32` **mit** `STRATA_KV_Q4_PROMPT=1` (Matrix-Core-Prompt-Kern) und `q4_0`
**ohne** Prompt-Kern. Damit ist der Unterschied zweier Arme gemessen, aber **nicht** der heutige
Default-Zustand von `q4u32`: der Schalter ist opt-in, ein `q4u32`-Default liefe heute im **Decode-Kern**.
Der `q4_0`-Arm (631,3 tok/s) ist dafür nur ein Stellvertreter — gleicher Kern, anderer Codec.

**Die Arme** (drei Starts je Arm, 3 Nadeln, 32K, 30 s, dieser Build):

| Arm | KV | Env | Cache | erwartete Route |
|---|---|---|---|---|
| `int8` | int8 | — | `--expert-cache 2967` | int8-Matrix-Pfad (keine Zeile) |
| `q4u32 off` | q4u32 | **ohne** `STRATA_KV_Q4_PROMPT` | `--expert-cache 2967` | **Decode-Kernel**, Zeile „the matrix-core prompt path for the 4-bit grid is opt-in" |

**Reihenfolge:** `int8`, `q4u32 off`, `q4u32 off`, `int8`, `int8`, `q4u32 off` (verschränkt, jeder Arm
einmal am Anfang, in der Mitte, am Ende).

**Kriterien, vorher festgelegt:**

* Ein Lauf zählt nur mit **3950 Slots ±1 %**, **Chunk 6144** und der **Opt-in-Zeile** (Decode-Kernel).
  Erscheint stattdessen „matrix-core prompt path serves the 4-bit uniform grid", lief der Schalter
  entgegen der Annahme **an** — der Lauf ist ungültig und wird nicht umgedeutet.
* Niveau je Anfrage (≥ 40 s langsam), verglichen wird nur innerhalb eines Niveaus.
* **Erwartung (Hypothese, kein Ergebnis):** `q4u32 off` liegt auf dem Niveau des `q4_0`-Arms aus A.20
  (631,3 tok/s), also ~5–7 % unter `int8`. Trifft das zu, ist der Schalter genau diese 5–7 % wert.
* **Ergebnis A:** `q4u32 off` liegt innerhalb 5 % von `int8` → der Matrix-Kern trägt im Motor bei 32K
  nichts Messbares, und der Default kann ohne ihn bleiben.
* **Ergebnis B:** `q4u32 off` liegt ≥ 5 % unter `int8` und auf dem Niveau von A.20s `q4_0` → der Schalter
  ist die 5–7 % wert; die Entscheidung „Schalter als Default" wird damit eine Risiko-, keine
  Geschwindigkeitsfrage.
* **Ergebnis C:** kein Niveau mit mindestens zwei Armen, oder die Route-Zeile fehlt → berichten, kein
  weiterer Start derselben Bauart.
* **Kein Decode-Vergleich**, keine Aussage über andere Kontexte als 32K.


### A.28 — vorregistriert: 128K, `int8` gegen `q4u32` (20.11.2026, vor dem ersten Lauf)

**Die Frage.** Der KV-Default gilt für **alles über 8K** (das Terminal lässt `--kv` dort weg). Die Zahlen im
PR sind 32K (Tempo) und 4K (Qualität). Bei 128K ist der KV-Teil viermal so groß — dort ist der Gewinn am
größten (der frei gewordene Platz geht in den Experten-Cache) und das Risiko am größten (131.060 Positionen
je Schicht statt 4.096 bei 4K).

**Was schon vorliegt (nicht neu gemessen, aber bisher in keinem PR-Text).** Der Qualitätsvergleich bei 128K
stammt vom **13.11.2026** (`Strata-data\bench\2026-11-13-128k-fair\`): 131.060 Positionen, gleiche Texte,
`int8` (mit Rotation) gegen `q4u32`, Build von damals, **ohne** Matrix-Core (auf Windows war der Schalter
damals aus; der Codec ist derselbe, die Aufmerksamkeit rechnete im FP32-Kern):

| Arm | mittlere KL gegen fp16 | Top-1 |
|---|---:|---:|
| `int8` (R2a) | 0.06238 | 98.49 % |
| `q4u32` | 0.06949 | 98.31 % |

gepaart über die Positionen: **+0.00530** KL, t **+3.44**, 95 %-Intervall [+0.00228, +0.00832]; McNemar
864/1000, p = 0.0018. **Bei 128K ist `q4u32` also etwas schlechter als `int8`** — klein (rund +8 % der KL
des int8-Arms), aber statistisch auflösbar. Bei 4K (cwe-4k) war der Unterschied +0.00052 (t −0.56), also
nicht auflösbar. Diese Zahl kommt in den PR-Text nach.

**Was neu gemessen wird.** Das *Tempo* bei 128K im ausgelieferten Zustand (Matrix-Core an, **kein**
gesetzter Schalter) und der Speichergewinn. Der einzige 128K-Lauf bisher lief ohne Matrix-Core, also gerade
nicht in der Konfiguration, die der Default ausliefert.

**Die Arme** (Swift IQ3_XXS, `--max-context 133120`, ROCm 7.2, gfx1200, ein Build, 30 s zwischen den
Starts, Tiefen 10/50/90 %):

| Arm | KV | Cache | Zweck |
|---|---|---|---|
| `int8-2500` | int8 | `--expert-cache 2500` | gepaartes Tempo |
| `q4u32-2500` | q4u32 | `--expert-cache 2500` | gepaartes Tempo |
| `int8-auto` | int8 | `--expert-cache auto` | der ausgelieferte Zustand: Slots und Tempo |
| `q4u32-auto` | q4u32 | `--expert-cache auto` | der ausgelieferte Zustand: Slots und Tempo |

Der feste Cache (2500 Slots) hält die Zahl der Experten-Slots in beiden Tempo-Armen **gleich**; die
Auto-Arme zeigen, was ein Start ohne Flags bekommt. Bei 128K braucht der KV-Teil viermal so viel Platz wie
bei 32K: der Slot-Blob ist 2,33 MB, der KV-Teil liegt bei 128K bei 1.584 MiB (int8) gegen 863 MiB
(q4u32), und die Auto-Zahl lag bei 32K bei 2.888 Slots (int8). 2500 Slots passen deshalb in **beide**
Tempo-Arme, ohne dass die Karte paginiert.

**Die Befehle** (Treiber: `bench/results/2026-11-14-q4u32-fastpath/run-runtime-128k.py`):

```
python -m serve.server --engine strata --config <config des Arms> --port 7603 --no-api-key
python tools/needle_bench.py --url http://127.0.0.1:7603 --lengths 128k --depths 10,50,90 --timeout 3600 --out needles-<arm>.json
```

**Die Gates, vorher festgelegt:**

* G-R8 wie in der Serie: VRAM-Budget **≥ 14.000 MiB** und Expert-Slots **≥ 1.000**, sonst ist der Arm
  *ungültig* (nicht langsam) und wird nach 90 s einmal wiederholt. Dazu die Paging-Warnung: steht sie im
  Log, liest der Arm die Experten über die SSD, und der Arm ist ungültig.
* Jeder Arm muss die Zeile **"the prompt attention on matrix cores on"** zeigen — der Default des Builds,
  kein gesetzter Schalter. Fehlt sie, ist der Arm ungültig.
* Der `q4u32`-Arm muss **"the matrix-core prompt path serves the 4-bit uniform grid"** zeigen; der `int8`-Arm
  darf keine 4-Bit-Zeile zeigen.
* Drei Prompt-Read-Raten je Arm (Tiefen 10/50/90 %), alle bei 128K; sie kommen aus dem Engine-Log
  (`prompt N tokens = ... read in X ms (Y tok/s)`).
* Alle drei Nadeln gefunden (der Harness prüft das selbst).

**Die erwarteten Ergebnisse:**

* **E1 (erwartet):** `q4u32-2500` liegt im Prompt-Tempo bei 128K **innerhalb 5 %** von `int8-2500`
  (Mittel über die drei Tiefen). Dann trägt die 32K-Zahl auch bei 128K.
* **E2 (erwartet):** `q4u32-auto` hat **mehr Expert-Slots** als `int8-auto` — der KV-Platz kommt im Cache
  an. Bei 32K waren es +18,7 % (3428 gegen 2888); bei 128K ist der KV-Anteil größer, also sollte der
  Abstand größer ausfallen.
* **E3 (eingreifend):** `q4u32-2500` liegt bei 128K **mehr als 5 %** unter `int8-2500`, oder ein Arm passt
  nicht (Paging, zu wenig Cache). Dann wird der Default **kontextbegrenzt** (z. B. nur bis 64K) — das ist
  eine Code-Änderung, und sie wird **vor** dem Push gemacht, nicht danach.
* **Was dieser Lauf nicht kann:** die Qualität bei 128K (liegt vor, s. o.), die gfx11-Familie und die APUs
  (keine Hardware), jede Aussage über Karten mit mehr VRAM, und die Fenster-Varianten aus Teil D (#3).

### A.28 — Das Ergebnis: bei 106K Tokens ist `q4u32` gleich schnell und gibt 17 % mehr Experten-Slots (20.11.2026)

**Alle Vorbedingungen erfüllt.** Ein Build (`0.1.63-q4u32`, SHA-256 `f2fd770cbc9048a4217486512a48a00ad9ccceb0f13ff560928b792d8efaa0a8`),
vier Arme, VRAM-Budget 15.419 MiB in jedem Arm, **keine** Paging-Warnung, **12 von 12 Nadeln gefunden**,
Prompt-Chunk 6144 in allen vier Armen. Und in allen vier Armen steht

```
strata: the prompt attention on matrix cores on (gfx1200) by default; STRATA_HIP_WMMA=0 turns it off
```

— der **Default** des Builds, kein gesetzter Schalter (`STRATA_HIP_WMMA` war in keinem Arm gesetzt). Das ist
zugleich der erste Nachweis, dass die geteilte Vorgabe aus dem PR (Matrix-Core auf gfx12 unter Windows **und**
Linux; die Linux-only-Schalter `STRATA_PF_FUSED` und `STRATA_MMQ_KQ` bleiben Linux-only) im Motor so
ankommt, wie sie gemeint ist. Die beiden `q4u32`-Arme zeigen zusätzlich „the matrix-core prompt path serves
the 4-bit uniform grid", die beiden `int8`-Arme keine 4-Bit-Zeile.

**Die Prompt-Read-Raten** (106.252 Prompt-Tokens je Anfrage, Tiefen 10/50/90 %, tok/s):

| Arm | 10 % | 50 % | 90 % | Mittel | Median | Slots | VRAM frei nach dem Laden |
|---|---:|---:|---:|---:|---:|---:|---:|
| `int8-2500` | 736,5 | 741,7 | 752,1 | **743,4** | 741,7 | 3322 | 1805 MiB |
| `q4u32-2500` | 742,5 | 740,9 | 729,1 | **737,5** | 740,9 | 3322 | 2629 MiB |
| `int8-auto` | 749,0 | 749,3 | 728,8 | **742,4** | 749,0 | 3079 | — |
| `q4u32-auto` | 731,9 | 738,9 | 719,2 | **730,0** | 731,9 | 3608 | — |

**Die gepaarten Arme (gleiche Slot-Zahl):** `q4u32` liegt **−0,8 %** im Mittel und **−0,1 %** im Median
gegenüber `int8`. Das ist **E1**: innerhalb der 5 %, die vorab festlagen. Die 32K-Zahl (A.23) trägt also
auch bei 106K Tokens.

**Die Speicher-Arme (der ausgelieferte Zustand, `--expert-cache auto`):** `q4u32` bekommt **3608 Slots
(5,88 GiB)** gegen `int8`s **3079 Slots (5,03 GiB)** — **+17,2 %** Experten-Slots. Das ist **E2**: der
KV-Platz kommt im Cache an. Bei 32K waren es +18,7 % (A.23); bei 106K sind es 17,2 %, weil der KV-Teil
relativ zum übrigen Speicher größer ist. Die 824 MiB mehr freie VRAM im gepaarten `q4u32`-Arm (2629 gegen
1805 MiB) sind derselbe Effekt in der anderen Richtung.

**E3 ist nicht eingetreten:** kein Arm lag mehr als 5 % unter dem anderen, kein Arm passte nicht. Der
Default bleibt für alles über 8K; **eine Kontextbegrenzung ist nicht nötig**.

**Was der Lauf einschränkt (und was nicht):**

* **Die Prompt-Länge ist 106.252 Tokens, nicht 131.072.** Das „128k" des Harness ist eine **Zeichen**-Länge
  (der Harness schneidet den Text nach Zeichen, ~3,2 je Token); die 32K-Läufe waren 30.782 Tokens. Das
  Kontextfenster stand auf 133.120 Tokens, der Prompt füllte 80 % davon.
* **Drei Anfragen je Arm**, nicht neun wie bei 32K. Die Spannweite je Arm ist 1,8–2,8 %, die
  Mittelwert-Unterschiede sind kleiner als die Streuung — die Aussage ist „gleichauf", nicht „auf die
  hundertstel genau gleich".
* **Die Auto-Arme sind kein gepaarter Tempo-Vergleich** (sie haben unterschiedliche Slot-Zahlen: 3079 gegen
  3608). Sie beantworten die Speicher-Frage; das Tempo ist der gepaarte Vergleich. Und dort liegt `q4u32`
  mit −1,7 % (Mittel) **unter** `int8`, obwohl es **mehr** Slots hat — der Unterschied ist die Slot-Zahl,
  nicht das Format.
* **Die Qualität bei dieser Länge ist nicht Teil dieses Laufs.** Sie liegt vor (13.11., 131.060 Positionen:
  `q4u32` +0.00530 KL gegen `int8`, t +3.44, Top-1 98,31 % gegen 98,49 %) und ist **kleiner**, als die
  Tempo-Differenz oben vermuten lässt, aber sie ist auflösbar. Der Default deckt also einen Bereich ab, in
  dem das Format nicht gleichwertig zu `int8` ist; das gehört in den PR-Text.
* **Ein Lauf, ein Tag, ein Build.** Alle vier Arme laufen auf demselben Build am selben Tag; ein Arm auf
  einem anderen Build wäre verworfen (A.6, Fall 5).

### A.28 — Nachtrag: die Kontextgrenze im Motor (20.11.2026, nach einem Review)

Der Review zum PR fand einen Widerspruch: der PR sagt „above 8K of context", der Motor aber wählte `q4u32` in
**jedem** Kontext, sobald kein `--kv` genannt war — auch bei 4K, wo das Release fp16 hält. Ein Start ohne
`--kv` und höchstens 8K Zellen wäre damit **nicht** byte-identisch zum Release gewesen, auch direkt am Motor
(ohne Terminal). Der Motor prüft jetzt selbst (`kv_default_mode(max_context)`: bis 8192 Zellen fp16, darüber
die gemessene Wahl), und dieselbe Grenze hat das Terminal schon (A.28, `context > 8192`).

Geprüft mit **zwei Starts ohne `--kv`** auf derselben Karte (RX 9060 XT, Windows, ROCm 7.2; Build nach der
Änderung, die Kernel byte-identisch, das Parity-Gate unverändert):

| `--max-context` | die eigene Zeile des Motors |
|---|---|
| 8192 | `strata generate: KV storage: fp16 (the default up to 8K of context; --kv names another)` |
| 16384 | `strata generate: KV storage: q4u32 (the default on this part above 8K: 54.5 % of int8's bytes, level with it at 32K and 106K; --kv int8\|fp16\|q4_0 names another)` |

Rohdaten: `Strata-data\bench\2026-11-14-q4u32-fastpath\logs\a28-post-ctx8192.engine.log` und
`a28-post-ctx16384.engine.log`. Damit ist jeder Start ohne `--kv` bei höchstens 8K Zellen byte-identisch zum
Release, und die Grenze steht an **zwei** Stellen (Motor und Terminal), nicht nur an einer.

---

## Teil B — Die Engine-Arbeit

Zwei Backends, zwei Wege. Beide sind „statische Analyse", bis der Paritätstest sie bestätigt.

### B1 — CUDA: der Uniform-Grid in den Prompt-Pfad

* **Änderung:** eine Zeile in `qsa_prompt_attn.cu:2526`. Der Uniform-Grid wird zugelassen, **solange
  kein Overlay hängt** — der Prompt-Pfad liest keinen Window, ein Overlay-Arm dürfte dort nicht
  laufen (sonst kämen die „exakten" Zellen aus dem Bulk: falsches Ergebnis, kein Fehler).
* **Gate:** `qsa_prompt_attn_parity` (der CUDA-Build) mit einem **neuen Fall**: ein Pool im
  Uniform-Layout muss **bitgleich** dasselbe liefern wie ein q4_0-Pool mit denselben Codes und
  Skalen — der Leser ist derselbe, also muss die Ausgabe identisch sein.
* **Risiko:** keins, solange der Overlay-Arm abgelehnt bleibt. Der `q4_0`-Leser hat keine
  Level-Tabelle; das ist der ganze Unterschied.

### B2 — HIP: der Matrix-Core-Kern liest den 4-Bit-Pool

Das ist die eigentliche Arbeit, und sie ist auf unserer Maschine die entscheidende, weil dort heute
**kein** q4-Codec den Matrix-Core-Pfad bekommt.

* **Struktur heute** (`qsa_prompt_attn.cu:1887-1966`, `:1966-2226`): Welle `w` besitzt die Dims
  `[64w, 64w+64)` — **genau eine** int8-Skalengruppe (64 Dims). Sie liest die V-Codes (4 × `uint4` je
  Zeile), die V-Skala ihrer Gruppe, die K-Fragmente und die K-Skala ihrer Gruppe. Die int8-Codes gehen
  **exakt** als FP16 in den WMMA (`i8_to_h`), die Skala wird in FP32 angewandt.
* **Der 4-Bit-Grid ändert drei Dinge:**
  1. **Die Datenquelle:** statt 1 Byte je Wert 18 Byte je 32 Werte (16 Nibble-Codes + fp16-Skala); die
     Nibbles müssen zu den exakten int8-Codes `(code−8)` expandiert werden (|code−8| ≤ 8, in FP16
     exakt — dieselbe Arithmetik wie im int8-Pfad).
  2. **Die Gruppen:** 8 Gruppen à 32 Dims statt 4 à 64. Eine Welle trägt dann **zwei** Gruppen; die
     Teilsummen pro Gruppe (`part[4][R][WCH+1]`), die V-Skalen (`vs[4][WCH]`) und das V-Staging
     (`v[4][WCH][VS]`) wachsen auf acht Gruppen.
  3. **Die Summationsreihenfolge:** die Skala wird jetzt zweimal je Welle angewandt statt einmal, die
     Reihenfolge der Gruppensummen ändert sich. Das Ergebnis bleibt FP32-nah, aber **nicht bitgleich**
     zum int8-Pfad (das ist im Paritätstest die 1e-4-Gate, nicht die Bitgleichheit).
* **Gate:** `hip_prompt_attn_wmma.exe` mit einem **neuen 4-Bit-Fall**: der Fehler gegen FP64 nicht
  größer als das Vielfache des int8-Pfads, und die Ausgabe gegen den FP32-Pfad auf 1e-4 der
  Ausgabeskala. Zusätzlich **timt** der Test den 4-Bit-Kern gegen den int8-Kern: das ist die erste
  **gemessene** Zahl zur Frage „ist der Uniform-Grid auf dem Matrix-Core-Pfad schneller als int8?".
* **Reihenfolge inside B2:** erst der Leser (Codes holen und expandieren), dann die Gruppen, dann die
  Skalen; nach jedem Schritt der Paritätstest.

### B3 — Das Fenster auf dem Fast-Path (entschieden: es kommt drauf)

> **Stand:** dieser Abschnitt ist die **Planung** (15.11., vor dem Bau). Gebaut und gemessen ist er in
> `A.16` (Gate rot) und `A.17` (Pass, Gate grün). Die Zeilennummern hier sind der Stand der Planung.

* **Die Entscheidung (2026-11-14):** das Fenster **muss** auf dem Fast-Path laufen. Der Produktarm
  (`q4u32` + Fenster, S 256/R 64) ist die Konfiguration, die im Paper steht; ein Produktpfad, der den
  Fast-Path verliert, wäre die halbe Sache. B3 ist damit **im Umfang**, nicht mehr offen.
* **Der Zustand vor B3:** der Prompt-Kernel kannte keinen Overlay (kein `k_win`/`v_win`); nur der
  Decode-Kernel liest ihn (`WinRows<KV_MODE>`, `WIN = KV_MODE >= 4`, `qsa_decode_attn.cu:213`).
* **Die Schwierigkeit, richtig gestellt:** der shared memory des Prompt-Kernels ist typisiert — `KElem`/
  `VElem` sind entweder `int8_t` oder `__half` (`qsa_prompt_attn.cu:96-109`). Der WMMA-Kern von v2 rechnet
  aber **fp16 × fp16** (`wmma_f16`), die int8-Codes sind nur Zulieferung. Damit ist die Frage nicht „int8
  oder fp16", sondern **„woher kommt der Operand je Zelle"** — und das ist für K und V verschieden.
* **Der Kern rechnet fp16, nicht int8 — und das entscheidet den Weg (statische Analyse, 15.11.2026,
  korrigiert):** `wmma_f16` ist
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c)` mit `wh8 = _Float16[8]`
  (`qsa_prompt_attn.cu` 1917–1918, 1941–1950): **fp16 × fp16 → fp32**. Die int8-Codes sind nur
  Zulieferung — `i8_to_h()` (1952) gießt sie exakt nach fp16, und die Blockskala wird **danach in FP32**
  auf die Gruppensumme angewandt (`s * cd.ksc[nt][g]`, 2197). Eine fp16-Zeile aus dem Seitenpool ist für
  diesen Kern also der **natürliche** Operand, kein Fremdkörper.
  *(Zwischenstand, den ich hier zuerst falsch notiert hatte: „der Kern rechnet int8 × int8, eine
  fp16-Zeile ist nicht als Code mit einer Skala je 32 Werte darstellbar, also ist gemischter smem
  ausgeschlossen". Der erste Satz ist falsch, der Schluss damit auch. Ein Korrektur-Pass mit
  `q·(k_fp16 − k_quant)` ist **nicht nötig**.)*
* **Der Weg, der daraus folgt: dieselbe WMMA, exakt gefüttert.** K und V sind verschieden:
  * **K: kostenlos.** Der K-Operand wird je Lane in Registern gebaut (`cd.kx[nt][kk]`, 2136–2159) und
    erst im WMMA-Schritt konvertiert (2190). Für eine Fensterzelle lädt man statt der 8 Codes **8
    Halves** aus dem Seitenpool (16 B, ausgerichtet wie `load8_win`) und setzt `ksc = 1.0`. Es gibt
    keinen Konflikt: die Skala wirkt je Zelle in FP32, nicht auf die Kachel.
  * **V: ein zweiter WMMA-Satz, kein Korrektur-Pass.** V wird als **Bytes** gestagt
    (`S.v[4][32][72]`, gelesen als Codes in 2269–2270); 64 Dims als Halves wären 128 B, das Polster
    gibt es nicht (und die V-Zeile auf 128 B zu verbreitern kostet 7 KiB LDS je Block — bei ~24 KiB
    heute der Sprung von 5 auf 3 residente Blöcke). Also: Fensterzellen in der quantisierten Passung
    auf **0** setzen (beim Stagen, nicht im heißen Loop) und einen **zweiten** Satz von 4 Kacheln × 2
    k-Schritten über genau diese Zellen rechnen, mit `B = V_exakt` aus dem Seitenpool und `A = p`
    (ohne `vdown`, in einen eigenen `tmpw`-Akkumulator, der ohne `vdown` in `acc` geht). Der Zweig ist
    warp-uniform und läuft nur in Chunks, die überhaupt Fensterzellen enthalten — bei S 128/R 8 sind
    das ~7 % der Zellen, also die letzten Chunks; die übrigen ~93 % zahlen **nichts**.
  * **Maske:** `S.valid[c] = row >= 0 || wrow >= 0` — genau die Regel des Decode-Kerns
    (`srow[c] < 0 && swin[c] < 0`, `qsa_decode_attn.cu` 293).
  * **Kosten:** die Schätzung, die hier stand (~7 % zusätzliche V-WMMA-Arbeit), ist in `A.18` gemessen;
    das K-Fenster kostet nichts.
* **Was das für den Arm bedeutet (Stand vor dem Bau):** `q4u32` **mit** Fenster war im Prompt-Pfad
  `false` (`q4u_wmma_ok` verlangte `k_win == nullptr`) und lief über den Decode-Kern; das ist der Arm,
  den A.1 mit 260,3 tok/s gemessen hat. B3 heißt: dieser Arm soll auf den Matrix-Pfad — mit den zwei
  Quellen oben, ohne Quantisierung der Fensterzellen.
* **Gate:** ein Parity-Fall mit Overlay-Zellen gegen den Decode-Pfad — der Decode-Pfad ist die
  Referenz, weil er das Fenster heute schon liest. Der Fall muss in `hip_prompt_attn_wmma.exe` neu
  gebaut werden (das Fenster ist dort bisher **nirgends** getestet, B6/Befund 6). **Der Fall muss
  „rot" auch wirklich rot sein:** mit einem Fenster lehnt `qsa_prompt_attn_batch` heute ab
  (`return false`, 2694–2696), und der Harness wertet das als Abbruch — nicht als grünes `PASS` über
  den Decode-Kern (die Falle aus B6).

### B4 — Die Reihenfolge und die Kosten

| Schritt | Inhalt | Gate | Aufwand |
|---|---|---|---|
| B1 | CUDA-Dispatch, eine Zeile | Parität (bitgleich) | klein |
| B2 | HIP-Matrix-Core liest 4 Bit | Parität + Timing | **groß** |
| B3 | Fenster auf dem Prompt-Pfad (entschieden: es kommt drauf): K exakt als fp16-Operand, V als zweiter WMMA-Satz | Parität mit Overlay | **mittel** (kein Korrektur-Pass, s. o.) |
| B4' | **Logzeile** „der Prompt-Pfad bedient diesen Arm" | G-R1 | klein |

**B4' ist Voraussetzung für A2**, nicht für A1: ohne die Zeile ist die Zuordnung „Zeit ↔ Pfad"
nicht belegbar.

### B5 — Was uns von „voll einsatzbereit und optimiert" trennt (Stand 15.11.2026)

Die Frage aus B lautet nicht „welche Optimierung fehlt", sondern „welche Lücke blockiert welche
andere". Es sind vier, und die Reihenfolge ist nicht frei wählbar:

| # | Lücke | Zustand | Blockiert | Wodurch entschieden |
|---|---|---|---|---|
| 1 | **Der Hänger** mit `STRATA_KV_Q4_PROMPT=1` | **erledigt** (`A.14`/`A.15`): Doorbell-Deadlock im Fenster-Launch, Helfer-Thread ist Default | nichts mehr | A.14 (Sonde), A.15 (Fix + 4K/32K 3/3) |
| 2 | **Das Fenster** auf dem Fast-Path (B3) | **gebaut, Gate grün** (A.17); im Motor belegt (A.18: Route-Zeile, 16/16 Nadeln), Kosten **+6,2…13,4 %** der Attention-Kernelzeit (Harness-A/B) | nichts mehr | A.16 (rot), A.17 (grün), A.18 (Route + Kosten) |
| 3 | **Die Rate** des 4-Bit-Kerns | **offen**: A.15 642,8 gegen int8 649,0 tok/s (99,1 %) — **beide im langsamen Niveau**; im schnellen Niveau 878,7 ohne `int8`-Gegenstück; im Harness ist der 4-Bit-**Kern** 37 % langsamer als int8 (11,33 gegen 8,26 ms) | die Aussage „so schnell wie int8" | A2, A.18 |
| 4 | **Der Rest von Teil C** | offen | Feinschliff | erst nach 2 |

**1. Der Hänger ist erledigt — er war nicht der Kern.** Die Prüfung am Code hatte ihn nicht gefunden,
und das war richtig so: die Chunk-Schleife ist blockuniform, `n` hängt nur an `qi`, die Adressierung des
4-Bit-Pools ist Zeile für Zeile die des Decode-Kerns (`base + row*144 + b*18`), die Gruppen- und
Skalenzuordnung stimmt für `GPW = 2`. `A.14` hat ihn dann mit der Sonde lokalisiert: der **Launch** des
6-Token-Verify-Fensters blockiert, während die GPU auf eine **Host-Doorbell** wartet. Der schnellere
Prompt-Pfad verliert diesen Wettlauf, ein langsamerer gewinnt ihn — deshalb sah es nach „q4 hängt" aus,
obwohl es jeden KV-Modus treffen konnte. `A.15` macht den Helfer-Thread zum Default (Windows-HIP-Build),
seither 4K und 32K 3/3 ohne Schalter. Die Reihenfolge, die hier stand (Selektions-Dump, `--prefill 512`,
echter Pool), ist damit **hinfällig** — sie hätte einen Deadlock im Launch gesucht, wo er nicht war.

**Nicht durch Lesen gefunden, sondern durch Messen:** die Längen-Hypothese (`A.12`), der Ort (`A.14`),
der Fix (`A.15`). Das ist die Lehre für B3: der Fensterfall im Harness muss **rot** sein können, sonst
belegt er nichts.

**Gemessen und nicht mehr offen:** dass es an der Länge liegt (`A.12`), dass ein grüner q4-Fall im
Harness ohne `STRATA_KV_Q4_PROMPT=1` den Matrix-Kern überhaupt testet (tut er nicht — er testet den
Decode-Kern und druckt `PASS`), **was der Hänger ist** (`A.14`: ein Doorbell-Deadlock im **Launch** des
Verify-Fensters, den der schnellere Pfad im Wettlauf verliert) und dass er **weg** ist (`A.15`: der
Helfer-Thread ist auf dem Windows-HIP-Build Default, 4K und 32K laufen 3/3 ohne Schalter). Die Rate ist
damit **keine** Lücke mehr: `q4u32` 642,8 gegen `int8` 649,0 tok/s (99,1 %, innerhalb der Streuung von
±4 %).

**2. Das Fenster** ist keine Optimierung, sondern Funktionsumfang (B3, entschieden). Solange es fehlt,
ist `q4u32` + Fenster — die Konfiguration, die im Paper steht — **kein** Fast-Path-Arm.

**3. Die Rate war der *letzte* Hebel, nicht der erste — und sie ist gemessen.** B2 hat den Kern 5,12×
schneller gemacht (97,3 → 19,0 ms/Chunk, gemessen); A.14 zeigt, dass der Arm damit **auf int8-Niveau**
liegt (Prompt: 619,4 gegen 626,7 tok/s, derselbe Build, dieselbe Nadel). Der naheliegende nächste
Schritt — **int8-WMMA** (`wmma_i32_16x16x16_iu8`, laut AMD-Tabelle 2× fp16) —
ist geprüft und **zurückgestellt** (`docs/research/KV_Q4_MATRIX_PATH_RDNA4.md`): die 2× kommen nicht
ganz an (Fremdmessung 1,20× bzw. 1,61×, weil das Descaling je 32-Werte-Block die Hälfte zurücknimmt),
er erzwingt **Q in int8** (ein Qualitätspreis, den wir heute nicht zahlen: beim Nachbarn +0,02 %
Perplexität) und ein für `iu8` **undokumentiertes** K-Mapping. Das ist ein eigenes Vorhaben mit eigener
Messung, kein Anhängsel an B2.

**4. Was nicht mehr zur Debatte steht** (weil A1 es gemessen hat): die Level-Tabelle ist **kein**
Preistreiber (`q4lm32` 268,4 ≈ `q4u32` 263,7 tok/s) — eine LUT-Stufe wäre die falsche Antwort; und
eine fp16-Kopie des Caches zu vermeiden ist erledigt (wir dequantisieren im Register/smem, der Grund,
aus dem llama.cpp PR #31802 überhaupt 17–79 % holt).

**Die Reihenfolge, die daraus folgt:**

1. 4K-Nadel mit `STRATA_KV_Q4_PROMPT=1`: **erledigt** (A.12) — es hängt auch bei 4k; die Länge ist
   nicht die Bedingung, der Übergang zum Verify-Fenster ist es.
2. Den Trace so erweitern, dass er die **Form des Aufrufs** nennt (`n_q`, Route) statt der Selektion
   der ersten Query — sonst ist „kein Trace nach dem Bulk" nicht von „der Tail nimmt einen anderen
   Zweig" zu unterscheiden (B6, Befund 4). **Erledigt, und das Ergebnis verschiebt den Verdacht**
   (A.13): der Tail nimmt gar keinen q4-Kern, das Fenster läuft im Decode-Kern.
3. Den **Launch** des Fensters reparieren (der Helfer-Thread als Default, wo Doorbells bedient werden
   müssen) — **erledigt** (A.15): Default auf dem Windows-HIP-Build, gemessen 3/3 bei 4K und 32K, ohne
   Schalter. A2 ist damit gelaufen: `int8` 649,0 / `q4u32` 642,8 tok/s (99,1 %). `q4_0` und
   `q4u32win` fehlen noch in dieser Serie.
4. B3 (Fenster) auf dem dann laufenden Kern — erst damit ist der Produktarm ein Fast-Path-Arm.
5. Teil C — und **nur** wenn A2 zeigt, dass die Rate noch die Lücke ist, int8-WMMA als eigenes
   Vorhaben. Nach A.14 ist sie es **nicht**; Teil C ist damit Feinschliff, keine Voraussetzung.

### B6 — Review der Änderungen (statische Analyse, keine Messung)

Der beauftragte Review-Lauf ist am Nutzungslimit gestorben; diese Runde ist die Prüfung von Hand. Sie
liest die drei geänderten Dateien (`qsa_prompt_attn.cu`, `qsa_decode_attn.cu`,
`qsa_prompt_attn_parity.cpp`) gegen den Stand `ac33b0c4b`. Alles hier ist **Code-Lesen**, nicht
gemessen — das steht bei jedem Befund dabei. Die Reihenfolge ist die, in der es weh tut.

| # | Befund | Wirkung | Art |
|---|---|---|---|
| 1 | `g_wmma_failed` ist zwischen int8 und q4 **geteilt** | ein q4-Launchfehler schaltet den int8-Matrixpfad für den Rest des Prozesses ab | Fehler |
| 2 | `qsa_prompt_attn_batch_gated` hat **keinen** q4-Zweig | Gate an ⇒ q4 verliert die Gate-Fusion, und die Logzeile sagt es nicht | Lücke |
| 3 | `pa_said` ist **once-per-process** | eine Zeile je Lauf kann die Route je Schicht nicht belegen — genau die Frage beim Hänger | Werkzeug |
| 4 | `QA_Q4TRACE` liest `steps` **ohne Query-Offset** | die Zeile nennt die Selektion der *ersten* Query, nicht die Form des Aufrufs; sie erscheint außerdem nur im q4-Zweig | Werkzeug |
| 5 | zwei leere `if constexpr (Q4) {}`-Blöcke (`qsa_prompt_attn.cu` 2018–2021) | toter Code, der wie ein fehlender Schritt aussieht | Junk |
| 6 | der Harness prüft weder den gated-Zweig (`Q4, GATE=true`) noch den Fensterfall (`k_win != nullptr`) | zwei Konfigurationen, in denen der Kern *nicht* läuft, sind ungetestet | Abdeckung |

**1. Ein Fehlerflag für zwei Routen** (`qsa_prompt_attn.cu`: `g_wmma_failed` 2360, gesetzt in
`launch_wmma` 2490, gelesen in `hip_wmma_usable` 2396 und 2559). `launch_wmma` ist für int8 **und**
q4 dieselbe Funktion; jeder Fehler setzt das globale Flag, und danach lehnt `hip_wmma_usable` **beide**
ab. Für ein opt-in-Experiment heißt das: **ein Fehler im Experiment verändert den Default** (int8
fällt still auf den FP32-Kern zurück, eine Zeile stderr, danach nichts). Der Fehlerfall des
Experiments darf den geprüften Pfad nicht mitnehmen — getrennte Flags, oder das Flag nur im
int8-Zweig setzen.

**2. Der gated-Zweig kennt q4 nicht** (`qsa_prompt_attn_batch_gated`, 2596–2609): seine Bedingung
fordert `pools.k_q4 == nullptr && pools.v_q4 == nullptr` und gibt für einen 4-Bit-Zustand deshalb
immer `false`. Der Aufrufer (`prefill.cpp` 4824) versucht zuerst den gated-Zweig und fällt dann auf
`qsa_prompt_attn_batch` — mit Gate **an** läuft q4 also über den ungated Kern, während die Zeile
„the matrix-core prompt path serves the 4-bit uniform grid" meldet. Die Fusion fehlt, nicht die
Funktion. Fix: die Bedingung um `q4u_wmma_ok` erweitern und `launch_wmma<8, true>` **mit** `qf/o16`
aufrufen (die Instantiierung `Q4, GATE=true` existiert bereits, sie wird nur nie gewählt).

**3. `pa_said` ist einmal je Prozess** (2589–2594): `static bool said` — die erste Route gewinnt, alle
weiteren Aufrufe schweigen. Genau beim Hänger ist die Frage aber „welche Route nahm **diese** Schicht",
und ein Motor, der Bulk-Prompt und Verify-Fenster in einem Prozess fährt, kann mit dieser Zeile nicht
beantwortet werden. Vorschlag: einmal je (Route × `n_q`-Klasse) oder ein Zähler je Route am Ende.

**4. Der Trace nennt nicht die Form des Aufrufs** (2455–2466): `cudaMemcpy(h.data(), steps, …)` liest
`steps[0]`, nicht `steps + q0 * kStepCount` — die Zeile zeigt also die Selektion der ersten Query
eines Aufrufs, für Bulk-Prompt **und** Verify-Fenster gleichermaßen (`cells 1 chunks 1`). „Nach dem
Bulk kein Trace mehr" belegt damit nur dann „der Host stand vor `launch_wmma<8,true>`", wenn der Tail
diesen Zweig überhaupt nehmen würde — und der Trace erscheint **nicht**, wenn der Aufruf im
Decode-Kern oder im gated-Zweig landet (oder wenn `q4u_wmma_ok` wegen `k_win != nullptr` `false`
sagt). Die Deutung von A.12 hängt an dieser Annahme; sie ist nicht belegt. **Der nächste Schritt ist
deshalb kein weiterer Motor-Lauf, sondern eine Zeile im Aufrufer**: `n_q`, `cap` und welche Funktion
geantwortet hat.

**5. Junk** (2018–2021): zwei leere `if constexpr (Q4) { }` hintereinander, direkt vor `const int t =
threadIdx.x`. Sie tun nichts und riechen nach einem entfernten Schritt. Weg damit.

**6. Was der Harness nicht prüft** (`qsa_prompt_attn_parity.cpp`): kein `gated`, kein `k_win`, kein
`STRATA_KV_Q4_PROMPT` (die Falle aus A.12). Er prüft genau die Instantiierung `Q4, GATE=false` ohne
Fenster — also die, die in der Engine am wenigsten Kritik hat. Der gated-Zweig ist heute **nirgends**
getestet, und der Fensterfall ist der, den B3 bauen soll: beides braucht einen Fall im Harness, bevor
es Default werden kann.

**Was das Review *nicht* gefunden hat** (und was damit als geprüft gilt): int8 bleibt byte-identisch
(`NGRP = 4` ist der Default, damit sind `SmemWT`-Layout, Ausdrücke und Reihenfolge die alten; die
Prüfsummen sind unverändert — das ist gemessen, nicht gelesen); die Chunk-Schleife ist blockuniform
(`n` hängt nur an `qi`, kein Barrieren-Deadlock); die 4-Bit-Adressierung ist Zeile für Zeile die des
Decode-Kerns (`base + row*144 + b*18`), und `q4u_block_codes`/`q4u_lane_codes` lesen nie über die 16
Byte eines Blocks hinaus; die Gruppen- und Skalenzuordnung stimmt für `GPW = 2`. **Kein** Befund in
der Arithmetik — der Hänger ist damit weiter offen, aber die Liste der Verdächtigen ist kürzer und die
Werkzeuge, die ihn finden sollen, sind jetzt selbst geprüft.

**Was von diesem Review schon gebaut ist** (Befunde 1, 2, 5 und die Werkzeuge 3 und 4): `g_wmma_failed`
und `g_q4_wmma_failed` sind getrennt, `qsa_prompt_attn_batch_gated` hat seinen q4-Zweig (mit demselben
Schalter, über `q4_prompt_on()`), die zwei leeren Blöcke sind weg, `pa_said` spricht einmal je *Route*,
und der Trace nennt `n_q`/`cap`/Route/Zellen der ersten und letzten Query. Der Harness setzt für fmt 3
jetzt `STRATA_KV_Q4_PROMPT=1` selbst. Gebaut (`strata`, `hip_prompt_attn_wmma`), Harness danach:
`FAILURES: 0`, **Prüfsummen unverändert** (`c1dc89ec2629c0b8`, `82ef1dc3fa5dd4e7`, `6c01e0a92922cca5`)
— die int8- und die q4-Ausgabe sind bitgleich zu vorher. B6, Befund 4, hat im selben Zug den
Verdacht verschoben (A.13).

---

## Teil C — Die Optimierung (beginnt erst nach A2)

### C.0 — Voranalyse: wo der 4-Bit-Fast-Path verliert (15.11.2026; statische Analyse, nichts gebaut; H5a/H7 in C.0.1 nachträglich gemessen)

**Die gemessene Ausgangslage** (A.18, Harness, gleiche Selektionen und gleiche Pools, 32K/1.024 Queries,
je ein Lauf): **int8 8,26 ms/Chunk, `q4u32` 11,33 ms/Chunk** — der 4-Bit-Kern ist **37 % langsamer**. Im
Motor fällt das nicht auf, weil die Attention dort rund **1 %** eines Prompt-Chunks ausmacht (A.18,
Hochrechnung). Diese Voranalyse fragt deshalb: **wo verliert der 4-Bit-Pfad Zeit, und welcher Griff
nimmt sie zurück?** Alles Weitere ist aus dem Code gelesen und als statische Analyse markiert.

**Was der 4-Bit-Pfad zusätzlich tut** (Zeilen aus `engine/src/kernels/cuda/qsa_prompt_attn.cu`):

| # | Was | Wo | Warum es teuer ist |
|---|---|---|---|
| a | **Byte-Laden statt Vektor-Laden** | `q4u_lane_codes` 1999–2007 (`codes[b0+j]`), `q4u_block_codes` 1990–1997 | je Zelle und Lane: K = 2 Kacheln × 4 k-Schritte × 8 **skalare 1-Byte-Ladungen** = 64, V = 2 Blöcke × 16 = 32 → **~96 skalare Ladungen**. Der int8-Pfad: **8 `uint2`- und 4 `uint4`-Ladungen** (2152–2153, 2195–2202), also 12 Vektor-Ladungen. Das ist der größte Einzelposten, und er ist **Instruktionszahl, nicht Bandbreite** |
| b | **Zweimalige Expansion** | Nibble → int8 im Fetch (mit `pack_i8x4`), int8 → fp16 im WMMA-Operand (`i8_to_h` 1952–1957) | der int8-Pfad hat nur die zweite Hälfte; der 4-Bit-Pfad macht beides |
| c | **Doppelte Skalengruppen** | `NGRP = 8` statt 4 (Kommentar 2013–2017: der `block_q4_0`-Schritt sitzt je 32 Werten) | `S.part` (q·k-Teilsummen je Gruppe) **8 × 12 × 33 Floats = 12.672 B** statt 6.336 B; `S.vs` 1.024 statt 512 B; die Gruppensumme in der Softmax **8 statt 4** LDS-Lesen je Zelle (2270) |
| d | **Die V-Zeile wird auf 64 B gestreckt** | `S.v[4][32][72]` = 9.216 B | obwohl die Quelle nur 36 B hat (2 Blöcke × 18 B): die expandierten int8-Codes belegen dieselbe Fläche wie im int8-Pfad — **44 % der gelesenen Bytes werden wieder zu 100 % LDS** |
| e | **Der Fensterapparat fährt auch ohne Fenster mit** | `tmpw[4]` = 32 VGPRs (2316–2318), der Laufzeit-Zweig `cd.wr[nt] >= 0 ? h8_win(...) : i8_to_h(...)` im Operand-Bau (2244–2246), `__any` (2233) | der Arm `--kv q4u32` **ohne** Fenster nutzt davon nichts — er zahlt aber mit |

**Die LDS-Bilanz** (aus den Arraygrößen berechnet; `TR = 2`, `R = 12`, `WCH = 32`, `VS = 72`):

| Block | int8 | `q4u32` heute | mit (c)+(d) |
|---|---:|---:|---:|
| `part` | 6.336 | 12.672 | 6.336 |
| `p` | 1.584 | 1.584 | 1.584 |
| `vs` | 512 | 1.024 | 1.024 |
| `v` | 9.216 | 9.216 | 5.120 |
| `valid` | 128 | 128 | 128 |
| **Summe** | **17.776 B** | **24.624 B** | **14.192 B** |

Der 4-Bit-Block braucht **39 % mehr LDS** als der int8-Block, und LDS ist laut dem Code selbst die
Occupancy-Grenze (Kommentar 1921–1923: *„LDS, not VGPRs, limits the occupancy here: 5 waves a SIMD"*; der
B3-Vermerk nennt 5 → 3 Blöcke für 24 KiB).

**Die Hebel, mit Erwartung, Preis und Messweg:**

| Hebel | Was | Erwartung | Preis | Messweg |
|---|---|---|---|---|
| **H1** | **Vektor-Laden**: die 4/8 Byte eines Nibble-Paares mit *einer* `uint32`-Ladung holen, die Nibbles im Register auseinandernehmen | ~96 skalare Ladungen → ~12 Vektor-Ladungen je Zelle und Lane; der größte Einzelposten | **keine Zahlenänderung** (dieselben Codes), kein Layoutwechsel | Harness |
| **H2** | **Gepacktes Staging**: die Nibbles gepackt in den shared memory (36 statt 64 B je Zelle) und **einmal** vor dem WMMA entpacken | −4.096 B LDS **und** die halbe Expansion (b) | der Zeilen-Stride (72/80 B, „banks spread") muss neu abgestimmt werden — Bankkonflikte sind messbar, nicht lesbar | Harness |
| **H3** | **Gruppen vor dem Schreiben summieren**: die zwei Gruppen einer Welle im Register summieren, dann `S.part` schreiben | **gemessen (17.11., nur zusammen mit H4 möglich): −0,9 … −2,0 %** zusätzlich zu H4, unter der Schwelle, aber in 6 von 7 Formen mit gleichem Vorzeichen | die fp32-Additionsreihenfolge ändert sich → die Prüfsumme ändert sich; ohne H4 gar nicht möglich (verschiedene Blockskalen je Gruppe) | Harness — **erledigt, nicht eingebaut** |
| **H4** | **Skala in den Operand falten**: `K = (code−8)·step` **vor** dem WMMA in fp16 | **gemessen (17.11.): −4,9 … −7,5 %** auf dem Attention-Kernel (`STRATA_PA_FOLD=1`), H4+H3 −5,8 … −9,4 % | **die K-Werte werden fp16-gerundet**: FP64-Abweichung 1,06e-05 → **6,22e-03**, sechs Fälle rot, `FAILURES: 6`. Deshalb nur als Opt-in | Harness + Qualitäts-Gate — **erledigt, Gate rot** |
| **H5** | **Fenster zellproportional**: die Fensterzellen eines Chunks in eine dichte 16er-Kachel sammeln und nur diese rechnen | ~~der zweite p.v-Satz kostet 6,2–13,4 % der Kernelzeit, weil er je Chunk läuft~~ — **gemessen widerlegt (17.11.):** der zweite Satz läuft in 5 von 65 Chunks (9 von 130 Kacheln), und der Aufpreis bleibt bei halbierter Matrixarbeit gleich: er ist das **divergente Lesen des Seitenpools**, nicht der Satz selbst | ein Gather in die Kachel (Aufwand im Register); H5a ist bitgleich, H5b ändert die Bits | Harness (`window%d timing:`) — **erledigt, negativ** |
| **H6** | **`WIN` als Template-Parameter**: die Q4-Instantiierung in `WIN = true/false` spalten, damit der Arm ohne Fenster weder `tmpw` noch den Laufzeit-Zweig trägt | unbekannt — aber die A.17-Lektion (ein **unbenutztes** 32-VGPR-Array kostete den int8-Kern den Faktor **2,16**) sagt, dass dieser Kern auf Registerdruck sehr empfindlich reagiert | keine Zahlenänderung; zwei Instantiierungen mehr (Compile-Zeit) | Harness + **M1** |
| **H7** | **`SPLIT` aus** (`STRATA_PA_FAST=1`): halbiert die WMMA-Zahl | **gemessen (17.11.): −9,8 %** (q4u32 32768: 6,137 → 5,533 ms), aber FP64-Abweichung 1,06e-05 → **7,18e-03** und `new vs old` 0,0016 der Skala (Grenze 1e-4), `FAILURES: 9` | Genauigkeit (ein FP16-Bild statt hi/lo) | Harness + Qualitäts-Gate — **erledigt, Gate rot** |

**Tote Enden (mit Begründung, nicht als Vorschlag):**

* **Ein Korrektur-Pass** `q·(k_exakt − k_quant)`: nicht nötig — der Fenster-Pfad liest die exakten Zellen
  direkt (A.17), es gibt nichts zu korrigieren.
* **V in fp16 stagen** statt der Codes: 128 B je Zeile statt 64 B, +7 KiB LDS je Block — genau der Grund,
  warum das Fenster einen **zweiten** WMMA-Satz braucht (B3) und nicht „einfach ein anderes Layout".
* **Den Decode-Kern für das Fenster behalten**: das kostet den ganzen Fast-Path (A.1: 256,6 gegen
  634,1 tok/s).
* **Die Query quantisieren**, um einen int8-WMMA zu nutzen: der Query kommt als fp16/bf16; eine int8-Query
  wäre ein eigener Codec mit eigener Qualitätsfrage — nicht in diesem Zweig. Auf gfx12 rechnet der int8-Pfad
  ohnehin über fp16 (`i8_to_h`); **ein int8-WMMA brächte dem 4-Bit-Pfad also nichts**, er entpackt nur
  weniger.

**Der Messplan — alles im Harness, gleiche Selektionen, gleiche Pools:**

* **M1 (zuerst, ohne GPU):** die Ressourcen der Instantiierungen. Der Compiler meldet VGPR/LDS je Kernel
  (`-Rpass-analysis=kernel-resource-usage` beim Übersetzen der `.cu`). Die Frage: wie viele VGPRs hat die
  Q4-Instantiierung, und wie viele Blöcke ergeben sich aus dem LDS? **Ohne diese Zahl ist jede Aussage über
  H6 eine Vermutung** — und die Zahlen oben sind aus den Arraygrößen *gerechnet*, nicht gemessen.
* **M2:** der Referenz-A/B: int8 8,26 gegen `q4u32` 11,33 ms/Chunk (32K/1.024 Queries). Die Streuung zwischen
  Läufen liegt bei ~2–3 % (10,825 / 11,162 / 11,330 / 11,33 gemessen) — **eine Änderung zählt erst ab 5 %**,
  und jeder Lauf braucht **drei** Wiederholungen.
* **M3:** je Hebel ein Build + ein Harness-Lauf (`hip_prompt_attn_wmma.exe 32768 1024 3`): `FAILURES: 0`, die
  Prüfsummen (bitgleich, wo das behauptet wird) und die Zeit. **Der int8-Arm läuft in jedem Lauf mit** — er
  ist der Vergleich, und seine Bits dürfen sich nicht ändern.
* **M4:** H5 zusätzlich über den `window%d timing:`-Zweig (der A/B aus A.18).
* **M5:** was der Harness **nicht** abdeckt: der **4-Bit-Writer** und der **Decode-Kern**. Im Motor ist der
  4-Bit-Decode langsamer als int8 (A.18: 28–31 gegen 33–38 tok/s, mit Vorbehalt der
  Spekulationsannahme) — dort steht dieselbe Nibble-Expansion. Wenn `q4u32` den int8-Decode *ersetzen*
  soll, gehört der Decode-Pfad in dieselbe Analyse.

**M1 ist gemessen (15.11.2026, Compiler-Ressourcenanalyse, gfx1200, `-O3`, kein Spill in den
laufenden Instantiierungen).** Die LDS-Größen sind **exakt die gerechneten** (17.984 / 19.008 /
21.648 B für int8, 24.832 / 25.856 / 30.608 B für `q4u32`, je `TR` 2/1/0) — die Bilanz oben stimmt also.
Die Occupancy aber **nicht so, wie dort vermutet**:

| Instantiierung | `TR` | LDS | VGPR | Wellen/SIMD |
|---|---|---:|---:|---:|
| int8 (`NGRP` 4) | 2 (Default) | 17.984 | 250 | **5** |
| int8 | 0 | 21.648 | 215 | **6** |
| `q4u32` (`NGRP` 8) | 2 (Default) | 24.832 | 240 | **5** |
| `q4u32` | 0 | 30.608 | 235 | **4** |

* **Im Default (`TR = 2`) laufen beide Kerne mit 5 Wellen je SIMD.** Die +39 % LDS kosten den
  4-Bit-Kern **keine** Occupancy — die Grenze ist dort die VGPR-Zahl (240 bzw. 250, beide in derselben
  Stufe), nicht der LDS. Der Vermerk „5 → 3 Blöcke" aus der Planung trifft den Default **nicht**.
* **`TR = 0` ist der einzige Fall, in dem LDS die Occupancy entscheidet:** int8 käme auf 6 Wellen
  (215 VGPR), `q4u32` fällt auf 4 (30.608 B). Diesen Pfad nimmt der Motor nicht.
* **H6 ist damit entwertet, bevor es gebaut ist.** Der Fensterapparat kostet die Q4-Instantiierung
  **keine** messbaren Register: alle zwölf Q4-Varianten (`SPLIT`/`GATE`/`RPF`, je `TR`) liegen bei
  identischen VGPR. Ein `WIN`-Template-Parameter hätte hier nichts zu holen. (Die A.17-Lektion galt dem
  **int8**-Kern, dessen 250 VGPR knapp unter der nächsten Stufe sitzen — dort bleiben 32 zusätzliche VGPR
  tödlich, hier sind 240 mit Reserve.)
* **Damit verschiebt sich die Erwartung.** H3 und H4 (weniger LDS) helfen im Default **nicht** über die
  Occupancy, nur über den LDS-Verkehr. Was bleibt, ist die **Instruktionszahl**: H1 (Vektor-Laden) und H2
  (eine statt zwei Expansionen). Die Occupancy-Stufe selbst kippt erst, wenn der Kern **unter 192 VGPR**
  kommt (6 Wellen) oder **unter 128** (8 Wellen) — das ist die Latte für H1 und H2.

**H1 ist widerlegt, bevor es gebaut ist (15.11.2026, Disassembly des Default-Kerns
`NGRP 8, Q4, SPLIT, TR 2`, gfx1200).** Die Vermutung war: der 4-Bit-Pfad lade die Nibbles als ~96
skalare Bytes je Zelle und Lane, wo int8 12 Vektor-Ladungen macht. Das Disassembly zeigt etwas
anderes:

| Ladung | Anzahl | Was sie ist |
|---|---:|---|
| `global_load_b128 … offset:2` | 2 | die **16 Code-Bytes eines Blocks in einer 16-Byte-Ladung** (der Block beginnt mit der 2-Byte-Skala, die Codes dahinter) |
| `global_load_u16` / `offset:18` | 6 | die **Skalen** (2 Byte je Block, `× 0x90` = 144 Byte je Zelle, `+18` = der nächste Block) |
| `global_load_u16` mit `× 512` davor | 64 | der **Fenster-Pfad** (`HD × 2` = 512 Byte je Zeile) — exakte Halves, kein Bulk |

**Keine einzige `global_load_u8`.** Der Compiler fasst die Byte-Schleife in `q4u_block_codes` bereits zu
einer 16-Byte-Ladung zusammen, und die Skalen sind 2 Byte — kleiner geht nicht. Die 70
`global_load_u16` des Kerns sind Fenster (64) und Skalen (6), nicht der Bulk. **H1 ändert keine
Instruktion** und wird nicht gebaut. (Die Voranalyse hat die *Quelltext*-Schleife gezählt, nicht das,
was der Compiler daraus macht — der Unterschied ist die Lehre.)

**H2 ist widerlegt, bevor es gebaut ist (15.11.2026, dasselbe Disassembly).** Die Vermutung war: die
Nibbles werden **zweimal** expandiert (einmal im Fetch, einmal vor dem WMMA), und gepacktes Staging
spart die erste Expansion plus 4.096 B LDS. Beides trifft nicht zu:

* **Die Expansion passiert genau einmal.** `q4u_block_codes` wird im Kompilat zu einer dichten Sequenz
  `v_and_b32 15` (112×), `v_lshrrev_b16 4` (24×) und `v_add_nc_u16 …, -8` (124×, in 31 Blöcken) — und
  dahinter steht **kein** zweites Entpacken. Die `ds_load_i8` (64×) sind dieselben wie im int8-Kern
  (64×): das ist das Lesen der bereits expandierten Bytes aus dem Shared Memory, das beide Kerne tun.
  Gepacktes Staging würde diese eine Zerlegung nur **verschieben** (vom Fetch an die WMMA-Stelle), nicht
  streichen — und an der WMMA-Stelle ist der Registerdruck am höchsten, nicht am niedrigsten.
* **Die LDS-Ersparnis trägt nicht.** M1: im Default begrenzt die VGPR-Zahl die Occupancy, nicht der LDS
  (beide Kerne 5 Wellen). 4.096 B weniger LDS ändern daran nichts.
* **Der Kern spillt praktisch nicht:** 2 `scratch`-Zugriffe in der SPLIT-Instantiierung, **keine** in der
  FAST-Instantiierung. Die Zerlegung kostet also Register, aber keine Speicher-Rundreise.

Was bleibt, ist die Zerlegung selbst: **~260 Instruktionen**, die der int8-Kern nicht hat (dort 2
`v_and`, 3 `v_lshrrev`). Das ist echte Arbeit je Zelle — aber sie ist **eine** Zerlegung von 4-Bit-Codes,
und die lässt sich nicht vermeiden, solange der WMMA fp16-Operanden will. **H2 wird nicht gebaut.**

> **Korrektur (15.11., CPU-Nachsorge am selben Disassembly).** Die *Zahl* der Zerlegungsinstruktionen ist
> bestätigt (112 `v_and`, 91 `v_lshrrev`), nicht aber ihre Zuordnung: die Operanden dieser `e32`-Instruktionen
> stehen im Literal-Pool, und meine Dekodierung hat die Positionen verwechselt. Der Satz „… an der
> WMMA-Stelle ist der Registerdruck am höchsten" ist damit **unbestätigte statische Analyse**. Der Bestand
> von H2 ruht auf den zwei messbaren Punkten: die LDS-Ersparnis trägt nicht (M1), und der Kern spillt nicht.

**Die Zerlegung in den WMMA ist der eigentliche Hebel, und er ist schon gezogen (15.11.2026,
Disassembly des Byte-Schleifen-Kerns `SPLIT, TR2, NGRP8, Q4`).** Direkt nach der 16-Byte-Ladung der Codes
steht im Kompilat die Wort-EBI-Sequenz: `v_and_b32` mit Maske `0x0F0F0F0F`, `v_lshrrev_b32` um 4,
`v_add_nc_u32` mit `0x78787878` (= gepacktes `−8`), und `pack_i8x4` ist zu `v_or_b32` verschmolzen. Das ist
genau das, was der old-Pfad (`prompt_attn_kernel<4>`) aus expliziten `__vsub4`-Intrinsics bekommt. Der
**Compiler hat die Quelltext-Byte-Schleife also bereits auf die Wort-EBI-Form gebracht** — H1s Schluss
(„kein Handwechsel im C-Code spart eine Instruktion") ist damit bestätigt, und die Meldung „die Voranalyse hat
die Quelltext-Schleife gezählt, nicht das Kompilat" war richtig.

**Ein Mikro-Hebel bleibt, unter der Messschwelle:** das Umsetzungsstadium. 256 Einzelinstruktionen
(126 `v_cvt_f16_f32_e32`, 66 `v_cvt_f16_f32_e64`, 64 `v_cvt_f16_i16`) = 2.048 B = 4,2 % des Kerns. Eine
manuelle Packung halbierte den Kodierungsanteil auf ~2,1 % — **unter der 5-%-Rauschschwelle**, und
`gfx12` kennt für zwei fp16-Halbworte nur die `c_pk`-Formen. Es wird nicht allein gebaut; nur zusammen mit
einem Hebel, der ohnehin baut.

#### C.0.1 — H5a und H7 gemessen (17.11.2026)

**H5a ist gebaut, gemessen und verworfen (17.11.2026) — bitgleich, aber ohne Gewinn.** Vorregistriert
war derselbe Aufruf wie in A.18 (`hip_prompt_attn_wmma.exe 32768 512 3`), der `window%d timing:`-A/B,
drei Wiederholungen je Stand, der int8-Arm in jedem Lauf, und **`NEWHASH` musste unverändert bleiben**
(bei einer bitgleichen Änderung ist das der Beweis).

Gebaut wurde: in `chunk()` ersetzt ein `__ballot(cd.wrow >= 0)` die wellenweite `__any`-Abfrage, und
die zwei 16er-Kacheln des zweiten p.v-Satzes bekommen je ein eigenes Flag — Kachel `ks` liest ihre
Zellen aus den Lanes `ks*16 … ks*16+15`. Eine Kachel ohne Fensterzelle lässt `tmpw` auf seiner
Null-Initialisierung, und der Epilog `tq + tmpw` ist dann `tq + 0.0f`: dieselben Bits, die der
übersprungene WMMA aus Null-Operanden addiert hätte.

* **Die Bitgleichheit ist belegt.** Alle sieben `NEWHASH`-Werte (int8 32768/1500/2100, q4u32 32768,
  q4u32 64, q4u32 512, Fenster 1/2/3) sind vor und nach dem Umbau identisch, ebenso die
  FP64-Abweichungen auf allen gedruckten Stellen (Fenster 1: `7.09e-06`, `new vs old 1.18e-05`);
  `FAILURES: 0` in allen sechs Läufen.
* **Der Gewinn ist null.** Der Fenster-Aufpreis im selben Aufruf (drei Läufe je Stand, in Prozent):

  | Fall | ohne H5a | mit H5a |
  |---|---|---|
  | 1 (6,6 % exakt) | +10,2 / +9,2 / +10,1 | +10,7 / +10,7 / +10,5 |
  | 2 (2,0 % exakt) | +7,1 / +4,2 / +7,4 | +10,0 / +7,9 / +7,6 |
  | 3 (2,9 % exakt) | +10,5 / +11,3 / +10,5 | +10,9 / +10,1 / +11,0 |

  Keine Verbesserung, in Fall 2 eher schlechter: **unter der 5-%-Schwelle, also negativ.** Der Umbau
  ist zurückgenommen; der Pfad bleibt wie im letzten Release.
* **Die Vorhersage stand vor dem Lauf fest und stimmt (statische Analyse aus dem Harness selbst).**
  Die exakten Zellen sind **zusammenhängend**: 8 Sinks am Anfang jeder Selektion (Chunk 0, Kachel 0)
  und 128 Recent-Zellen am Ende (die letzten vier Chunks, beide Kacheln). Bei 32K/512 sind das 5 von
  65 Chunks und **9 von 130** Kachel-Durchläufen; H5a kann davon genau **eine** Kachel überspringen.
  Erwartung war damit 1–2 % *des Aufpreises*, also 0,1–0,2 Prozentpunkte — nicht messbar. Die frühere
  Aussage in A.18 („jeder Chunk hat Fensterzellen") war falsch und ist dort korrigiert.

**Wo der Fenster-Aufpreis wirklich sitzt (gemessen, 17.11.2026).** `SPLIT` aus (`STRATA_PA_FAST=1`)
halbiert **jede** WMMA im Kern, auch die des zweiten p.v-Satzes. Drei Läufe je Einstellung, gemittelt:

| Fall | Aufpreis mit SPLIT | Aufpreis ohne SPLIT | absolut mit / ohne SPLIT |
|---|---|---|---|
| 1 | +10,6 % | +13,3 % | 0,59 / 0,67 ms |
| 2 | +8,5 % | +6,2 % | 0,25 / 0,17 ms |
| 3 | +10,7 % | +12,1 % | 0,23 / 0,24 ms |

Der absolute Aufpreis sinkt in keinem Fall systematisch — im größten Fall (1) steigt er sogar, obwohl
die halbe Matrixarbeit wegfällt. Der Fenster-Aufpreis ist also **nicht** der zweite p.v-Satz, sondern
das **divergente Lesen des Seitenpools** in den Fenster-Chunks: die `h8_win`-Gathers je Lane, das
übersprungene V-Staging, die per-Lane-Auswahl `wr >= 0`. Das ist der Preis der Exaktheit, nicht einer
Doppelrechnung — und damit ist auch **H5b** (dichte Kachel, ändert die Bits) zwecklos, weil es
dieselbe Stelle angreift. Ein Fenster-Hebel bliebe nur das **Layout** des Seitenpools (zusammenhängende,
für die Welle coalescte Zeilen statt zeilenweiser Gather) — ein neuer Hebel, nicht H5.

**Nebenbefund 1 (Messtechnik, wichtig).** Zwischen den beiden Messphasen sind die **unberührten** Arme
— int8 3,10 → 4,60 ms, q4u32 ohne Fenster 4,10 → 6,12 ms — um **~49 %** langsamer geworden: das ist
der Kartenzustand, nicht der Code. Absolute Zeiten sind in dieser Sitzung **nicht** über Builds hinweg
vergleichbar; belastbar ist nur das **A/B-Verhältnis im selben Lauf**. Die ±2–3 % aus M2 gelten für
Wiederholungen in einer Phase, nicht über Phasen.

**Nebenbefund 2 (H7, aus denselben Läufen).** `SPLIT` aus bringt **−9,8 %** (q4u32 32768: 6,137 →
5,533 ms) und −7,7 … −8,8 % in den Fensterfällen, aber das Qualitäts-Gate wird **rot**: die
FP64-Abweichung geht von 1,06e-05 auf **7,18e-03** (~Faktor 680, Ausgabenskala 7,8), `new vs old`
auf **0,0016 der Skala** (Grenze 1e-4), `FAILURES: 9`. Das ist die vorhergesagte fp16-Rundung
(~2⁻¹⁰ nach der Akkumulation über 32 Werte). **H7 ist damit gemessen und abgelehnt**, nicht offen.

#### C.0.2 — H4 und H3 gemessen (17.11.2026; rein experimentell, Default bleibt)

**Vorregistriert:** drei Arme in **einem** Binary, geschaltet über `STRATA_PA_FOLD` — `0` = der Pfad des
letzten Release, `1` = **H4** (die Block-Skala in den K-Operanden gefaltet, fp16-Produkt statt
fp32-Multiplikation der Gruppensummen), `2` = **H4+H3** (zusätzlich ein Akkumulator über die beiden
Gruppen der Welle). Aufruf wie in A.18 (`32768 512 3`), drei Runden, Arme **verschränkt** (0,1,2 / 0,1,2
/ 0,1,2) wegen des Kartenzustands, int8-Arm als Kontrolle, Qualitäts-Gate wie es ist.

**Warum H3 nicht allein geht:** die beiden Gruppen einer Welle sind zwei 4-Bit-Blöcke mit
**verschiedenen** Skalen, und `Σ s_g·ksc_g ≠ (Σ s_g)·ksc`. Erst wenn die Skala im Operanden steckt
(H4), dürfen die Gruppensummen vor der Skalierung addiert werden. H3 ist also der zweite Teil von H4,
kein eigener Hebel.

| Fall | heute (FM 0) | H4 (FM 1) | H4+H3 (FM 2) | H4 allein | H4+H3 | H3 zusätzlich |
|---|---|---|---|---|---|---|
| q4u32 32K/512 | 5,986 ms | 5,816 | 5,721 | **−4,9 %** | **−5,8 %** | −0,9 % |
| q4u32 512 | 1,030 ms | 0,974 | 0,947 | **−7,5 %** | **−9,4 %** | −2,0 % |
| Fenster 1 mit | 6,089 ms | 5,915 | 5,834 | −5,0 % | −5,6 % | −0,6 % |
| Fenster 1 ohne | 5,446 ms | 5,286 | 5,201 | −5,1 % | −5,9 % | −0,9 % |
| Fenster 3 mit | 2,422 ms | 2,314 | 2,265 | −6,5 % | −7,8 % | −1,4 % |
| Fenster 3 ohne | 2,158 ms | 2,056 | 2,003 | −6,8 % | −8,5 % | −1,8 % |

(Die ms-Spalten sind Mittel aus drei Läufen; die Prozentzahlen sind **gegen den int8-Kontrollarm
desselben Laufs normiert**, weil der zwischen den Phasen driftet — Kontrolle 4,513 / 4,613 / 4,579 ms,
also ±2 % Rauschen. Die gepaarten Runden für 32K geben −4,3 / −4,6 / −5,9 % (H4) und −5,0 / −6,1 /
−6,3 % (H4+H3).)

**Was daran zählt:**

* **Der Hebel ist ~5 % groß** (H4), **H3 legt ~1–2 % drauf** — und das ist unter der 5-%-Schwelle,
  aber in sechs von sieben Formen mit gleichem Vorzeichen, also nicht Nichts. Zusammen **−6 %** auf
  dem Attention-Kernel. Das ist die Größe, die der LDS-/Skalen-Posten überhaupt hat.
* **Der Preis steht dagegen:** die FP64-Abweichung geht von **1,06e-05 auf 6,22e-03** (Faktor ~590, das
  sind 8e-4 der Ausgabenskala), `new vs old` auf **0,0012 der Skala** (Grenze 1e-4) — **sechs** Fälle
  werden rot (alle q4u32), `FAILURES: 6`. Es ist genau die vorhergesagte fp16-Rundung des Produkts
  (~2⁻¹¹ je Wert, über 64 Summanden).
* **Der int8-Zweig ist unberührt:** die drei int8-`NEWHASH` sind in allen neun Läufen identisch
  (`232d4781a2b59ccc`, `9f901d5231c4013e`, `d791e4bfb2fd064d`) — der `Q4`-Wächter sitzt richtig, die
  verbotenen Prüfsummen-Änderungen treten nicht auf.
* **Der Fensteraufpreis bleibt in allen Armen ~+12 %** (Fenster 1: +11,4…12,3 % / +11,7…12,0 % /
  +12,3…12,5 %). H4/H3 fassen weder das Fetch noch den zweiten p.v-Satz an — konsistent mit C.0.1.

**Entscheidung:** nicht eingebaut, aber als **Opt-in** (`STRATA_PA_FOLD=1|2`, Default `0`) im Baum
gelassen, damit die Zahl nachvollziehbar bleibt. Der Default-Pfad ist bitgleich zum Release (FM 0 ist
die alte Anweisungsfolge), der int8-Zweig ohnehin. **Was es kostet:** drei Instantiierungen je
Trim/RPF-Variante (6 → 18 für die 4-Bit-Route), also Compile-Zeit — wenn das stört, ist es ein
Revert ohne Verhaltensänderung. **Nicht gebaut:** der CUDA-Zweig (kein `nvcc` auf diesem System); die
Änderung ist reiner Template-Code im gemeinsamen Body.

**Die Reihenfolge nach der Messung (17.11., neu):** ~~H1~~ → ~~H2~~ → ~~H6~~ → ~~H5a~~ (bitgleich,
ohne Gewinn) → ~~H7~~ (Gate rot) → ~~H4/H3~~ (−5…−6 %, Gate rot, als Opt-in behalten) → der
**cvt-Mikrohebel** nur zusammen mit einem Bau — sonst ist die Hebel-Liste abgearbeitet. **Motor-Serie
~~A.20~~/~~A.21~~ gefahren:** `q4_0` 5–7 % unter `q4u32` (kein Prompt-Kern), `int8` und `q4u32` gleichauf,
`q4u32` ohne den Schalter **2,5× langsamer** — damit steht die Default-Entscheidung an. M2: 3
Wiederholungen, ≥5 % zählt, und seit dem 17.11. zusätzlich: **nur das A/B-Verhältnis im selben Lauf**
ist über Builds hinweg belastbar (der Kartenzustand verschob die Absolutzeiten um ~49 %).

**Die Reihenfolge nach Erwartung (vor M1, überholt):** M1 → **H1** (kein Zahlenrisiko) → **H2** → **H3**
(Qualitäts-Gate) → **H6** (nach M1 entscheiden) → **H5** → **H4**/**H7** nur mit Qualitätsentscheidung.

**Was die Optimierung *nicht* lösen kann — der ehrliche Schluss.** Im Motor sind die Arme **gleichauf**
(A.19), und die Attention ist ~1 % des Prompt-Chunks; selbst eine perfekte Attention-Kernel-Optimierung
bewegt die Prompt-Zeit um Bruchteile eines Prozent. **Der Wert der Optimierung ist der Attention-Anteil bei
langen Kontexten**: der Anteil wächst linear mit der Auswahllänge (quadratisch in der Attention gegen
linear im MLP), also von ~1 % bei 32K auf ~4 % bei 128K — und der 37-%-Rückstand des 4-Bit-Kerns wäre
dann ~4 % des Prompts. Das ist die Rechnung, die vor dem Bau eines Hebels da sein muss.

### C.1 — Die ältere Kandidatenliste (14.11., vor B2; durch C.0 überholt)

Erst messen, dann optimieren. Die Kandidaten, alle aus dem Code gelesen (statische Analyse), mit der
jeweiligen Erwartung:

1. **Bandbreite.** Der 4-Bit-Bulk liest die Hälfte der Bytes; der Prompt wird bandbreitenbegrenzt.
   Das ist der Grund, warum G-R2 „nicht langsamer" verlangt und nicht „langsamer".
2. **Skalengruppen.** 8 Gruppen à 32 Dims bedeuten doppelt so viele fp32-Skalen-Multiplikationen und
   Skalen-Reads je Zelle wie 4 à 64. Ansätze: die Skala in bf16 anwenden, oder je Zelle einmal in ein
   Register falten.
3. **Shared Memory.** Die Expansion 4 → 8 Bit verdoppelt den Schreibtraffic in den shared memory
   (32 B je 32 Dims). Alternative: die Nibbles gepackt lassen und erst vor dem WMMA entpacken → halbe
   Fläche, mehr Entpack-Instruktionen.
4. **Layout.** Der Zugriff auf die 16 Codebytes je Block ist nach `kv_q4.hpp:73` genau das Muster, für
   das `__constant__` am schlechtesten ist — der Global-Zugriff im Prompt-Pfad ist der Ort, an dem das
   zählt.

**Regel:** keine Optimierung geht ohne eine **gemessene** Verbesserung auf dieser Maschine (A1 gegen
A2, gleicher Harness) und ohne das Paritätsgate. Der Default-Pfad bleibt byte-identisch, solange
keine Messung das Gegenteil sagt.

---

## Teil D — Was *nicht* in diesem Plan ist

Die restlichen offenen Serien aus dem Recap bleiben offen und werden hier **nicht** angefasst:

* **#2 Attribution** (Schrittregel gegen Fenster auf dem uniformen Gitter): die Arme „q4_0 + Fenster"
  und „uniform LS ohne Fenster" fehlen weiterhin.
* **#3 128K-Matrix**: `q4u32+C`, `q4u32z`, `q4u32z+C` fehlen bei 128K.
* **#4 6-Bit-Nullpunkt**, **#5 End-to-end-Gate des Shears** (das ist bereits gescheitert:
  `q4u32z − q4u32` = +0,00138, Ganztext, t 1,64), **#6 seitenweise Geometrie**, **#7 Gruppengröße 32**.

Ein Hinweis für später: **#6 wird billiger**, wenn B2 steht — eine eigene Geometrie je Seite braucht
eigene Leser, und die gibt es dann an zwei Stellen.

---

## Teil E — Die Regeln (unverändert, aber hier bindend)

* Ein GPU-Job zur Zeit. 30 s zwischen zwei Engine-Starts. **Kein Build, während ein Lauf die Binärdatei
  benutzt.**
* Vorregistrierung vor dem Lauf; die Gatter stehen oben, bevor die erste Zahl entsteht.
* Rohdaten außerhalb des Repos; im Repo nur das README, der Treiber und die Kernels.
* Nur selbst erzeugte Messungen; Literaturzahlen nur mit Quelle und als die Zahlen des zitierten
  Papiers.
* Keine Verallgemeinerung ohne Messung. Was aus dem Code gelesen ist, steht als **statische Analyse**
  im Text, nicht als Ergebnis.
* Commits mit Area-Präfix: `[Engine]` für die Kernelarbeit, `[Benchmark]` für die Läufe, `[Docs]` für
  Paper und READMEs.
* Der Default-Pfad bleibt byte-identisch, solange keine Messung ihn rechtfertigt.


---

## Teil F - Der Rebase auf Upstream 0.2.15 (die Nachprüfung, 17.11.2026)

**Was gemacht wurde:** der q4u32-Zweig wurde auf den Upstream-Stand 0.2.15 gehoben (Merge, kein
Rebase: 268 Commits, auf die die Papiere und dieses Protokoll verweisen). Der PR-Branch
`pr/q4u32-default` steht auf `upstream/main` und trägt dieselbe Arbeit in sechs lesbaren Commits.

**Die Sorge, die der Merge weckt:** ein Merge kann die Arithmetik verändern, ohne dass es auffällt.
Upstream hat den Prompt-Kern nicht angefasst (`qsa_prompt_attn.cu`, `kv_q4.hpp`,
`qsa_prompt_attn_parity.cpp` und `gfx12_platform_defaults.hpp` sind zwischen der Basis und 0.2.15
byte-identisch), aber 132 Upstream-Commits liegen in `generate.cpp` — und die Konflikte lagen in
genau den Dateien, die den Default und die Version setzen.

**Das Gate (gelaufen, 17.11.2026, derselbe Befehl wie oben):**
`build-hip\hip_prompt_attn_wmma.exe 32768 512 3` (HIP, ROCm 7.2, gfx1200, frisch aus dem
Upstream-Stand gebaut):

* **FAILURES: 0**, Exit 0;
* **alle neun NEWHASH identisch** zu den Werten von oben: `232d4781a2b59ccc`, `2455dbed6eba900f`,
  `9f901d5231c4013e`, `d791e4bfb2fd064d`, `82ef1dc3fa5dd4e7`, `6c01e0a92922cca5`, `ad466135a4d13e70`,
  `06c917251ed1d5cf`, `1d3b59ff22ef3ea5`.

**Was das beweist:** der Rebase hat keine Arithmetik angefasst. Der PR-Branch trägt exakt den
gemessenen Kernel; die Messungen A.16–A.27 gelten für ihn unverändert.

**Was der Merge anfaßte (und was nicht):** die Konflikte lagen in `CMakeLists.txt` (die
Versionszeile), `engine/src/program/generate.cpp` (unser KV-Default), `setup.py`, `terminal/strata_cli/launch/`
(das Terminal), `docs/CAPABILITIES.md`, `AGENTS.md` und `sycl/.../qsa_decode_attn.dp.cpp`. Die
Engine-Version ist nach dem Heben `0.1.63-q4u32` (die Basis 0.1.63 plus das Suffix); die
Goldens des Terminal-Tests folgen auf 0.1.63. Upstreams 0.2.15 hat DeepSeek stillgelegt — das
wurde **nicht** zurückgeholt: der `MULTI_FILE_FAMILIES`-Zweig und der `_dsv41`-Treiber im
Test-Harness sind raus, die Modell-Prüfung wieder auf die Upstream-Form.
