# A/B/C/D — Re-Evaluation nach dem Upstream-Merge: Kontrollexperiment, Re-Baseline, Speed, Docs

**Status: vorregistriert (2026-11-10); die Stufen A–C sind gelaufen, D läuft.** Kein Lauf dieser
Kampagne durfte vor diesem Dokument starten; Stufe A startet erst danach. Die Regeln je Stufe
standen vor dem Lauf und werden nicht nachträglich ergänzt — Abweichungen werden unter
„Abweichungen und Grenzen“ offen genannt, nicht stillschweigend geändert.

**Wo die Ergebnisse stehen:** das **Laufprotokoll** (außerhalb des Repos:
`C:\Users\krone\git\Strata-data\bench\2026-11-10-merge-reeval\protocol.md`) mit allen Zahlen,
allen Gattern und allen Abweichungen 1–11; die **Ergebnisse in den Dokumenten**: das Paper
`docs/paper/KV-mean-centered-calibrated-quantization.tex` (v0.10), das Fenster-Blatt
`docs/paper/KV-Window-Hardening.tex` (v1.11), die Leiter `docs/paper/Ansatz_Improvements.md`
(A2, A6, K1) und das Miniprojekt-Dokument `bench/results/2026-11-10-centering/README.md`
(C0–C4, D1). **Kurzfassung der Befunde:** der `int8`-Pfad ist byte-identisch; die cwe-4k-Zahlen
des Vorgängers gelten auf dem Merge unverändert; der **Produktcodec ist 16/16 + Fenster ohne
Kalibrierung** (auf `int8`-Niveau: +0,00002 nats, t = −0,02); das Centering ist ein **Ersatz** für
den Level-Split, kein Zusatz; die Zeitmessungen der Kampagne sind zurückgezogen und durch eine
saubere Nachmessung ersetzt.

Katalogzeile: `docs/paper/Ansatz_Improvements.md` §1/§7 — die „Anker (heute)“ der Leiter (K1–K3)
und jede Zahl in §1 sind Messungen des Port-Builds **0.1.47**. Diese Kampagne setzt sie auf den
gemergten Build neu; sie öffnet keinen Hebel der Leiter und legt keinen neuen Arm an. B1 erfüllt
nebenbei Katalogzeile E2 (`q4_0`/`k8v4` auf dem cwe-Harness, P2). Anschluss: der letzte Lauf des
alten Builds (`bench/results/2026-11-03-tag-arm/`; der Kampagnen-Record ist nicht in diesem PR enthalten, Engine-SHA `CD5700EC0793F83B…`) und die
Post-Merge-Validierung (Merge-Commit `19a13e369`, HIP-Build `build-hip-port`, 386 Ziele grün,
alle KV/Paritätstests grün).

## Warum

1. **Der Engine-Stand hat gewechselt.** Alle publizierten Zahlen der KV-Arme — cwe-4k-KL 0,02090,
   5,29 bit/cell, 1.016 MiB bei 128K, Prefill 528 gegen 526 t/s, 568 MiB frei, +416 Slots,
   Break-even ≈ 27.600 Zellen — stammen vom Port-Build 0.1.47. Der gemergte Baum ist 0.1.59.1 mit
   2.603 Upstream-Commits über den QSA-/Selection-/Prefill-Flächen (Merge `19a13e369` aus
   upstream/main `87dfca0e7`, Nachzüge bis `60f0785d9`). Ob die Zahlen übertragbar sind, ist die
   Messung selbst, keine Annahme.
2. **Der Byteidentitäts-Claim ist veraltet.** `docs/DETAILS.md` sagt „default path byte-identical
   to the last release (dump-logits SHA `A014F504…`)“ — „last release“ ist inzwischen 0.1.59, nicht
   der Stand, gegen den gemessen wurde. Der Claim ist neu zu begründen, und zwar gegen **pures
   upstream/main**, nicht gegen 0.1.47.
3. **Upstream-Defaults haben sich bewegt.** Statisch aus dem Merge-Diff: `qsa_kv_resident_min()`
   fiel auf nicht-Metal von 20.480 auf 4.096 Zellen (`engine/src/core/layer.cpp:881–892`; Env
   `STRATA_KV_RESIDENT_MIN` überschreibt; Metal behält 20.480 mit eigener Messbasis im Kommentar);
   der Prefill-Ring wurde umgebaut (Chunk-Deckel, „ring sized for the prompt chunk“);
   `STRATA_SELECT_TWO_PASS` ist ein CUDA-sm_80+-Pfad und auf HIP immer aus
   (`engine/include/strata/kernels/qsa_select.hpp:106`, `engine/src/prefill/prefill.cpp:8162`). Keine dieser
   Änderungen darf still als „ohne Effekt“ gelten.
4. **Alle künftigen Läufe laufen auf dem neuen Build.** Die Leiter §7 rechnet gegen „heute“-Anker
   (K1: +0,00402, t = −3,55, `int8` führt in 7/8 Segmenten). Ohne Re-Baseline würden künftige
   Prüfungen gegen 0.1.47-Anker gemessen.

## Was der gemergte Stand ist

| | gemergter Build (Branch) | Upstream-Referenz (nur Stufe A) |
|---|---|---|
| Quelle | HEAD `60f0785d9` (Merge `19a13e369` aus upstream/main `87dfca0e7`) | Worktree von `87dfca0e7` |
| Version | 0.2.13.1 / build 0.1.59.1 (CLI-Smoke) | reines upstream 0.2.13 / 0.1.59 |
| Build-Ordner | `build-hip-port` (besteht; 386 Ziele, `kv_win_parity` inklusive) | `strata-upstream-1059\build-hip` (A0, neu) |
| Toolchain | vcvars64 (VS 2022 BuildTools), Ninja, ROCm 7.2 `clang++.exe`, `CMAKE_HIP_ARCHITECTURES=gfx1200` | identisch |
| Configure | Release, `STRATA_GEOMETRY=flash_next`, `STRATA_AMD_COMMUNITY_TUNING=ON`, `STRATA_NATIVE_EXPERTS=ON`, `STRATA_Q6K_EXPERTS=ON`, `STRATA_MMQ_KQUANTS=ON`, `STRATA_PREFILL_MMQ=ON`, `STRATA_MMQ_TQ_OVERLAY=ON`, HIP an, CUDA/SYCL aus, Tests aus (Cache-Stand von `build-hip-port`) | dieselben Einträge |
| Karte | RX 9060 XT (gfx1200), Windows 11, ROCm 7.2 | dieselbe |

## Stufe A — Kontrollexperiment: der gemergte Build gegen pures upstream/main

**Frage:** Ist der Default-Pfad (`--kv int8`, kein Fenster) des gemergten Builds vom selben Stand
pures upstream/main unterscheidbar? Nur wenn **nein**, ist jede Arme-Differenz in Stufe B
garantiert KV-Format-Sache und der Byteidentitäts-Claim neu tragbar.

### Aufbau

- **A0 — Upstream-Build.** Worktree von `87dfca0e7` (keine Submodule im Tree, `third_party/` ist
  getrackt) und Build mit denselben Cache-Einträgen wie `build-hip-port` (Tabelle oben). Einmalig,
  ~30–60 min, vcvars64-Umgebung.
- **A1 — zwei Configs, ein Arm.** Zwei Kopien von `bench/configs/cwe4k-swift-iq3_xxs.json`,
  identisch außer `exe` und Port; `lib_dirs` beide `7.2\bin` (siehe Grenze 2); `env` bleibt leer
  (kein `STRATA_KV_WINDOW` — der Kommentar der Vorlage gilt weiter). Je Engine genau ein Lauf:
  `--kv int8`, keine Fenster-Umgebung, `--ids data\quant_quality\cwe-4k\set.ids.json`
  (32.746 Positionen, dieselben wie alle Paper-Tab.-6-Läufe).
- **A2 — Vergleich.** `RUN.tsv` beider Läufe byte-weise: SHA-256 und Zeilenzahl; bei
  Abweichung: erste abweichende Position, Anzahl, Muster (alle/wenige, Segment-Enden, nur lange
  Segmente). Die `seg.json` beider Läufe muss dieselbe Positionsmenge zeigen.

### Entscheidungsregel (vor dem Lauf festgelegt)

- **Identisch** (Hash gleich, je 32.746 Datenzeilen): Stufe B startet. Der Claim wird in Stufe D
  formuliert als „Default-Pfad identisch zu purem upstream/main `87dfca0e7` auf demselben
  Build-Setup (RUN.tsv byte-identisch, Datum, beide Engine-SHAs)“.
- **Verschieden:** Stufe B startet **nicht**, bevor die Quelle benannt ist. Kandidaten (statisch
  aus dem Merge-Diff, geordnet nach Eingriffstiefe): (i) `qsa_kv_resident_min` 20.480 → 4.096 —
  greift nur bei gestreamtem KV, der Yardstick hält den KV resident, also nur relevant, falls ein
  Lauf doch streamt (am stderr prüfbar); (ii) der Prefill-Ring-Umbau (Chunk-Deckel, Ring-Zeilen);
  (iii) Selektions-/`old_sel`-Pfade. Das Ergebnis benennt die Ursache und ersetzt keine Regel.

## Stufe B — Re-Baseline der Arme auf dem gemergten Build

**Frage:** Wo stehen die Arme auf 0.1.59.1 — und wie viel von `int8`s Abstand zur fp16-Decke ist
überhaupt Quantisierungsbudget?

### B1 — cwe-4k, sechs Arme

Werkzeug-Standard überall (`--prefill` nicht gesetzt = 512, die alten Läufe reproduzierende
Einstellung; top-64-KL, die Anker-Definition der K1–K3):

| Arm | `--kv` | Fenster-Umgebung | Zweck |
|---|---|---|---|
| `fp16` | `fp16` | — | die Decke: KL(fp16‖int8) = das Budget |
| `int8` | `int8` | — | die Referenz (K1-Basis) |
| `q4_0` | `q4_0` | — | flacher Anker (Katalog E2: der llama.cpp-Anker auf demselben Harness) |
| `k8v4` | `k8v4` | — | flacher Anker (Katalog E2) |
| `plain` | `q4lm8v` | — | der Branch-Codec ohne Fenster |
| `win` | `q4lm8v` | `STRATA_KV_WINDOW=1 S=256 R=64 BATCH=512` | das Fenster in der Paper-Tab.-6-Geometrie (256/64/512) |

### B2 — Nadeln

Dasselbe Protokoll wie `bench/results/2026-10-24-needle-window/`: je 6 Nadeln bei 32K und 128K,
`int8` gegen das Fenster 256/64, Kriterium die gepflanzten Wörter. Regel unverändert: **6/6 und
6/6**, Antworten = gepflanzte Wörter.

### B3 — cwe-128k, die Kernarme

Dieselben zwei 128K-Passagen wie Paper Tab. 7/8 (Ids wie die alten Läufe; liegen diese nicht mehr
vor, wie in `2026-11-03-tag-arm/` aus cwe-8k konkatenieren und die Konstruktion dort dokumentiert
lassen). Je Passage vier Arme: `fp16`, `int8`, `plain`, `win` (`--prefill auto` → B=6.144,
`--short-read 6144`, `--topk 256`, `--pin`/`--prompt-cache 3` für die Schnittkante wie tag-arm,
Abweichung 1 dort). Auswertung `tools/kv_k1.py --block 256` (Moving-Block-Bootstrap, fester Seed)
und die K2/K3-Statistiken: Ganzsegment, p99, letzte 1.024, **je Text**. Setzung: top-256 wie die
letzten 128K-Läufe (e1/tag-arm); S0 (2026-10-25-s0-kl-support) hat die Rangfolgestabilität zwischen
top-64 und top-256 gezeigt, die neuen Anker stehen damit unter der 256er-Definition.

### Fragen und Entscheidungsregeln (vor dem Lauf festgelegt)

- **`B1a` — das Budget (deskriptiv):** KL(fp16‖int8) auf cwe-4k, gepaart je Segment. Die Zahl wird
  der neue Decken-Anker; sie wird nicht gegen eine Schwelle geprüft.
- **`B1b` — steht die Rangfolge?** Erwartung nach 0.1.47: `int8` vorn vor dem Fenster vor `plain`
  auf der gepaarten Mean-KL (t = −3,55). Dreht sich das Vorzeichen auf einer robusten Statistik,
  ist das ein eigener Berichtspunkt unter dem Ergebnis — die Leiter-Annahmen („Fenster schließt
  34–58 % der Tail-Mean-KL“) wären neu zu lesen. Kein Kill, kein Start einer neuen Bau-Arbeit aus
  dieser Kampagne heraus.
- **`B1c` — Nadeln:** 6/6 und 6/6 wie 2026-10-24; ein Ausfall ist ein Bug-Report, kein
  Qualitätsergebnis.
- **`B3` — K2/K3-Lesung je Text (deskriptiv):** Ganzsegment, p99 und letzte 1.024, je Text,
  gepaart gegen `int8`. Die Ergebnisse ersetzen die Ankerzeilen in §7 (K2/K3, „Anker (heute)“).

### Two-Pass-Matrix — nicht durchführbar auf dieser Karte

Der mündlich vereinbarte vierte Punkt (2×2: `STRATA_SELECT_TWO_PASS` an/aus × {`int8`, Fenster})
ist auf HIP **nicht messbar** (statisch belegt): `STRATA_SELECT_TWO_PASS` ist ein CUDA-sm_80+-Pfad
des Prompt-Scorers; `qsa_two_pass_on()` ist auf HIP immer `false`, und ein gesetztes Flag meldet
das einmal laut (`engine/include/strata/kernels/qsa_select.hpp:106`;
`engine/src/prefill/prefill.cpp:8162`: „HIP: a set STRATA_SELECT_TWO_PASS says once it is not used“).
Ersatz: in **einem** B1-Arm der Laufzeitumgebung `STRATA_SELECT_TWO_PASS=1` mitgeben und die
„not used“-Zeile im stderr als Beleg einsammeln (Nullprobe, dass der Scorer im ganzen Vergleich
aus war). Die Frage „schrumpft der Fenster-Nutzen unter besserer Selektion“ bleibt offen und ist
nur auf CUDA sm_80+ messbar — als Grenze notiert, nicht als Ergebnis.

## Stufe C — Speed und Speicher

**Frage:** stehen die Produktzahlen des Fensters auf 0.1.59.1 noch? Zu prüfende Anker (alle 0.1.47,
Paper Tab. 4 / `docs/DETAILS.md`):

| Anker (0.1.47) | Neuprüfung |
|---|---|
| Fenster 256/64, B=6.144: **5,29 bit/cell**, **1.016 MiB** bei 128K | MiB aus dem Engine-Report; bit/cell ist Rechnung (b4-Modell `4,50 + 16·(R+S+B)/L`, `2026-11-01-b4-byte-model`) |
| `int8`: **8,25 bit/cell**, **1.584 MiB** | dito |
| **−36 % Speicher**, 568 MiB frei, **+416 Slots** | freier Speicher/Slots aus dem Report desselben Laufs |
| Prefill **528 gegen 526 t/s** (Fenster gegen `int8`, +0,4 %) | prompt tok/s |
| Decode in Streubreite (−1,0 t/s unter `--spec 4`) | decode tok/s |
| Break-even ≈ **27.600 Zellen** | Rechnung; nur falls sich die Geometrie ändert |

Aufbau: `int8` gegen das Fenster 256/64 **am selben Tag auf demselben Build**
(docs/COMMUNITY_BENCHMARKS.md-Regel: kein Vergleich über Tage), 128K-Kontext wie Paper Tab. 4
(B=6.144, `--spec 4`), `bench/configs/int8-bench.json` als Vorlage (dort `--max-context 32768` →
für 128K eine eigene Kopie). Zusätzlich je Arm ein `tools/strata_bench.py`-Lauf (4K/16K-Prompt,
Agent-Szenario mit Cache-Reuse, `--long` für den 32K-Punkt) als Streubreiten-Kontrolle.

## Stufe D — Docs und Paper fortschreiben

1. `docs/DETAILS.md` (KV-Quant-Sektion): neue Zahlen mit vollständiger Provenanz — „gemergter
   Build 0.1.59.1 (Commits bis `60f0785d9`), Windows HIP, RX 9060 XT (gfx1200), ROCm 7.2, gemessen
   <Datum>, Rohdaten in `bench/results/2026-11-10-merge-reeval/`“.
2. Byteidentitäts-Claim umformulieren: von „gegen den Vorrelease (dump-logits SHA `A014F504…`)“ zu
   „identisch zu purem upstream/main `87dfca0e7`“ — mit Stufe-A-Ergebnis, Datum und beiden
   Engine-SHAs. Nur falls Stufe A „identisch“ sagt; sonst ehrlich die Abweichung und ihre Quelle.
3. `docs/paper/Ansatz_Improvements.md`: §1 und die „Anker (heute)“-Spalten der Leiter §7 (K1–K3)
   auf die neuen Zahlen setzen; die 0.1.47-Zahlen bleiben als Historie erkennbar
   (bench/results-Konvention: jeder Ordner beschreibt seinen eigenen Tag).
4. Die Paper-Quellen (`.tex`) nur mit einer Versionshistorie-Zeile erweitern (§0-Regel „Sichtbare
   Versionierung“); keine stillen Zahlenersetzungen.

## Nicht Teil dieser Messung, aber vor einem PR

1. Debug-`fprintf` in `engine/src/core/layer.cpp:1435` (die `QSA_LAYER:`-Zeile) entfernen oder hinter
   ein Env-Flag — vor jedem Review.
2. CUDA- und SYCL-Builds: auf dieser Maschine keine Toolchain — im PR als „not run“ deklarieren
   (HIP: 386 Ziele grün, belegt).
3. `hip_handoff`, `vmm_test`, `conversation_copy_streams_test`: als „unconfirmed —
   environment-suspected“ führen (upstream-identische Tests, Treiber-Verdacht).

## Güte-Gatter (je Lauf, vor der Auswertung)

- Engine-SHA-256 der `strata.exe` vor jedem Lauf notieren (gemergter Build; in Stufe A auch der
   Upstream-Build). Branch-HEAD (`60f0785d9`) und Worktree-Commit (`87dfca0e7`) stehen im
   Protokoll.
- Jeder Arm `DONE` ohne „nonfinite“, kein „do not fit“, kein `ERR`.
- cwe-4k: **32.746 Positionen**, dieselbe Positionsmenge über alle Arme (`seg.json`); die
  `env`-Umgebung in der `seg.json` registriert (kein stilles Fenster).
- Fenster-Arme: die Ring-Zeile im stderr („ring sized …“) sichtbar, mit B=512 (B1) bzw. B=6.144
  (B3).
- Stufe A: zusätzlich byte-Identität beider `RUN.tsv` und gleiche `seg.json`-Positionsmengen.
- Ein GPU-Job zur Zeit; 30 s zwischen zwei Engine-Starts; Rohdaten unter
  `C:\Users\krone\git\Strata-data\bench\2026-11-10-merge-reeval\`.

## Abweichungen und Grenzen, offen genannt

1. **Two-Pass-Matrix nicht durchführbar** (CUDA sm_80+; siehe Stufe B). Ersatz: Nullprobe. Die
   Frage bleibt auf CUDA-Hardware offen.
2. **DLL-Version:** die Vorlage-Config führt `lib_dirs 7.1\bin`; die neuen Configs setzen
   `7.2\bin` für **beide** Engines (beide Builds sind mit 7.2 kompiliert; ein 7.2-Binary mit
   7.1-DLLs ist ungetestet und würde den A-Vergleich riskieren). Absolute Differenzen zur
   0.1.47-Baseline sind damit nicht nur Engine-Stand — deshalb Re-Baseline statt Diff-gegen-alt.
3. **cwe-128k ist ein wiederholter Korpus** (Wiederholung gegen Länge,
   `2026-11-03-repeat-control`): dieselbe Grenze, die schon die Paper-Tab.-7-Läufe hatte; die
   gepaarte Lesung bleibt gültig, das absolute 128K-Urteil nicht.
4. **4K bleibt Diagnosepunkt**; der Einsatzbereich ist 32K–256K (tag-arm-Fazit). B3 misst 128K;
   32K/64K nur nach Bedarf nach dem deploy-range-Protokoll.
5. **`q4lm32` (16/16) und der Tag-Arm sind nicht Teil dieser Kampagne** (T-Serie geschlossen
   2026-11-04). Falls ein `q4lm32`-Arm re-baselinet werden soll, ist das eine eigene Registrierung.
6. **Keine CUDA/SYCL-Aussage** aus dieser Kampagne (HIP only).
7. Der Results-Index (`tools/bench_results_index.py --check`) war vor diesem Ordner schon rot (die
   committete Indexseite endet bei 2026-10-20; die Forschungsordner danach fehlen drin). Dieser
   Ordner verschlimmert das nicht und repariert es auch nicht.

## Kosten

| Stufe | Aufwand |
|---|---|
| A0 Upstream-Build | ~30–60 min, einmalig |
| A1 + A2 | 2 cwe-4k-Läufe (~1–2 h je Arm, bisherige Erfahrung) + Minuten Auswertung |
| B1 | 6 Arme cwe-4k (~1–2 h je Arm) |
| B2 | Nadeln 32K/128K (~1 h) |
| B3 | 4 Arme × 2 Texte à ~5 h ≈ 2 GPU-Tage |
| C | halber Tag (same-day-Paare) |
| D | offline, ein Abend |

## Reproduktion

```powershell
# A0 (einmalig): Worktree von reinem upstream/main und Build mit den build-hip-port-Cache-Einträgen
git worktree add C:\Users\krone\git\strata-upstream-1059 87dfca0e7
cmd /c "call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cmake -S C:\Users\krone\git\strata-upstream-1059 -B C:\Users\krone\git\strata-upstream-1059\build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="C:/Program Files/AMD/ROCm/7.2/bin/clang++.exe" -DCMAKE_HIP_ARCHITECTURES=gfx1200 -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_SYCL=OFF -DSTRATA_GEOMETRY=flash_next -DSTRATA_AMD_COMMUNITY_TUNING=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_TQ_OVERLAY=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DSV41=OFF -DSTRATA_BUILD_SPLIT=OFF -DSTRATA_BUILD_CONVERSATION_TESTS=OFF && cmake --build C:\Users\krone\git\strata-upstream-1059\build-hip -j 12"

# A1: zwei Config-Kopien von bench\configs\cwe4k-swift-iq3_xxs.json (cwe4k-swift-iq3_xxs-merged.json
#     mit exe build-hip-port\strata.exe, cwe4k-swift-iq3_xxs-upstream.json mit exe des A0-Builds;
#     sonst identisch zur Vorlage, nur lib_dirs 7.2\bin und eigene Ports; env bleibt leer)
python tools\quant_quality.py run --config bench\configs\cwe4k-swift-iq3_xxs-merged.json `
    --ids data\quant_quality\cwe-4k\set.ids.json --kv int8 `
    --out C:\Users\krone\git\Strata-data\bench\2026-11-10-merge-reeval\a1\merged-int8.tsv
python tools\quant_quality.py run --config bench\configs\cwe4k-swift-iq3_xxs-upstream.json `
    --ids data\quant_quality\cwe-4k\set.ids.json --kv int8 `
    --out C:\Users\krone\git\Strata-data\bench\2026-11-10-merge-reeval\a1\upstream-int8.tsv

# A2: byte-Identität (beide Dateien 32.746 Datenzeilen)
(Get-FileHash ...\a1\merged-int8.tsv).Hash -eq (Get-FileHash ...\a1\upstream-int8.tsv).Hash
#   bei Abweichung: erste abweichende Zeile suchen, Anzahl und Positionsmuster zählen

# B1: sechs Arme, Werkzeug-Standard (top-64, prefill 512); ein GPU-Job zur Zeit
python tools\quant_quality.py run --config bench\configs\cwe4k-swift-iq3_xxs-merged.json `
    --ids data\quant_quality\cwe-4k\set.ids.json --kv <fp16|int8|q4_0|k8v4|q4lm8v> `
    --out C:\Users\krone\git\Strata-data\bench\2026-11-10-merge-reeval\b1\<arm>.tsv
#   Fenster-Arm (statt --kv-Passage oben):
python tools\quant_quality.py run --config bench\configs\cwe4k-swift-iq3_xxs-merged.json `
    --ids data\quant_quality\cwe-4k\set.ids.json --kv q4lm8v `
    --env STRATA_KV_WINDOW=1 --env STRATA_KV_WINDOW_S=256 --env STRATA_KV_WINDOW_R=64 `
    --env STRATA_KV_WINDOW_BATCH=512 `
    --out C:\Users\krone\git\Strata-data\bench\2026-11-10-merge-reeval\b1\q4lm8v-win.tsv

# B1-Auswertung (REF = fp16; gepaarte Segment-Statistik Arm - int8 wie Paper Tab. 6, das
# Auswerteskript wird mit dem Lauf im Ergebnisabschnitt dokumentiert)
python tools\quant_quality.py compare ...\b1\fp16.tsv ...\b1\int8.tsv ...\b1\q4_0.tsv `
    ...\b1\k8v4.tsv ...\b1\q4lm8v.tsv ...\b1\q4lm8v-win.tsv
python tools\quant_quality.py areport data\quant_quality\cwe-4k\set.ids.json ...\b1\fp16.tsv `
    ...\b1\int8.tsv ...\b1\q4lm8v.tsv ...\b1\q4lm8v-win.tsv

# B2: Nadeln wie 2026-10-24-needle-window (Protokoll und Kriterien dort)

# B3: 128K, je Text und Arm wie 2026-11-03-tag-arm T3 (--prefill auto, --short-read 6144,
#     --topk 256, --pin und --prompt-cache 3 wie dort dokumentiert)
python tools\kv_k1.py --ref ...\b3\t1-fp16.tsv --run ...\b3\t1-int8.tsv --run ...\b3\t1-plain.tsv `
    --run ...\b3\t1-win.tsv --kl-topk 256 --block 256 --json ...\b3\t1-k1.json
```

Rohdaten (TSVs, stderr, `seg.json`) liegen außerhalb des Repos unter
`C:\Users\krone\git\Strata-data\bench\2026-11-10-merge-reeval\`; im Repo steht dieses Dokument
und nach den Läufen der Ergebnisabschnitt darunter.

## Ergebnisse (Stand 2026-11-11)

**Woher das hier steht:** das vollständige Laufprotokoll mit **allen** Zahlen, den **Abweichungen 1–14**
und den offenen Entscheidungen liegt außerhalb des Repos unter
`C:\Users\krone\git\Strata-data\bench\2026-11-10-merge-reeval\protocol.md` (184 KB). Die
**Ergebnisabschnitte** des Miniprojekts (C0–C4, D1) stehen **im Repo** unter
[bench/results/2026-11-10-centering/README.md](../2026-11-10-centering/README.md); die abgeleiteten
Vergleichstabellen des 128K-Laufs (alle sechs Arme in **einem** Durchlauf) liegen **im Repo** unter
[d1/](d1/) (`compare-all6.json`, `paired-all6.json`).

**Stufe A (Kontrollexperiment):** der gemergte Build ist für `int8` **byte-identisch** zum
pures-upstream-Build (SHA `BC6FBB1A…`, 32.746 Zeilen). Der Merge hat die `int8`-Qualität nicht
bewegt.

**Stufe B (Re-Baseline):** B1 (sechs Qualitätsarme, cwe-4k) und B2 (sechs Nadeln) sind durch, alle
Wirksamkeitsgatter grün (SHA, 12/12 Parität, 6/6 + 6/6 Nadeln). Die Ring- und Nullproben sind
protokolliert.

**Stufe C (Laufzeit, 32K-Nadel, freie Maschine, gleiche Geometrie):**

| Arm | Prompt-Lesedauer | Faktor |
|---|---:|---:|
| `int8` | 51,7 s | 1,00× |
| `q4_0` (uniform, affin) | 50,0 s | 0,97× |
| `q4lm8v` (Lloyd-Tabelle) | **116,4 s** | **2,25×** |
| `q4lm8v` + Kalibrierung | 118,5 s | 2,29× |
| `q4lm8v` + Fenster | 116,5 s | 2,25× |

**Der 2,25×-Preis ist ein Kernel-Struktur-Preis, kein Matrix-Core-Preis** (der Prompt-Schnellpfad
lehnt die nicht-affine Level-Tabelle ab; `STRATA_HIP_WMMA=1` ändert auf dieser Karte die
Prompt-Raten nicht). Die Aufklärung und die beiden Kandidaten stehen in
[docs/research/KV_FASTPATH_LITERATURE.md](../../docs/research/KV_FASTPATH_LITERATURE.md) und in der
Leiter (Stufe S6).

**Stufe D (das Produkt):** der Produktcodec ist **`q4lm32w` (16 Stufen auf beiden Seiten, mit
Fenster) OHNE Kalibrierung**.

| Arm (cwe-4k) | KL | Top-1 | ΔKL vs `int8` | t |
|---|---:|---:|---:|---:|
| `int8` | 0,01688 | 99,05 % | — | — |
| `q4_0` (Stock) | 0,02530 | 98,71 % | +0,00843 | −3,99 |
| `q4lm8v` + Fenster (16/8) | 0,02090 | 98,84 % | +0,00402 | −3,55 |
| `q4lm8v` + Fenster + Kalibrierung | 0,01871 | 98,87 % | +0,00184 | −1,85 |
| **`q4lm32w` (16/16)** | **0,01689** | 98,89 % | **+0,00002** | **−0,02** |

**Bei 128K (Text `s00`, 131.060 Positionen) hält der Gleichstand nicht:** der 16/16-Arm liegt
+0,00741 nats hinter `int8` (t = +4,80), der p99 2,0583 gegen 1,6615 → **K2 und K3 fallen**, und
das 5,0-Bit-Ziel ist auf diesem Korpus **nicht erreicht** (Grund: der Bulk-Codec im Langkontext,
nicht das Fenster). Der Fat-Tail ist real (Exzess-Kurtosis 211 bei `int8`) und gehört **dem Text**,
nicht dem Codec (Rangkorrelation 0,848 zwischen `int8` und `q4lm32w`).

**Die Kalibrierung: drei Messungen, das Vorzeichen nicht etabliert.** Sie ist ein **Ersatz-Hebel**,
kein Produktpfad:

| Codec | 4K | 128K |
|---|---:|---:|
| `q4lm8v` + Fenster (16/8, der ausgelieferte) | +0,00218 (n.s.) | **−0,00045** (n.s.) |
| `q4lm32w` (16/16, der Produktcodec) | −0,00079 (n.s.) | **+0,00314** (t = +2,05, marginal) |

**Die Merge-Gegenprobe ist exakt:** der unkalibrierte 16/8-Fensterarm misst bei 128K **0,07524** —
auf fünf Nachkommastellen die Zahl des Vorgängers für denselben Arm. Der Merge ist neutral ohne
Restabweichung; die 0,6 %, die zunächst als Merge-Abweichung gelesen waren, sind die
Kalibrierungs-Wirkung (+0,00045, nicht signifikant).

**Offen und im Protokoll benannt:** der zweite 128K-Text (zurückgestellt), der **End-to-End-Arm für
den gescherten Codec** (der affine Preis ist im Blockmaß gemessen, die nats-Zahl fehlt), **D2**
(Indexer-Auswahl, ergebnisoffen registriert) und der Fast-Path (die drei Preise sind gemessen: der
Tabelleneintrag im Operandenbau, das uniforme Gitter ohne und mit per-Block-Nullpunkt). Die
Block-Kernaussage der Kampagne ist **bestätigt** (der Bias kostet 2,01 dB Block-MSE; die früheren
0,17/0,36 dB waren ein Artefakt eines trunkierten Zellsatzes).

