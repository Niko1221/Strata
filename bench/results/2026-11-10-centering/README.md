# Miniprojekt „Centering": die systematischen Momente vor der Rotation entfernen

**Status: vorregistriert (2026-11-10). C0 bis C4 sind gelaufen; die Ergebnisse stehen unten.**
Dieses Dokument stand vor dem ersten Lauf; die Gatter standen fest. Anlass ist die Recherche in
`docs/research/KV_ROTATION_SURVEY.md` und die Auswertung der S2-Rohdaten aus
`bench/results/2026-10-27-s2-channels/` (Rohdaten: `Strata-data/bench/2026-10-27-s2-channels.txt`; der
Kampagnen-Record selbst ist nicht in diesem PR enthalten).
Katalogzeile: `docs/paper/Ansatz_Improvements.md`, **A6** (Per-Kanal-Offset vor der Rotation); die
Rotations-Variante **A7** (randomisierter Hadamard H·D) läuft im selben Harness mit, ist aber
**nicht** Gegenstand dieses Projekts.

**Das Prinzip des Projekts:** *Ein Bias ist bei randomisierten Verfahren immer problematisch — eine
Zufallsrotation entfernt einen systematischen Anteil nicht, sie verteilt ihn nur auf zufällige
Koordinaten.* Für einen gegebenen Sign-Zug S ist S·μ ein **fester** Vektor mit fester Energie
(‖H·S·μ‖ = ‖μ‖, orthogonal): der Bias landet nur zufällig, er verschwindet nicht. Genau das ist
Flagrots Beobachtung („rotation flattens a spiky piece of a key but can give a flat one a new
peak"). **Darum ist die Reihenfolge fest: erst den systematischen Anteil entfernen, dann über
Randomisierung reden** — vorher ist eine Randomisierung nicht sinnvoll messbar. Dieses Projekt
entfernt den systematischen Anteil **konsequent**, also alle Momente, die systematisch sind, nicht
nur den ersten.

## 1. Warum

**Der Befund.** Die S2-Sonde akkumuliert je Kanal unter anderem Mittel, Varianz, Schiefe und
Exzess-Kurtosis der K-Werte über 32.746 Token (12 QSA-Layer × 256 Kanäle). Ausgewertet am
2026-11-10, mit den Standardfehlern der Moment-Schätzer für diese Stichprobe als Rauschmaß:

| Moment | vor FWHT | nach FWHT | Rauschen | systematisch? |
|---|---|---|---|---|
| Mittel (1. Moment) | Median \|mean\|/sd **0,2673**, p90 0,7337, max 3,5778 | **0,2825**, p90 0,6967 | 0,0055 | **ja, und die Rotation entfernt ihn nicht** |
| Schiefe (3. Moment) | Median 0,1050, p90 0,3531, max 1,7199 | 0,0836, p90 0,2255 | 0,0135 | **ja, kanalspezifisch** (62,6 % der Kanäle > 5× Rauschen vor, 58,5 % nach) |
| Exzess-Kurtosis (4. Moment) | Median 0,2220, p90 0,8452, max 6,7842 | 0,1295, p90 0,4373 | 0,0271 | ja, aber die Rotation flacht es stark ab (65,0 % → 48,1 % über 5× Rauschen) |

Drei Lesarten:

1. **Der Mittelwert ist echt** (rund 60-mal über dem Rauschen) **und die Rotation verteilt ihn nur.**
   Aus einem festen Offset je Kanal wird ein fester Bias in *jedem* rotierten Koordinatenpaar
   (dieselben Offsets addieren sich mit den Hadamard-Vorzeichen). Er ist **kein gemeinsamer DC** —
   der Mittelwert der Kanalmittelwerte je Layer liegt bei −0,016 bis +0,086 bei einer Streuung der
   Kanalmittel von 0,43 bis 0,64, also \|mean\|/sd_typ = 0,007 bis 0,068: die Offsets sind
   **kanalspezifisch**.
2. **Die Schiefe ist ebenfalls systematisch und kanalspezifisch** — aber sie ist **nicht**
   layerweit: der Mittelwert der Schiefe je Layer liegt nur bei 0,2 bis 2,0× Rauschen, mit 46 bis
   56 % gleichgerichteten Kanälen. Es gibt also keinen „Layer-Schief", den man mit einem einzigen
   Versatz räumt; jeder Kanal trägt seine eigene. Die Rotation nimmt sie nur leicht zurück
   (0,1050 → 0,0836).
3. **Die Kurtosis ist der Teil, den die Rotation schon erledigt** (0,2220 → 0,1295). Sie ist der
   Grund, warum die Rotation überhaupt im Plan steht.

**Warum das bei uns teuer ist.** Unsere Skala ist eine **Max-Statistik**: `d = amax / lmax`, mit
`amax` als Shuffle-Maximum über die 32 Werte eines Blocks (`engine/src/kernels/cuda/kv_q4.cu:138-144`;
`lmax` = 87 für 16 Stufen, 69 für 8 Stufen). Ein fester Bias bläht dieses Maximum auf — und die
Skala wird an **alle** Werte des Blocks bezahlt, nicht nur an den verrauschten.

**Was das für die Spannungsnutzung heißt.** Die Zahl der effektiv genutzten Stufen je Block ist
das Signal, an dem das Centering zu messen ist. Für einen Block mit 32 Werten und Stufentabelle L
gilt: die Stufenweite ist 2·d, das Signal nutzt die Spanne seiner eigenen Streuung. **Definition
des Werkzeugs:** die *Spannungsnutzung* u eines Blocks = (Streuung des Signals im Block) / d, in
Stufen. Ohne Bias ist u der Quotient aus Signalstreuung und Blockmaximum; mit Bias wächst das
Maximum, d wächst, u sinkt. **Die Hypothese: das Centering hebt u messbar (die Skala wird enger),
und die MSE folgt daraus.**

**Größenordnung (Schätzung, keine Messung).** Nach der Rotation: sd ≈ 1,1 bis 1,67, Bias
|mean_post| ≈ 0,283 · sd; Signalmaximum über 32 Koordinaten ≈ 2,1 · sd, Biasmaximum über denselben
Block ≈ 0,6 · sd → die Skala ist um grob **15 bis 30 %** zu groß, der MSE skaliert mit d² → **etwa
+1 bis +2 dB**. Zum Vergleich: der Rest der Bulk-Seite bis zum konvergierten per-Block-Fit ist
**0,17 dB** (`kv_q4_parity`: q4_0 2,938e-02 = 0 dB, kLloyd 2,740e-02 = −0,30 dB, kLloydLs
2,551e-02 = −0,61 dB, konvergierter Fit 2,456e-02 = −0,78 dB).

**Die Messung (C1) bestätigt die Schätzung:** der gemessene Gewinn ist **−1,9 dB** beim Anker und
**−2,6 dB** beim `q4lm8v`-Codec — in der Größenordnung der Schätzung und **größer** als die
Level-Tabelle (0,30 dB) und die per-Block-Skala (0,31 dB) zusammen.

**Was die Literatur sagt (Belege in `docs/research/KV_ROTATION_SURVEY.md`).** OptR (arXiv:2608.02691)
zentriert die Keys vor der Rotation; ein öffentlicher `llama.cpp`-Fork hat ein K-Cache-Mean-Centering
mit **Calibration-Basis-Guard** (ein Versatz aus der falschen Basis verschlechtert messbar).
NSNQuant (arXiv:2505.18231) zentriert kalibrierungsfrei über eine laufende Schätzung.

## 2. Die Korrektur, so einfach wie möglich

Die Korrektur ist **eine Subtraktion vor dem FWHT** — kein neues Cache-Format, kein neues Feld im
Block, kein zusätzliches Byte im Cache, **kein Umbau des Lesers**:

1. **K:** `k' = k − m_K` je Kanal, **vor** dem FWHT. Der Writer bekommt eine Offset-Tabelle
   `m_K` (12 Layer × 2 KV-Köpfe × 256 Kanäle = 6.144 Werte, 12 KB in fp16).
2. **V:** `v' = v − m_V` je Kanal, vor dem FWHT. Der Leser muss `m_V` **einmal** auf die
   Aufmerksamkeits-Ausgabe addieren (ein Vektoradd pro Token und Kopf): mit
   `Σ aᵢ = 1` ist das exakt: `Σ aᵢ (vᵢ − m_V) = (Σ aᵢ vᵢ) − m_V`.
3. **Die Scores bleiben unangetastet:** `⟨q, k⟩ = ⟨q, k′ + m_K⟩ = ⟨q, k′⟩ + ⟨q, m_K⟩` — der zweite
   Teil ist **pro Query konstant** (der Query ist pro Token fix) und verschiebt alle Logits dieser
   Query gleich, der Softmax bleibt exakt. Also: **kein Add-Back auf dem Score-Pfad.**
4. **Reihenfolge auf dem Ausgabepfad:** erst un-rotieren (H), **dann** `m_V` addieren — im
   rotierten Raum wäre der Add-Back `H·m_V`. Eine vertauschte Reihenfolge bricht das Modell
   still; das ist dieselbe Klasse wie die Inverse-Falle beim randomisierten Hadamard (A7).
5. **Das Fenster bleibt exakt:** der Ring speichert den fp16-Zwilling der (zentrierten, rotierten)
   Zellen — dieselben Werte, die der Writer schreibt. Die Genauigkeit des Rings ändert sich nicht.
6. **Die Selektion sieht nichts:** der Indexer-Key ist die Projektion von `x`
   (`engine/src/core/layer.cpp:1399-1407`), **vor** dem FWHT ab 1472.

**Was die Korrektur NICHT braucht:** keine Änderung am Blockformat (18 Byte bleiben 18 Byte), keine
Änderung am Leser außer dem einen Vektoradd, keine Änderung an der Level-Tabelle, keine Änderung
am Ring, keine Änderung an der Streaming-Granularität.

**Was sie braucht (und das ist der Preis):** eine Offset-Tabelle pro Modell — **12 Layer × 256
Kanäle = 3.072 Werte (6 KB in fp16)**, *nicht* 6.144: die S2-Sonde mittelt über die beiden
KV-Köpfe (`accumulate(..., rows = T*2, 256)`), sie kann also **keine** kopfgetrennten Offsets messen.
Per-Kopf-Offsets (6.144 Werte) bräuchten eine Erweiterung der Sonde; die geteilte Tabelle ist
das, was die Messung trägt, und sie ist der ehrlichere Weg (ein Versatz, der pro Kopf
verschieden wäre, ist mit dieser Sonde nicht sichtbar). Dazu die Frage, woher die Tabelle kommt:
- **(a) Kalibriert und ausgeliefert:** ein Kalibrierlauf über einen Text (analog zum
  Expert-Profil), die Tabelle ins Pack. Robust, aber ein Artefakt im Modell.
- **(b) Online geschätzt (kalibrierungsfrei):** ein laufender Mittelwert je Kanal (EMA) über die
  ersten Zellen; die Zellen vor der Schätzung werden ohne Versatz quantisiert (die ältesten, also
  die Sinks — der Bereich, den die Auswahl ohnehin am wenigsten liest). Kein Artefakt, aber
  komplexer (ein Schwellen- und ein Nachlaufverhalten, das der Plan festlegen muss).
- **(c) Aus dem Modellgewicht abgeleitet:** die K-Kanäle sind `w_norm · x̃`; falls der Offset aus
  einem nicht-null Erwartungswert der normalisierten Aktivierungen stammt, wäre er über
  `E[x̃]` und den Norm-Gewichtsvektor bestimmbar — **ohne** Kalibrierung. Ob das trägt, ist offen
  (der Offset ist in den Daten real, die Kette Normgewicht → Aktivierungs-Erwartung → K-Mittel ist
  nicht gemessen). Wenn (c) trägt, ist das die einfachste Korrektur überhaupt: eine Tabelle, die
  aus dem Gewicht abgeleitet wird.

**Die Entscheidung zwischen (a), (b) und (c) fällt in Stufe C0** — sie hängt an der Stabilität des
Offsets über Texte und Längen.

**Und die anderen systematischen Momente (das „konsequent").** Das Centering entfernt den ersten
Moment. Die **Schiefe** (3. Moment) ist ebenfalls systematisch und kanalspezifisch (Median 0,1050
vor und 0,0836 nach der Rotation; 62,6 % der Kanäle über 5× Rauschen) — eine **symmetrische**
Lloyd-Tabelle kann eine schiefe Quelle nicht ausnutzen: die Stufen werden auf die leere Seite
verteilt. Sie zu entfernen geht nur **kanalweise** (es gibt keinen Layer-Schieft, s. §1), also mit
derselben Tabelle: pro Kanal ein Versatz **und** ein Asymmetrie-Parameter, oder eine eigene
Stufentabelle je Kanal (12 KB je Layer und Kopf — dieselbe Währung wie die Offsets). Die
**Exzess-Kurtosis** (4. Moment) ist der Teil, den die Rotation schon flacht (0,2220 → 0,1295); sie
zu entfernen hieße eine andere Codebook-Familie (Katalogzeile A5), nicht eine Korrektur im Writer.
**Die Reihenfolge, die dieses Projekt daraus macht:** (1) der 1. Moment (Versatz) — billig, exakt
erhaltend, erwartet der größte Hebel; (2) der 3. Moment (Asymmetrie) — **nur wenn** der
Parity-Harness zeigt, dass danach noch Stufen verschenkt werden; (3) der 4. Moment — gehört in
eine andere Zeile (A5). Die Messung entscheidet die Reihenfolge: u (die Spannungsnutzung) ist das
Werkzeug, und wenn u nach dem Versatz noch weit unter ihrem Maximum liegt, ist der nächste
Kandidat die Asymmetrie.

## 3. Der Basis-Vertrag (der teuerste Teil)

Ein Versatz ist **basisgebunden**: gemessen im unrotierten Kanalraum, angewendet vor dem FWHT.
Der `llama.cpp`-Guard existiert genau deshalb. Für uns heißt das:

- Der Offset wird **im unrotierten Kanalraum** gemessen (das ist die Ebene, in der die Tabelle
  lebt) und **vor** dem FWHT abgezogen.
- **Snapshot:** `conversation_snapshot.cpp` verweigert heute windowed States („exact side pool
  outside the image"). Ein zentrierter Zustand gehört in dieselbe Klasse: das Snapshot-Format
  braucht ein **eigens Kennzeichen** (analog zum vorhandenen `rotated`-Flag), und ein Snapshot eines
  zentrierten Zustands darf **nicht** in einen unzentrierten zurückgespielt werden (sonst fehlt
  `m_K`). Das ist ein Schema-Flag, kein Formatumbau.
- **Streaming:** die Host-Kopie trägt dieselben zentrierten Werte; der `m_V`-Add-Back ist
  pfadunabhängig (er hängt an der Attention, nicht am Speicherort).
- **Der `int8`-Pfad bleibt byte-identisch:** das Centering ist opt-in (eigener Schalter, kein
  Default-Pfad).

## 4. Die Stufen und ihre Gatter (vor dem Lauf festgelegt)

| Stufe | Was | Gatter (Kill) | Kosten |
|---|---|---|---|
| **C0** | **Offset-Stabilität und Moment-Status — zwei Läufe auf dem aktuellen Build.** **C0a:** cwe-4k (derselbe Text und dieselbe Länge wie die Messung vom 2026-10-27) → die Referenz **desselben** Builds. **C0b:** cwe-8k (zweiter Text, zweite Länge, 65.571 Ids) → der Gate-Vergleich. Der Vergleich **C0a ↔ C0b** ist G1; der Vergleich der 0.1.47-Datei mit C0a isoliert zusätzlich den **Build-Effekt** auf die Offsets (deskriptiv, kein Kill). Beide Läufe über den batched Prompt-Pfad (`--short-read 0`), weil der Verify-Pfad graph-capture-geschützt ist. Im selben Lauf die **Signifikanz aller drei Momente** berichten (Mittel, Schiefe, Kurtosis gegen die Standardfehler der Schätzer: 0,0055 / 0,0135 / 0,0271 bei n = 32.746), damit die Entfernungsreihenfolge (§2) auf Messdaten steht und nicht auf einer Vermutung | **G1 (C0a ↔ C0b):** Korrelation der standardisierten Kanalmittelwerte ≥ 0,90 **und** mittlere relative Abweichung ≤ 0,25 sd. Darunter: (a) wird unbrauchbar, es bliebe (b) oder (c) — und (b) ist erst nach C1 zu bauen. **G1b:** die Schiefe und die Kurtosis werden auf dem zweiten Text mit ihrer Signifikanz berichtet (Deskriptiv, kein Kill: sie entscheiden die Reihenfolge in §2, nicht das Projekt) | 2 Läufe, ~3,5 min (4K) + ~7 min (8K) GPU (S2-Rohzeit laut `2026-10-27-s2-channels`: „ein Lauf, 3,5 min GPU-Zeit, kein Kernel geändert") |
| **C1** | **Der Effekt, ohne Engine-Änderung — auf den echten Zellen.** Der Parity-Harness zieht **synthetische Gauss-Zellen** (`nd(rng)*2.0f`); dort gibt es keinen Kanal-Offset, die Centering-Frage ist also dort **nicht** messbar. **Deshalb zwei Schritte:** (C1a) ein Lauf mit `STRATA_KV_Q4_DUMP=<datei>` (FROM/TO auf die ersten 8192 Positionen geklemmt, ~400 MB) → die **echten** rotierten K/V-Zellen des batched Promptpfads, im selben Lauf mit der S2-Sonde (Momente und Zellen aus einem Durchgang über eine Passage); (C1b) die Auswertung auf dem Host (`tools/kv_codec_mse.py`): MSE und Spannungsnutzung u je Block, Variante für Variante. **Die Varianten in fester Reihenfolge:** (1) uniform+max (der Anker `q4_0`), (2) uniform+max + Centering, (3) uniform+LS (der A8-Kandidat, **affin**), (4) uniform+LS + Centering, (5) lloyd16+max, (6) lloyd16+LS, (7) lloyd16K+lloyd8V (der q4lm8v-Codec) — jeweils mit und ohne Centering; (8) H·D + Centering nur zum Protokoll (A7) | **G2 erfüllt** (MSE-Gewinn **−1,9 dB** beim Anker, **−2,6 dB** beim q4lm8v-Codec — Faktor 2,8 bzw. 1,8 im MSE). **G3 nicht erfüllt** (+0,35 %), weil das registrierte Maß σ/Schritt **skaleninvariant** ist und ein Bias-Entfernen per Definition nicht zeigen kann; der Bericht trägt stattdessen \|d\|/σ (die Skala relativ zur Streuung des Blocks — die Größe, die „besser den Wertebereich ausnutzen" meint). **A7 mit Messung geschlossen**: drei H·D-Züge innerhalb 0,01 dB, und auf den zentrierten Daten ist der RHT (−0,18 dB) *schlechter* als das uniforme Gitter mit der gefitteten Skala (−0,20 dB) | C1a: 1 Lauf, ~4 min GPU (4K, batched Promptpfad, ein GPU-Job); C1b: Host, Minuten (keine GPU) |
| **C2** | **Die Korrektur, gebaut.** Writer-Änderung (Subtraktion vor dem FWHT), `m_V`-Add-Back auf dem Ausgabepfad, Offset-Tabelle im Weg der Wahl (a)/(b)/(c), Snapshot-Kennzeichen, opt-in-Schalter | **G4:** `int8`-Pfad byte-identisch (SHA-Gatter wie in der Kampagne), `kv_q4_parity` und `kv_win_parity` grün, `kv_window_parity` unverändert | **M** (Writer + Leser + Pack) |
| **C3** | **Die Wirkung.** Ein cwe-4k-Arm (gepaart gegen `int8` und gegen den Window-Arm) | **G5:** die gepaarte KL verbessert sich; 128K nur bei einem 4K-Unterschied | **M** (ein Arm) |
| **C4** | **Der Produktcodec: 16 Stufen auf beiden Seiten + Fenster + Centering.** Der `q4lm8v`-Arm benutzt 8 Stufen auf V — ein 3-Bit-Codebook in 4-Bit-Bytes, **ohne Byte-Gewinn** (aus dem Parity-Harness mit Centering: lloyd16K+lloyd8V+LS 6,527e-03 gegen lloyd16+LS 5,072e-03 = **+1,10 dB für dasselbe Bytebudget**). Der Produktcodec ist `--kv q4lm32w` (16/16, mit Fenster). **Zwei Arme auf cwe-4k, dieselben Ids, derselbe Build:** (a) `q4lm32w` **ohne** Centering, (b) `q4lm32w` **mit** `STRATA_KV_CENTER` — (b) gegen (a) isoliert das Centering, (a) gegen den `q4lm8v`-Fensterarm isoliert den Level-Split | **G6 (K1 der Leiter §7), auf dem 16/16-Arm:** gepaarte Mean-KL (Arm − `int8`) ≤ 0 **oder** obere 95-%-Grenze < +0,001 nats, und `int8` führt in höchstens 4 von 8 Segmenten. **Erwartung (keine Messung):** MSE ≈ 5,1e-03 (das Niveau des 16/16-Arms aus dem Parity-Harness) → KL ≈ 0,0170–0,0175 → die Differenz zu `int8` (0,01688) läge bei +0,0001 bis +0,0006. **Was den Arm killt:** wenn der 16/16-Arm **schlechter** ist als der 8-Stufen-V-Arm (dann wäre die Level-Tabelle nicht der Verursacher, und der C3-Befund wäre falsch interpretiert) oder wenn die gepaarte Differenz nicht kleiner wird als +0,00184 | **M** (zwei Arme, je ~18 min GPU: 32.746 Zeilen, 8 Segmente) |

**Reihenfolge:** C0 → C1 → C2 → C3. C0 und C1 sind billig und entscheiden, ob C2 überhaupt
gebaut wird. **C0 ist der eigentliche Kill**: ein Offset, der mit dem Text wandert, ist kein
Offset, sondern Rauschen — und dann hilft auch die schönste Tabelle nichts.

### Stufe D1 (vorregistriert, 2026-11-10): 128K **out of sample**

**Die Frage, die C3 offen lässt.** Die Offset-Tabelle ist auf **cwe-4k** kalibriert (ein Text, eine
Länge). C3 misst auf **denselben** Ids — das ist ein **In-Sample**-Test: er zeigt, dass der Offset
auf dem Kalibriertext hilft, **nicht**, dass er auf fremden Texten hilft. **D1 ist der einzige Test,
der die Kalibrierungs-Artefakt-Hypothese prüft**, und er ist billiger als er aussieht, weil die
Offsets unverändert bleiben (der Test misst die Übertragbarkeit, nicht die Tabelle).

**Material:** die Offsets aus cwe-4k (`center-table.bin`), die Texte aus den 128K-Draws
(`data/quant_quality/cwe-128k/t1.ids.json` und `t2.ids.json` — zwei Texte, jeder 128K; die Dateien
liegen **außerhalb** des Tracked-Bereichs, weil sie 1,8 MB je Text sind). **Arme:** `fp16` (die
Referenz, auf die die absolute KL gemessen wird — und zugleich die Gegenprobe zum Vorgänger, dessen
128K-Zahlen 0,10715 (T1) und 0,06435 (T2) sind), `int8` (der K1-Anker), der Produktarm `q4lm32w`
**ohne** Kalibrierung und derselbe Arm **mit** `STRATA_KV_CENTER` — der Paar-Arm gegen den
Paar-Arm isoliert die Übertragbarkeit des Offsets, der Arm gegen `int8` die K2/K3-Bedingungen der
Leiter (Ganzsegment-KL ≤ `int8` auf **beiden** Texten, p99 ≤ `int8` auf beiden Texten und beiden
Statistiken).

**Geometrie:** die Produktionsgeometrie bei 128K (`STRATA_KV_WINDOW_S=256`, `_R=64`,
`_BATCH=6144`, `--prefill 6144`) und der `STRATA_VERIFY_ASYNC_LAUNCH=1`-Schalter, den jeder
Kampagnen-Arm braucht (ohne ihn stirbt der Verify-Pfad an der #267-Klasse).

**Gatter (vor dem Lauf festgelegt):**
- **G7 (die Übertragbarkeit):** der Arm **mit** Centering muss auf **beiden** Texten besser sein als
  derselbe Arm **ohne** Centering (gepaartes ΔKL < 0 je Text). Fällt der Vorteil auf einem Text ins
  Negative, ist der Offset ein Artefakt des Kalibriertexts und **nicht** als Produktpfad zu
  führen.
- **G8 (K2/K3):** Ganzsegment-KL ≤ `int8` auf beiden Texten **und** p99 ≤ `int8` auf beiden Texten.
- **Der 128K-Lauf startet nur bei einem 4K-Unterschied** (Leiter §7, S4). Fällt C4 auf 4K durch,
  entfällt D1 — und das wird im Protokoll so festgehalten.

**Kosten:** zwei 128K-Läufe je Arm (Prefill + Scoring der 128K-Positionen, ~70 min je Arm bei
131.061 Positionen) — der teuerste Test des Projekts. Deshalb die Vorab-Registrierung und der
4K-Trigger. **Gestuft:** erst Text 1 (vier Arme), Text 2 nur, wenn Text 1 die Frage nicht bereits
beantwortet.

**Nachtrag 2026-11-11 (vor dem Lauf):** die Frage hat sich durch C4 verschoben. Das Centering ist
**kein Produktpfad** mehr (auf dem 16/16-Codec bringt es end-to-end nichts, siehe C4), der
Produktarm ist `q4lm32w` **ohne** Kalibrierung. D1 prüft deshalb (a) ob der 16/16-Arm sein
`int8`-Niveau **bei 128K** hält (K2/K3 — die eigentliche Produktfrage), und (b) ob das Centering
auf den Texten, für die es **nicht** kalibriert wurde, irgendetwas bringt — denn **das p99** ist
das einzige Kriterium, das dem kalibrierten Arm nützt (0,2263 gegen 0,2410 auf 4K, dort aber
**nicht** gepaart signifikant: McNemar p = 0,18). Genau deshalb läuft der kalibrierte Arm mit und
wird mit denselben gepaarten Werkzeugen ausgewertet (`tools/kv_tail_top1.py`: der Tail über eine
**arm-unabhängige** Positionsmenge, der Top-1 mit exaktem McNemar).

### Zusatzarm (2026-11-11, 12:50, auf Wunsch des Nutzers): `q4lm8v` + Fenster + Centering bei 128K

**Begründung:** das ist die **ausgelieferte** Konfiguration (der Vorgänger-Produktarm) **mit** der
Kalibrierung — und damit der einzige Arm, an dem die Kalibrierung im Langkontext überhaupt noch
etwas entscheiden kann. Drei Gründe:
1. **Die Merge-Gegenprobe** für den ausgelieferten Arm (der Vorgänger berichtet für seinen
   Fenster-Arm auf demselben Text T2: Ganzsegment 0,07524, p99 0,0210, Tail-Mittel 0,00437) — und
   die „T2" ist, wie der `gold.json` zeigt, unser `s00`.
2. **Die Kalibrierung im Langkontext**: auf 4K war ihre gepaarte Wirkung auf diesen Arm
   **+0,00218** (nicht signifikant, Intervall [−0,00022, +0,00459]). Ob sie bei 128K trägt, ist
   damit offen.
3. **Der direkte Vergleich mit dem 16/16-Arm im Langkontext** — auf 4K ist der Unterschied zwischen
   „16/8 + Fenster + Kalibrierung" (0,01871) und „16/16 + Fenster" (0,01689) **nicht signifikant**
   (t = +1,73, Intervall [−0,00052, +0,00416]). Bei 128K ist der 16/16-Arm hinter `int8`
   (+0,00741) — die Frage, ob der kalibrierte 16/8-Arm dort *besser* ist als der 16/16, ist damit
   offen und messbar.

**Gatter (vor dem Lauf):** der Arm muss (a) 131.060 Positionen und 8/8 Segmente liefern, (b) eine
**andere** SHA als der unkalibrierte 16/16-Arm haben (Wirksamkeitsgatter) und (c) im gepaarten
Vergleich gegen `int8` ausgewertet werden (K2/K3, Tail, Top-1). **Die Erwartung ist eine
Rechnung, keine Hoffnung:** die Wirkung der Kalibrierung ist **pro Zelle** kontextunabhängig, also
sollte der kalibrierte 16/8-Arm bei 128K rund 0,0018 nats unter dem 16/16-Arm landen (0,0736 gegen
0,07176) und damit **weiterhin hinter `int8`** (0,06435). Ein Ergebnis *über* dem 16/16-Arm wäre
die erste Konfiguration, in der die Kalibrierung im Langkontext etwas gewinnt.


### Ergebnis (2026-11-10): der Bias kostete ein Drittel des Fehlers

Auf den **echten** Zellen (C1a/C1b: 12 QSA-Layer × 32.746 Zellen × 2 KV-Köpfe, Block-MSE auf
denselben Zellen, gegen die publizierten Zahlen des Parity-Harness validiert):

| Variante | ohne Centering | mit Centering | Δ |
|---|---:|---:|---:|
| uniform+max (der Anker `q4_0`) | 9,093e-03 | **5,826e-03** | **−1,93 dB** |
| uniform+LS (der affine A8-Kandidat) | 8,686e-03 | 5,564e-03 | −1,93 dB |
| lloyd16+max | 8,443e-03 | 5,445e-03 | −1,91 dB |
| lloyd16+LS | 7,868e-03 | **5,072e-03** | **−1,91 dB** |
| lloyd16K+lloyd8V (der q4lm8v-Codec) | 1,288e-02 | **7,033e-03** | **−2,63 dB** |
| RHT(H·D) + uniform+LS (drei Züge) | −0,21 bis −0,24 dB | −0,18 bis −0,19 dB | ~0 |

**Der Bias kostete mehr als die Level-Tabelle (0,30 dB) und die per-Block-Skala (0,31 dB) zusammen.**
Zwei eigene Messfehler waren unterwegs und sind hier benannt, weil sie das Bild zuerst falsch
gemacht haben: (1) der **K-Offset wurde auch auf V angewendet** — die S2-Sonde misst **nur K**, und K
und V haben verschiedene Offsets; (2) die **S2-Sonde mittelt über die beiden KV-Köpfe**, ihre Zahl ist
damit weder ein Kopf- noch ein V-Offset. **Die Tabelle muss je Kopf und je Seite sein.**

**Korrektur 2026-11-11 (Abweichung 15): die offene Blockfrage ist geklärt, die Kernaussage steht.**
Das Kampagnen-Werkzeug (`tools/kv_codec_mse.py`) und das Attributions-Skript
(`tools/kv_center_attrib.py`) **stimmen auf dem vollen Dump auf sechs Nachkommastellen überein**:
`lloyd16+max` 8,448640e-03 (Werkzeug) gegen (1,4488e-02 + 2,4088e-03)/2 = 8,4484e-03 (Attribution),
zentriert 5,317906e-03 gegen 5,31791e-03 — zusammen **2,01 dB**. Die Angaben **„0,17 dB (K) und
0,36 dB (V)"**, die in Paper und Leiter standen, kamen aus einem Lauf des Attributions-Skripts über
einen **trunkierten** Zellsatz (`--max-cells`): dort werden die Offsets aus zu wenigen Zellen
geschätzt und tragen fast kein Signal. **Die frühere Rücknahme der Kernaussage (Abweichung 11) ist
damit widerrufen** — sie beruhte auf demselben trunkierten Lauf. Die Seitenzahlen des vollen Dumps:
**K −1,70 dB, V −4,63 dB** (je Layer zwischen −1,03 dB (Layer 7, K) und −7,02 dB (Layer 47, V)).

**C0 (der eigentliche Kill) hat nicht gefeuert:** über Text und Länge (cwe-4k gegen cwe-8k) liegt die
Korrelation der Kanalmittelwerte bei **0,9489** und die mittlere Abweichung bei **0,0799 sd** (Gate:
≥ 0,90 und ≤ 0,25 sd). Der Build-Effekt 2026-10-27 (Engine 0.1.47) gegen den aktuellen Build
(0.1.59.1) ist **exakt null** (Korrelation 1,0000, Abweichung 0,0000 sd) — der Merge hat die
K-Verteilung nicht bewegt.

**Stand der Umsetzung (C2 ist gebaut, C3 ist gelaufen):** die zentrierte Rotation und der
V-Add-Back als Kernel (`engine/src/kernels/cuda/kv_q4.cu`), alle Rotationsstellen über **eine** Writer-
Funktion (damit keine sie vergessen kann), der Loader `STRATA_KV_CENTER=<datei>` in `generate.cpp`,
der Generator `tools/kv_center_table.py` (aus dem Dump; der Dump hält die rotierten Zellen, die
Tabelle braucht den Kanalraum → der FWHT ist selbstinvolvierend), das Snapshot-Kennzeichen (+32) in
`conversation_snapshot.cpp`. **Nicht umfasst und als Grenze zu nennen:** die Draft-Layer (der MTP
rotiert seine eigenen K/V; der Draft-Layer ist keiner der 12 QSA-Layer → seine Offsets sind
ungemessen, er bleibt unzentriert) und der Hybrid (rotiert V ohne Centering → kein Add-Back).

### Ergebnis C3 (2026-11-10): die Wirkung im Engine — der Abstand zu `int8` halbiert sich

Der Arm ist **cwe-4k**, 32.746 Positionen, top-64-KL gegen fp16, gepaart gegen `int8`:

| Arm | KL (alle) | Top-1 | KL p99 | PPL |
|---|---:|---:|---:|---:|
| `int8` | 0,01688 | 99,05 % | 0,1888 | 2,860 |
| `q4lm8v` (plain) | 0,03182 | 98,40 % | 0,5785 | 2,888 |
| **`q4lm8v` + Centering** | **0,02505** | 98,75 % | 0,4127 | 2,871 |
| `q4lm8v` + Window | 0,02090 | 98,84 % | 0,2811 | 2,858 |
| **`q4lm8v` + Window + Centering** | **0,01871** | 98,87 % | 0,2665 | 2,862 |

plain −21 % KL, Window −10,5 % KL. Die gepaarte Differenz `int8 − Arm` (Segmentmittel, 8 Segmente):

| Paar | ΔKL | t | Segmente „up" | 95-%-Intervall (per Position) |
|---|---:|---:|---:|---|
| `int8` − win (ohne Centering) | −0,00402 | −3,55 | 1/8 | [−0,00677, −0,00127] |
| **`int8` − win + Centering** | **−0,00184** | **−1,85** | 2/8 | **[−0,00429, +0,00062]** |
| `int8` − plain + Centering | −0,00818 | −3,95 | 0/8 | [−0,01125, −0,00510] |

**G5 (die gepaarte KL verbessert sich) ist erfüllt.** Der Abstand zu `int8` fällt von +0,00402 auf
+0,00184 (−54 %) und ist **statistisch nicht mehr signifikant** (t = −1,85; das per-Position-Intervall
schließt 0 ein). **K1 (Leiter §7) ist zwei von drei Bedingungen erfüllt:** obere Grenze +0,00062 <
+0,001 ✓; Differenz ≤ 0 ✗ (positiv, aber halb so groß wie der Anker); `int8` führt in höchstens 4 von
8 Segmenten ✗ (6 von 8, der Anker 7 von 8). **Der ehrliche Satz: der Arm ist `int8` statistisch nicht
mehr unterlegen, aber noch nicht gleichauf.** *(Die Wirkung der Kalibrierung auf **diesen** Arm ist
+0,00218 nats und **nicht** signifikant — siehe C4 unten; die Zeile hier bleibt, weil sie den
Abstand zu `int8` beschreibt, nicht die Wirkung der Kalibrierung.)*

**Ein verworfener Lauf, im Protokoll benannt:** der erste Centering-Arm war **byte-identisch** zum
unzentrierten (SHA gleich) — der Verify-Pfad rotierte selbst (`verify.cpp:2010`) und hatte weder die
zentrierten Stellen noch den V-Add-Back. Das ist der Grund, warum die SHA-Gatter in dieser Kampagne
**Wirksamkeitsgatter** sind und nicht nur Qualitätsgatter: ein Arm, der „erfolgreich" läuft und dabei
nichts tut, fällt nur im Zahlenvergleich auf. Die wiederholten Läufe haben verschiedene SHAs.

**Der nächste Hebel, den C3 sichtbar macht:** die 8-Stufen-V-Tabelle unseres `q4lm8v`-Arms ist ein
**Messarm ohne Byte-Gewinn** (ein 3-Bit-Codebook in 4-Bit-Bytes: +1,10 dB MSE für dasselbe Bytebudget).
Der Produktcodec braucht **16 Stufen auf beiden Seiten** (`q4lm32w` + Fenster + Centering) — das ist
der Arm **C4**. Danach: 128K **out of sample** (die Offsets aus cwe-4k, die Texte aus den 128K-Draws),
weil die Kalibrierung bisher nur auf einem Text und einer Länge geprüft ist (G1: cwe-4k gegen cwe-8k).

### Ergebnis C4 (2026-11-10, abends): der Produktcodec 16/16 ist auf `int8`-Niveau — und die Kalibrierung wird auf ihm überflüssig

Zwei Arme auf cwe-4k, derselbe Build, dieselben Ids, das Fenster in der Produktionsgeometrie
(S 256, R 64, Chunk 512): `q4lm32w` (16/16) **ohne** und **mit** der Offset-Tabelle. Beide
32.746 Zeilen, 8/8 Segmente, verschiedene SHA → das Wirksamkeitsgatter ist erfüllt (der erste
Versuch scheiterte an der Verify-Fenster-Klasse #267, weil der Abhilfe-Schalter
`STRATA_VERIFY_ASYNC_LAUNCH=1` fehlte — Abweichung 9).

| Arm | KL (alle) | Top-1 | KL p99 | PPL | ΔKL vs `int8` (Segment) | t | `int8` führt in |
|---|---:|---:|---:|---:|---:|---:|---:|
| `int8` | 0,01688 | 99,05 % | 0,1888 | 2,860 | — | — | — |
| `q4_0` (Stock) | 0,02530 | 98,71 % | 0,3955 | 2,863 | +0,00843 | −3,99 | 8/8 |
| `q4lm8v` + Fenster (16/8) | 0,02090 | 98,84 % | 0,2811 | 2,858 | +0,00402 | −3,55 | 7/8 |
| `q4lm8v` + Fenster + Centering | 0,01871 | 98,87 % | 0,2665 | 2,862 | +0,00184 | −1,85 | 6/8 |
| **`q4lm32w` (16/16)** | **0,01689** | 98,89 % | 0,2410 | 2,852 | **+0,00002** | **−0,02** | 5/8 |
| **`q4lm32w` + Centering (16/16)** | 0,01768 | 98,97 % | 0,2263 | 2,856 | +0,00081 | −0,99 | 5/8 |

Per-Position-Intervalle (fp16-Referenz, autoritative Neuberechnung aller neun Arme in **einem**
Vergleichsdurchlauf, `b4\paired-all.json`):

| Paar | Δ | 95-%-Intervall | t | schlechter in |
|---|---:|---|---:|---:|
| `int8` − `q4lm32w` | −0,00002 | [−0,00240, +0,00237] | −0,01 | 3/8 |
| `int8` − `q4lm32w`+C | −0,00081 | [−0,00307, +0,00145] | −0,70 | 3/8 |
| `q4lm8v`-Fenster − `q4lm32w` | +0,00400 | [+0,00138, +0,00663] | **+2,99** | 7/8 |
| `q4lm8v` − `q4lm8v`+C | +0,00677 | [+0,00350, +0,01003] | +4,06 | 8/8 |
| `q4lm8v`-Fenster − `q4lm8v`-Fenster+C | +0,00218 | [−0,00022, +0,00459] | +1,78 | 6/8 |
| `q4lm32w` − `q4lm32w`+C | −0,00079 | [−0,00341, +0,00183] | −0,59 | 5/8 |

**Die drei Befunde:**

1. **Der Produktcodec 16/16 liegt auf `int8`-Niveau** (+0,00002, t = −0,02; das Intervall schließt 0
   ein). Gegen den 8-Stufen-V-Fenster-Arm ist er **+0,00400 nats besser** (t = +2,99, **signifikant**)
   — die 1,10 dB Block-MSE aus C1 sind end-to-end real.
2. **Die Kalibrierung ist ein Ersatz für den Level-Split, keine Ergänzung:** +0,00677 auf dem plain
   16/8-Arm (signifikant), +0,00218 mit Fenster (**nicht** signifikant, das Intervall schließt 0 ein),
   **−0,00079 auf 16/16** (nichts). Beide Hebel holen dieselbe verschwendete Spannung zurück.
3. **Die Produktentscheidung dreht sich:** Produktpfad ist `q4lm32w` **ohne** Kalibrierung — 16/16,
   Fenster, kein Artefakt (keine 12-KB-Tabelle je Modell), keine Basisdisziplin im Snapshot, keine
   1,8 % Laufzeit, und trotzdem auf `int8`-Niveau. Die Kalibrierung bleibt ein Hebel für Codecs, die
   Spannung verschenken (der 16/8-Probe).

**Korrektur zu Abweichung 10:** die C3-Tabelle in diesem Dokument nannte den Effekt der Kalibrierung
auf dem Fenster-Arm **+0,00416**; die autoritative Neuberechnung ergibt **+0,00218** (nicht
signifikant). Der Wert im C3-Protokolleintrag war falsch (Abweichung 10 im Kampagnen-Protokoll); die
Qualitätszahlen selbst (KL, Top-1, p99, PPL) sind unverändert.

**K1 (Leiter §7)** bleibt für den 16/16-Arm formal **nicht** erfüllt: Differenz ≤ 0 ✗ (mit
+0,00002, praktisch null), obere Grenze **+0,00237 > +0,001** ✗, `int8` führt in 5 von 8 Segmenten ✗
(gefordert ≤ 4). **Die zweite Bedingung ist auf diesem Korpus für keinen Arm erfüllbar** — die
Paritätsauflösung des Korpus liegt bei 0,00200. Das wird vermerkt, nicht stillschweigend geändert.

**Der Widerspruch zum Block-Harness, offen benannt:** C1 sagt, das Centering verbessere den
**16/16**-Codec um 1,91 dB Block-MSE; der Engine sagt end-to-end **nichts**. Die Gegenmessung
(je Layer und je Seite getrennt, `tools/kv_center_attrib.py`) läuft; ihr Ergebnis kommt ins Protokoll.
Bis dahin stützt sich die Produktentscheidung auf die **end-to-end**-Messung.

**Der Rotations-Vergleich läuft im selben Harness mit** (H·D, drei Vorzeichenzüge; A7), weil der
Harness ihn für einen Bruchteil der Kosten mitmessen kann — aber **G2 und G3 gehören dem
Centering**; die RHT-Varianten haben eine eigene, unbestätigte Erwartung (≈ 0) und werden **auf dem
zentrierten Signal** bewertet (Variante 3 gegen 2), weil eine Randomisierung einen systematischen
Anteil nicht entfernt, sondern nur anders verteilt.

### Ergebnis D1 (2026-11-11, abends): alle sechs 128K-Arme — und die Kalibrierung ist auch im Langkontext **kein Produktpfad**

Text `s00` (131.060 Positionen, `data/quant_quality/cwe-128k/t1.ids.json`), Build 0.1.59.1, die
Produktionsgeometrie bei 128K (Fenster S 256, R 64, Chunk 6.144), fp16 als Referenz, KL-Stützung:
die 64 wahrscheinlichsten Tokens der Referenz.

| Arm | KL (ganzes Segment) | p99 | Top-1 vs fp16 | PPL |
|---|---:|---:|---:|---:|
| `fp16` (Referenz) | — | — | 100,00 % | 2,751 |
| `int8` (K1-Anker) | 0,06435 | 1,6615 | 98,45 % | 2,697 |
| `q4lm32w` (16/16) | 0,07176 | 2,0583 | 98,30 % | 2,691 |
| `q4lm32w` + Centering (16/16) | 0,06861 | 1,9240 | 98,38 % | 2,703 |
| `q4lm8v` + Fenster (16/8, ohne Centering) | **0,07524** | 2,1567 | 98,16 % | 2,669 |
| `q4lm8v` + Fenster + Centering (16/8, der ausgelieferte Arm) | 0,07479 | 2,3328 | 98,18 % | 2,758 |

Die gepaarten Differenzen **über die Positionen** (`d1/paired-all6.json`; alle sechs Arme in **einem**
Vergleichsdurchlauf, damit die Paare vergleichbar sind):

| Paar (A − B) | ΔKL | 95-%-Intervall | t |
|---|---:|---|---:|
| `int8` − `q4lm32w` | −0,00741 | [−0,01044, −0,00439] | −4,80 |
| `int8` − `q4lm32w`+C | −0,00427 | [−0,00714, −0,00139] | −2,91 |
| `int8` − `q4lm8v`-Fenster | −0,01090 | [−0,01394, −0,00785] | −7,01 |
| `int8` − `q4lm8v`-Fenster+C | −0,01044 | [−0,01353, −0,00736] | −6,64 |
| `q4lm32w` − `q4lm32w`+C | **+0,00314** | [+0,00014, +0,00615] | **+2,05** |
| `q4lm8v`-Fenster+C − `q4lm8v`-Fenster | **−0,00045** | [−0,00362, +0,00271] | −0,28 |
| `q4lm32w` − `q4lm8v`-Fenster+C | −0,00303 | [−0,00614, +0,00007] | −1,91 |

#### Die Kalibrierungs-Wirkung: jetzt **drei** Messungen, und das Vorzeichen ist nicht etabliert

| Codec | 4K (cwe-4k) | 128K (t1) |
|---|---:|---:|
| `q4lm8v` + Fenster (16/8, der ausgelieferte) | +0,00218 (t = +1,78; n.s.) | **−0,00045 (t = −0,28; n.s.)** |
| `q4lm32w` (16/16, der Produktcodec) | −0,00079 (t = −0,59; n.s.) | **+0,00314 (t = +2,05; marginal)** |

(Vorzeichen: positiv = die Kalibrierung senkt die KL.) **Das Vorzeichen dreht bei beiden Codecs mit
der Länge**, und nur **eine** der vier Messungen ist überhaupt signifikant (16/16 bei 128K, t = +2,05,
marginal — das Intervall schließt fast null ein). **Die offene Frage aus C4 ist damit beantwortet:
die Kalibrierung ist kein Produktpfad, sondern ein Ersatz-Hebel, und ob sie den verlorenen
Level-Split zurückholt, hängt vom Codec *und* von der Länge ab.**

#### Der Tail und die Top-1 — dort ist die Kalibrierung wirksam

`tools/kv_tail_top1.py`, Referenz fp16, Tail = die 1 % der Positionen, in denen `int8` am schlechtesten
ist (1.311 Positionen, arm-unabhängig definiert):

| Codec-Paar | Tail-Mittel ohne C → mit C | Δ | Top-1-McNemar (ohne C / mit C richtig) | p |
|---|---|---:|---|---:|
| 16/16 | 3,42218 → 3,32933 | −0,09285 | 836 / 949 | **0,0080** |
| 16/8 (ausgeliefert) | 3,44739 → 3,30761 | −0,13978 | 1.041 / 1.070 | 0,5423 |

**Auf dem 16/16-Produktcodec verbessert die Kalibrierung bei 128K die Top-1-Übereinstimmung mit fp16
signifikant** (949 gegen 836 richtige Positionen, exakter McNemar p = 0,0080) — der einzige
signifikante Kalibrierungsbefund der ganzen Kampagne. Auf dem 16/8-Codec: nichts (p = 0,5423). Und im
Tail (den Positionen, an denen `int8` bricht) sind **beide** kalibrierten Arme besser als ihre
unkalibrierten Zwillinge — die Kalibrierung hilft genau dort, wo der Fehler ohnehin groß ist, auch
wenn sie den Mittelwert kaum bewegt. Das ist dasselbe Bild wie bei 4K (die Kalibrierung verbessert
p99 und Top-1, senkt aber den Mittelwert nicht).

#### Die Merge-Gegenprobe ist **exakt** (und eine frühere Deutung war falsch)

Der Vorgänger berichtet für seinen Fenster-Arm auf diesem Text **0,07524** (`docs/DETAILS.md`:
„whole segment 8-bit 0,06435, window 0,07524, plain 0,08834"). Unser **unkalibrierter** Arm misst auf
dem gemergten Build **0,07524** — auf fünf Nachkommastellen identisch. Die 0,6 %, die ich vorher als
„Merge-Abweichung" abgetan hatte, **sind die Kalibrierungs-Wirkung** (+0,00045, nicht signifikant),
kein Merge-Effekt. Der Merge ist für den Fenster-Arm **neutral ohne Restabweichung**; der kalibrierte
Arm (0,07479) weicht ab, weil er ein anderes Format ist. (`int8` stimmt ebenfalls exakt.)

#### Eine Vorab-Erwartung, die nicht eingetroffen ist

Der Zusatzarm war mit der Erwartung registriert worden, der kalibrierte 16/8-Arm solle bei 128K rund
0,0018 nats **unter** dem 16/16-Arm landen (0,0736 gegen 0,07176). **Gemessen: 0,07479 gegen 0,07176** —
er landet 0,00303 **über** dem 16/16-Arm (gepaart t = −1,91, nicht signifikant), also **über** dem
`int8`-Anker um 0,01044 (t = −6,64, signifikant). Die Erwartung war aus der 4K-Messung abgeleitet und
**falsch**: die Wirkung der Kalibrierung ist auf dem 16/8-Codec im Langkontext nicht nur schwächer als
erwartet, sondern **negativ**. Das ist der Grund, warum der 16/16-Arm der Produktcodec bleibt.

#### Das Ergebnis in einem Satz

**Auf 128K verliert der 4-Bit-Bulk-Codec gegen den 8-Bit-Anker (der Level-Split holt 0,00400 nats bei
4K, der Langkontext kostet 0,00741), die Kalibrierung holt auf dem 16/16-Codec bei 128K einen Teil davon
zurück (+0,00314, marginal, mit signifikant besserer Top-1-Übereinstimmung), aber sie ist auf dem
ausgelieferten 16/8-Codec im Langkontext wirkungslos (−0,00045) — und der Produktpfad bleibt
`q4lm32w` (16/16) **ohne** Kalibrierung, weil die Kalibrierung ein Artefakt (12 KB je Modell), die
Basisdisziplin im Snapshot und 1,8 % Laufzeit kostet.**

## 5. Was dieses Projekt nicht ist

- **Kein 5,0-Bit-Ziel.** Der Byte-Budget wird durch das Centering **nicht** berührt (kein zusätzliches
  Bit im Cache; die 12 KB liegen im Modell, nicht im KV).
- Kein Ersatz für die Level-Tabelle (0,30 dB) und nicht für die per-Block-Skala (0,31 dB) — und
  umgekehrt: es ist auch **kein Addendum** zu ihnen. Im Blockmaß ist das Centering mit **2,01 dB**
  sogar größer als beide zusammen (0,61 dB); end-to-end sind die beiden Hebel **Substitute**: auf dem
  16/16-Codec gibt das Centering nichts (−0,00079 nats, n.s.), weil die Level-Tabelle dieselbe
  verschwendete Spannung zurückholt; auf dem 16/8-Probe ist es umgekehrt, dort holt das Centering
  (+0,00677 nats) genau die Spannung zurück, die die 8-Stufen-V-Tabelle verschenkt.
- **Kein Ersatz für den Window-Overlay.** Das Fenster lieferte den großen Sprung (KL 0,0318 → 0,0209
  auf cwe-4k, gemessen 2026-11-10); das Centering ist eine Bulk-Hebelwirkung. Beide zusammen tragen
  0,0318 → 0,0187.
- **Kein „bias-free".** Entfernt ist der **erste** Moment (der Kanal-Mittelwert). Die Schiefe (3.
  Moment) bleibt systematisch (Median 0,1050 vor der Rotation), die Ausreißer bleiben (die Rotation
  flacht sie ab), und die Bias der Max-Statistik (das ist die per-Block-Skala) ist eine *andere*
  Bias-Entfernung. Das Centering braucht zudem ein **kalibriertes Artefakt** (3.072 Werte je Seite,
  12 KB) — es ist „mean-centered **calibrated**", nicht bias-frei.
- **Kein Ersatz für die Randomisierung, sondern ihre Voraussetzung.** Ein zufälliger Hadamard (A7)
  entfernt einen systematischen Anteil nicht, er verteilt ihn: für einen gegebenen Sign-Zug ist
  S·μ ein **fester** Vektor mit fester Energie (‖H·S·μ‖ = ‖μ‖). Deshalb wird A7 **erst nach** dem
  Centering gemessen, und die Erwartung an A7 bleibt ≈ 0: nach der Rotation stehen Mittel (0,2825),
  Schiefe (0,0836) und Kurtosis (0,1295) nahe an Gauss, aber nicht an Null — ein RHT verteilt diese
  Restanteile nur anders, er nimmt sie nicht weg. **Die Messung schließt A7 für dieses Modell:** drei
  Züge innerhalb 0,01 dB, auf den zentrierten Daten −0,18 dB (RHT) gegen −0,20 dB (uniform + LS). Das
  Regime-Gate ist die **prä-Rotation-Exzesskurtosis** (−0,18, leichter als Gauß) — die Versicherung
  gegen schwere Ränder wird hier nicht gebraucht (Belege: `docs/research/KV_ROTATION_SURVEY.md`).

## 6. Reproduktion (nach dem Bau)

```powershell
# C0: der zweite S2-Lauf (Text und Länge variieren), dann der Vergleich
python tools\quant_quality.py run --config bench\configs\cwe4k-swift-iq3_xxs-merged.json `
    --ids <zweiter Text> --kv q4lm8v --prefill 512 --short-read 0 --allow-no-positions `
    --env STRATA_KV_S2_STATS=<datei> --out <leer>
python tools\kv_s2_channels.py --tsv <datei> --json <datei.json>

# C1: der Parity-Harness mit und ohne die Offsets (gleiche Zellen, gleicher Build)
ctest --test-dir build-hip-port -R kv_q4_parity --output-on-failure

# C2/C3: der cwe-4k-Arm mit dem Schalter, gepaart ausgewertet
python tools\quant_quality.py run --config bench\configs\cwe4k-swift-iq3_xxs-merged.json `
    --ids data\quant_quality\cwe-4k\set.ids.json --kv q4lm8v --env STRATA_KV_CENTER=1 `
    --out ...\c3\q4lm8v-center.tsv
```

Rohdaten (TSVs, stderr, `seg.json`, S2-Dateien) liegen außerhalb des Repos unter
`C:\Users\krone\git\Strata-data\bench\2026-11-10-centering\`.

## 7. Grenzen, die schon jetzt gelten

- Die **128K**-Wirkung ist **gemessen** (D1, Text `s00`, 131.060 Positionen, sechs Arme) — und sie
  bestätigt die 4K-Aussage nicht, sie **präzisiert** sie: die Kalibrierung ist auf dem 16/16-Codec
  bei 128K **marginal positiv** (+0,00314, t = +2,05, mit signifikant besserer Top-1-Übereinstimmung,
  McNemar p = 0,0080), auf dem ausgelieferten 16/8-Codec **wirkungslos** (−0,00045, n.s.) — das
  **Vorzeichen ist über die Länge nicht etabliert**. **Offen bleibt:** der **zweite** 128K-Text
  (zurückgestellt) und damit die Frage, ob die Wirkung *über* die Texte hinweg trägt; der Umfang
  dieses Projekts ist also **ein 4K-Text und ein 128K-Text**, nicht zwei Längen und zwei Texte.
- Die **Schiefe** ist systematisch, aber nur **kanalspezifisch** (kein Layer-Schieft: der Mittelwert
  je Layer liegt bei 0,2 bis 2,0× Rauschen). Ihre Entfernung bräuchte dieselbe kanalspezifische
  Tabelle wie die Offsets (12 KB je Layer und Kopf) — ob sie nach dem Versatz überhaupt noch Stufen
  kostet, entscheidet die Spannungsnutzung u in C1, nicht dieses Dokument.
- Ob der Offset aus dem Modellgewicht ableitbar ist (Variante c), ist **ungeprüft**; ohne (c)
  bleibt ein Artefakt (Variante a, der aktuelle Stand) oder eine Onlineschätzung (Variante b).
- Die Basis-Regeln (Snapshot-Kennzeichen, Streaming, `m_V`-Add-Back in der richtigen Reihenfolge)
  sind **statisch** abgeleitet; ihre Prüfung ist Teil von C2 (Parität und ein Test, der die
  Reihenfolge absichtlich bricht und das Ergebnis prüft). Der Verifikations-Pfad ist dabei eine eigene
  Rotationsstelle — sie ist im Fix mit erfasst (`verify.cpp:2010`ff., Ausgabe `verify.cpp:2204`ff.).
- **Die Zeitmessungen dieser Kampagne sind zurückgezogen** (Fremd-GPU-Last, Abweichung 7 im
  Kampagnen-Protokoll). Für das Centering heißt das: es gibt **noch keine** Laufzeitmessung mit
  Offset — der Loader liest eine 12-KB-Tabelle einmal, der Writer zieht 256 Werte je Zeile ab, der
  Leser addiert einen Vektor je Token und Kopf. Das ist eine **Rechnung, keine Messung**, und wird
  erst mit dem belastbaren Laufzeitvergleich berichtet.
