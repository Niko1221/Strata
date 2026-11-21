# Der Fast-Path für nichtlineare (Level-Tabellen-) KV-Codecs: Ansätze, Literatur, RDNA4-Sachlage

**Stand: 2026-11-11.** Recherche, während die 128K-Läufe (D1) liefen. Fragen:
(1) Wie lässt sich ein **nichtlinearer** KV-Codec (Lloyd-Level-Tabelle) über einen schnellen
Attention-Pfad führen? (2) Welche Ansätze gibt es prinzipiell? (3) Welche sind für Strata **elegant**
und passend? (4) Was ist auf **RDNA4** anders — und was haben die Community-Benchmarks dazu gesagt?

**Kurzfassung:** Es gibt zwei tragfähige Familien — **(I) vor dem MMA dequantisieren** (Codebook/LUT:
TurboQuant, FLUTE, PolarQuant, Beyond) und **(II) den Codec affine machen** (Nullpunkt-Familie: KIVI,
OffQ, FP4/MXFP4). Familie (I) ist die, mit der die Literatur heute läuft, und sie ist **portabel über
die Backends**; Familie (II) ist die, die auf Strata **elegant** ist, weil sie den vorhandenen Kernel
unverändert lässt. Und die RDNA4-Sachlage: **unser 2,25×-Preis ist ein Prompt-Preis, und der
Matrix-Core-Pfad liefert auf unserer Karte keinen Prompt-Gewinn** (gemessen) — die Matrix-Core-Achse
ist damit **nicht** der Fast-Path, sondern ein **Decode**-Hebel; auf den iGPUs ist derselbe Pfad ein
**2,4×-Prefill**-Hebel.

## 1. Was unser Preis really ist (der Ausgangspunkt)

| Arm | Prompt-Lesedauer (32K-Nadel, freie Maschine, gleicher Build) |
|---|---:|
| `int8` | 51,7 s |
| `q4_0` (uniformes Gitter, **affin**) | 50,0 s |
| `q4lm8v` (Lloyd-Tabelle, **nicht-affin**) | 116,4 s (**2,25×**) |
| `q4lm8v` + Kalibrierung | 118,5 s (+1,8 %) |

Der Grund steht im Quellcode (`engine/src/kernels/cuda/qsa_prompt_attn.cu:2518-2524`): Mode 4 (die batched
Prompt-Attention) liest `code - 8` und wendet die Skala **danach** an — das funktioniert nur für ein
**affines** Gitter. Die Lloyd-Tabelle ist nicht affin (`x = level(code)·d`), also wird sie abgelehnt, und
zurück bleibt der Decode-Kern als Fallback (32 Queries pro Schritt). **Der Eigenkommentar des Kernels
benennt die Schuld:** *„Mode 4 could carry the level table the way it carries the int8 codes; **that
is owed if a level-table format ships**, and not needed to measure one."*

**Also: der zu lösende Teil ist die Affinität, nicht die Matrix-Cores.** Ob der Fast-Path danach
zusätzlich auf die Matrix-Cores geht, ist eine **andere, kartenspezifische** Frage (Abschnitt 4).

## 2. Familie (I): vor dem MMA dequantisieren (die Codebook-Familie)

**Der Referenzfall ist die AMD-Produktivierung von TurboQuant** (Chakrabarti et al., ROCm Blog,
2026-06-11; dieselbe Gruppe wie das arXiv:2607.09683-Paper aus unserer Recherche). TurboQuant ist
**genau unsere Nichtlinearität**: Rotation + **fixed codebook** (Beta-Lloyd-Max) + per-Token-Skala.

**Wie sie es schnell machen (das ist die übertragbare Lehre):**

| Optimierung | Wirkung (AMD, MI355X, Triton → GCN-ISA → FlyDSL) |
|---|---|
| GQA-Grouping + ein parametrisierter Kernel für Prefill und Decode | 2,49–3,30× |
| **SoA-KV-Layout** (coalesced 128-Bit-HBM-Reads) + **Pair-LUT** (FLUTE: zwei Dequant-Werte pro Byte-Load + Gather) + BF16-Dot | +0,2–0,3× |
| **native MFMA-Dispatch** (GCN-ISA-Instruktionen statt `tl.dot`) | +0,05–0,1× |
| 4-Wellen-paralleles Q·K + **V-Dequant in LDS** (alle GQA-Query-Heads teilen einen globalen V-Load) | +0,24–0,36× |
| **BF16-Q-Eingabe** direkt in die MFMA + Matrix-Core-PV | **+1,6–4,4×** |
| breitere MFMA, In-Register-Softmax, transponiertes V-LDS | +3,1–4,6× |

**Ergebnis: 32 % → 65 % (Triton) → 89 % (HIP) → 95 % (FlyDSL) der BF16-Durchsatz**; FlyDSL
überholt BF16 um +15,8 %, FP8-KV führt mit +37,5 %. Der **Decay-Schritt** (Dequantisieren in
LDS/Shared Memory, dann die **BF16**-Werte in die MFMA) ist also genau **unsere A8(a)-Variante
„Dequantisieren vor dem MMA"** — nur mit der LUT-Optimierung von FLUTE und dem GQA-Tiling.

**Die inhaltlichen Lehren der AMD-Studie, die direkt in unsere Leiter gehen:**
- **„Apply rotation and LUT-based quantization to K, and use standard uniform quantization for V."**
  (Keys sind quantisierungsempfindlicher als Values.) Bei uns ist es genau andersherum gemessen: der
  **Level-Split** (16/8 → 16/16) liegt auf der **V**-Seite (1,10 dB Block-MSE), und die
  **Kalibrierung** wirkt auf V stärker (−4,63 dB Block-MSE) als auf K (−1,70 dB) — die Blockmessung
  auf dem **vollen** Zellsatz (Abweichung 15 im Protokoll; die früheren 0,17/0,36 dB waren ein
  Artefakt eines trunkierten Zellsatzes). Das ist kein Widerspruch,
  sondern zwei verschiedene Knöpfe: die *Präzision* (Level-Anzahl) und die *Darstellung*
  (Rotation+LUT). Die AMD-Aussage betrifft die Darstellung.
- **„Use Walsh-Hadamard rotation instead of random rotation … both more kernel-friendly and more
  accurate."** ✓ deckt sich mit unserem A7-Befund.
- **„Omit QJL im Default-4-Bit-Pfad."** ✓ wir haben kein QJL.
- **„Skip boundary layers"** (die erste/letzte Attention-Layer sind am empfindlichsten) — **aber nur
  bei Voll-Attention-Modellen**; bei Hybrid-Modellen (unser Fall: 12 von 48 Layern tragen den Cache)
  ist der Effekt schwach, sagen die AMD-Leute explizit. → **für uns kein Hebel.**
- **„The benefit is strongest when KV-cache capacity is the bottleneck. In compute-bound settings,
  TQ can underperform BF16 due to quantization and dequantization overheads."** ← **die Warnung, die
  für unseren Prefill gilt**: unser 2,25× ist ein **Compute**-Fall (Prefill), nicht ein
  Kapazitätsfall. Die LUT-Dequantisierung kostet dort Rechenzeit, die sie in einem
  speicherbegrenzten Decode nicht kostet. **Ein Fast-Path, der den Prefill nur „so weit" bringt wie
  `q4_0`, ist das Richtige; einer, der die Dequant-Kosten in den Prefill legt, ist es nicht.**

**Weitere Mitglieder der Familie:** **FLUTE** (Guo et al., *Fast Matrix Multiplications for Lookup
Table-Quantized LLMs*, EMNLP-Findings 2024) — die LUT-Optimierung selbst; **PolarQuant** (Han et
al., Teil von TurboQuant) — Polarform mit 1-D-Codebooks (immer noch eine LUT, aber eine 1-D-Tabelle);
**Beyond** (EMNLP 2026) — **gelernte** nicht-uniforme Level: die *schwierigste* Variante für den MMA
(nur über den LUT-Weg).

## 3. Familie (II): den Codec **affin** machen (die Nullpunkt-Familie)

**Das Prinzip:** ein **verschobenes uniformes Gitter** mit einem (per Kanal, per Block, per Token)
**Nullpunkt**: `x ≈ s·(code − z)` — das ist **affin** ✓ und reitet damit Mode 4 mit demselben Kernel
wie heute (`Σ qᵢ·(code−z)`, die Skala danach). Das ist genau die Idee, die die Literatur als
**Zero-Point**/asymmetrische Quantisierung führt (OffQ: „can be absorbed into the zero-point"; KIVI:
asymmetrische K-Quantisierung per Kanal) — **und** unsere eigene Kalibrierung ist strukturell
dasselbe Objekt (ein per-Kanal-Offset, exakt und softmax-invariant auf der K-Seite, exakt
zurückaddiert auf der V-Seite).

**Der Preis:** die Lloyd-Tabelle ist **nicht-uniform** (für N(0,1) gerechnet, dichter um null). Ein
verschobenes uniformes Gitter hat diese Dichte nicht → es ist grober. **Diese Differenz ist im
Block-Harness messbar** (`tools/kv_codec_mse.py`, **Host, Minuten**): die Variante „uniform +
per-Block-Nullpunkt" gegen „lloyd16 + max", jeweils mit und ohne die per-Kanal-Kalibrierung.

**Und das ist die elegante Antwort für Strata**, weil sie **zwei Fliegen mit einem Klapfen schlägt:**
- **affin** → Mode 4 reitet ohne Codeänderung am Kernel → **die 2,25× verschwinden**;
- **und die per-Kanal-Kalibrierung bleibt kompatibel**, weil sie eine *exakte* Korrektur ist (K:
  softmax-invariant; V: exakt zurückaddiert) — sie ist **kein** Bestandteil des Codes, sondern eine
  Verschiebung vor der Rotation.

**Die Variantenliste aus A8 shrinks damit auf zwei:**
- **(c) verschobenes uniformes Gitter mit per-Block-Nullpunkt** — **die elegante** (affin, kein LUT,
  kein Kernel-Umbau; der Preis ist die Uniformität, im Harness messbar);
- **(a) Dequantisieren vor dem MMA (LUT)** — die **literaturübliche** (TurboQuant/FLUTE), wenn die
  Lloyd-Tabelle bleiben soll (und wenn der Prefill nicht compute-gebunden ist);
- **(b) stückweise affine Zerlegung** — die aufwendigste, und ohne Beleg, dass sie sich lohnt.

## 3b. Der Code beantwortet die Frage besser als die Literatur — und meine eigene „Affinitäts"-These war falsch

**Was ich zuerst glaubte (und in der Leiter stand):** *„der Kern rechnet `Σ qᵢ·(code−8)` im
Tensor-Core und wendet die Skala **danach** in FP32 an — eine nicht-affine Tabelle passt nicht."*
**Was der Kernel tatsächlich tut** (`engine/src/kernels/cuda/qsa_prompt_attn.cu`, der Pfad `KV_MODE 4`):

```cpp
for (int g = 0; g < NG; ++g) {              // die Skalengruppen (8 × 32 dims bei q4_0)
    float tg[4] = {0.f, 0.f, 0.f, 0.f};
    for (int kk = 0; kk < KPG; ++kk) {       // die 16-dim-MMA-Schritte der Gruppe
        b[0] = i8x2_to_h2(...S.kv.k[...]);   // die Codes -> exakt fp16 (der Bit-Trick)
        mma16816(tg, ah, b);                // die Produkte sammeln sich in der Gruppe
    }
    const float s0 = S.ks[cb + 2*tig][g];   // die GRUPPEN-Skala (32 Werte)
    sc[0] = fmaf(tg[0], s0, sc[0]);          // -> nach der Gruppen-Partialsumme
}
```

**Daraus folgt dreierlei, und es dreht die Fragestellung:**

1. **Die Skala ist pro *Gruppe* (32 Werte) und wird nach der Gruppen-Partialsumme angewendet.** Der
   Kernel unterstützt also **bereits** eine affine Karte, die **entlang der Head-Dimension variiert**.
   Meine Affinitäts-These war falsch: es gibt keine „eine Skala für die ganze Zeile"-Beschränkung.
2. **Der einzige fest verdrahtete Teil ist die *Wertabbildung* `code − 8`** (der `i8x2_to_h2`-Trick:
   zwei Codes werden exakt in ein fp16x2 verpackt). Und **der Kernel sagt es selbst**:
   *„Mode 4 could carry the level table the way it carries the int8 codes; **that is owed if a
   level-table format ships**."* → **der Fast-Path für die Lloyd-Tabelle ist keine Affinitäts-Reparatur,
   sondern ein Tabelleneintrag im Operandenbau:** `level[code]` statt `code − 8`. Die Level sind
   kleine ganze Zahlen (±87) → **exakt in fp16** ✓, die Gruppenskala bleibt unverändert ✓.
3. **Die „billige Abkürzung" (Weg 2, der affine Shear) ist damit noch billiger, als ich dachte:**
   eine **pro Block affine Karte** (ein Nullpunkt je 32 Werte) lässt den Operanden einen affinen
   Ausdruck des Codes bleiben — der Bit-Trick überlebt mit einem pro-Block-Konstanten — also
   **kein Lookup, überhaupt keine Änderung am Operandenbau**. Der Preis ist rein qualitativ.

### Die angewandte Mathematik dazu (der Hinweis des Nutzers)

**Die Lloyd-Tabelle *ist* der optimale Compander.** Das ist eine alte, exakte Aussage: Max 1960,
*„An optimal class of companding laws"*, und die Standardvorlesungen dazu (Stanford EE269,
MIT 6.450): *„Companding: we can find a nonlinear scalar transformation to warp the signal axis so
that optimal non-uniform bins become uniform bins … a uniform quantizer can be used as a universal
quantizer … the quantizer based on this scalar quantizer also satisfies the Lloyd–Max
conditions."* → **Die Lloyd-Level-Tabelle und der optimale Compander sind dasselbe Objekt.** Unsere
Level-Tabelle ist also bereits die *nichtlineare* (pro Element) Lösung des Problems.

**Und die Ingenieurs-Form davon sind die Shears:** μ-Law (logarithmisch) und A-Law
(stückweise-linear, 8 Segmente) — die Standard-Compander der Telefonie; die anwendungsnahe
Optimierung dieser Nichtlinearitäten ist eine eigene Literaturlinie (z. B. *„Designing of the
piecewise linear optimal compander"*, Perić et al., KaunasTech/ITC 2013: die stückweise-lineare
optimale Compander-Konstruktion für gaußförmige Quellen).

**Die Randbedingung, die in der reinen Quantisierungstheorie keine Rolle spielt und die hier
entscheidend ist:** **die Attention ist ein linearer Operator in den KV-Werten**
(`scores = q·k` linear, `output = P·v` linear; nur der Softmax dazwischen ist nichtlinear). Also
muss **jede elementweise Nichtlinearität vor der Summation invertiert werden** — ein Compander, der
erst *nach* der Summe invertiert würde (wie im Telefonie-Signalpfad, wo der Empfänger am Ende
expandiert), ist **für die Attention mathematisch falsch**: `γ⁻¹(Σ aᵢ·γ(vᵢ)) ≠ Σ aᵢ·vᵢ`, und auf der
K-Seite ebenso (`q·γ(k) ≠ γ(q·k)`). **Deshalb ist der LUT-Weg (Operand = dequantisierter Wert)
der einzige mathematisch zulässige**, und deshalb ist die Frage nicht „welche Nichtlinearität",
sondern **„welche Nichtlinearität pro Element (⇒ Lookup) oder pro Block (⇒ frei)"**.

**Damit ist die Entscheidungsfrage präzise und billig:**

| | nichtlineare Karte | **pro Element** (Lookup) | **pro Block/Gruppe** (frei) |
|---|---|---|---|
| Operandenbau | `level[code]` | Lookup (16 Einträge, exakt in fp16) | `code − z_b` (der Bit-Trick mit einer pro-Block-Konstanten) |
| Fast-Path-Kosten | ein Lookup je Element (FLUTE/AMD: Pair-LUT, SoA) | **keine** | **keine** |
| Qualität | die Lloyd-Tabelle (unser Anker) | der Preis des Uniformitäts-Verlusts | **im Block-Harness messbar** |

**Und der philosophische Punkt:** Unsere Lloyd-Tabelle ist der *optimale Compander* für N(0,1) —
also die beste nichtlineare Karte. Die Frage „welche affine Näherung stellt den nichtlinearen
Quantisierungsfall ausreichend gut affin dar" (die Formulierung des Nutzers) ist damit genau die
Frage: **wie viel vom Lloyd-Max-Vorteil bleibt, wenn die Karte auf eine pro-Block affine Karte
beschränkt wird?** Das ist eine **Block-MSE-Frage** und damit eine **Host-Frage ohne Kernel**
(`tools/kv_codec_mse.py`: „uniform + per-Block-Nullpunkt" gegen „lloyd16 + max", je mit und ohne
die per-Kanal-Kalibrierung).


## 3c. Der Architekturentscheid, **gemessen** (2026-11-11): was die affine Beschränkung kostet

**Der Zähler, den die Leiter verlangt hat, ist gelaufen:** `tools/kv_affine_price.py` auf den
**Engine-Zellen** (der Dump `cells-cwe4k.bin`, 12 QSA-Layer, 768 Records, 1.572.480 Blöcke bei
stride 8) — dieselben Zellen und dieselben Implementierungen wie die Kampagne (die vier bekannten
Varianten sind **bitweise identisch** zum Kampagnen-Werkzeug: 1,124507e-02 / 1,177749e-02 /
1,024466e-02 / 1,100851e-02 auf denselben Blöcken, Block für Block geprüft).

### Die Zahlen (Block-MSE, unkalibriert; die dB-Spalte gegen `lloyd16+max`)

| Variante | MSE | K | V | dB |
|---|---:|---:|---:|---:|
| `uniform+max` (`q4_0`, der Anker) | 9,102041e-03 | 1,562787e-02 | 2,576217e-03 | +0,33 |
| `lloyd16+max` (die Engine-Tabelle `q4lm32`) | 8,443299e-03 | 1,447745e-02 | 2,409149e-03 | 0,00 |
| `lloyd16+LS` (die Engine-FIT-Variante) | 7,867572e-03 | 1,349121e-02 | 2,243936e-03 | −0,31 |
| **`uniform+LS`** (uniformes Gitter, Skala per LS, **Nullpunkt 0**) | 8,693901e-03 | 1,492703e-02 | 2,460767e-03 | **+0,13** |
| **`uniform+LS+z`** (die Scherung: Nullpunkt je Block) | **6,459338e-03** | 1,110243e-02 | 1,816245e-03 | **−1,16** |

**Korrektur 2026-11-13 (Abweichung 20 im Protokoll):** diese Tabelle stand bis heute **fünfmal zu
klein** hier (1,820408e-03 usw.). Der Zähler in `tools/kv_affine_price.py` (`Acc.n[side]`) wurde bei
jeder Variante erhöht, also fünfmal je Datensatz und Seite, und jede MSE war damit 1/5 ihres Werts.
Der Fehler ist behoben (`Acc` zählt jetzt je Variante und Seite), die Zahlen oben sind mit dem
korrigierten Werkzeug auf demselben Dump und demselben `--stride 8` neu gerechnet. Die **dB-Werte
waren als Verhältnisse unberührt** (0,434 / 1,290 / 0,857 dB wie zuvor 0,43 / 1,29 / 0,86), und
**jede Entscheidung der Kampagne beruhte auf dB** — die Schlussfolgerungen ändern sich nicht, die
absoluten Zahlen schon. Eine unabhängige Rechnung auf **allen** 4.095 Positionen (ohne `stride 8`,
12.574.464 Blöcke, `tools/kv_residual_offset.py`) bestätigt sie: 9,087669e-03 / 8,432340e-03 /
7,857215e-03 / 8,680457e-03 / 6,464105e-03.

Kalibriert (`--center`, die Engine-Tabelle): dieselbe Richtung — Uniformitätspreis **0,413 dB**,
Scherungspreis **1,285 dB**, die Scherung **1,18 dB** besser als `lloyd16+max`. Die implementierbare
Regel (zwei LS-Durchläufe) liegt **0,36 dB** (unkalibriert) bzw. **0,44 dB** (kalibriert) von der
erschöpfenden Oracle entfernt.

### Die drei Preise — und damit die Antwort

| Preis | Wert | Bedeutung |
|---|---:|---|
| **Uniformitätspreis** (`lloyd16+LS` / `uniform+LS`) | **0,43 dB** | die **Form** (nicht-uniforme Stufen) kostet das — bei **denselben** Bytes |
| **Scherungspreis** (`uniform+LS` / `uniform+LS+z`) | **1,29 dB** | die **Platzierung** (der Nullpunkt) kauft das — aber **ein Extra-Budget** (0,5 bit/Zelle) |
| der kernelfreie Weg (`lloyd16+max` / `uniform+LS`) | 0,13 dB | der Weg ganz ohne Kernel-Änderung |

### Warum der Nullpunkt so viel holt: es ist der **Bias**

Die rotierten K/V-Zeilen tragen einen **festen Kanal-Offset** (der C1-Befund: der Blockmaß-Fehler
sinkt um 2,01 dB, wenn man ihn vorher abzieht). Ein **um null zentriertes** uniformes Gitter
verschenkt deshalb Range: der fitted Nullpunkt liegt im Median **0,94 Schritte** vom Gitterzentrum
(90 %-Quantil 2,27), und die Konstante, die der Kernel je Block addieren müsste, ist **unter einer
halben Stufe** groß. **Der per-Block-Nullpunkt entfernt den Bias pro Block** — ohne Tabelle, ohne
Basisdisziplin im Snapshot, ohne Artefakt.

**Er ist aber kein Ersatz für die per-Kanal-Kalibrierung:** die Kalibrierung holt 2,01 dB, der
per-Block-Nullpunkt 1,29 dB, **beide zusammen 3,18 dB** (8,4433e-03 → 4,0594e-03). Der Kanal-Offset
ist ein Vektor über alle Blöcke, der Block-Nullpunkt ein Skalar je Block.

### Was das für die Architektur heißt — und was nicht

1. **Der kernelfreie Weg ist billig, aber er kostet 0,43 dB** (gegen die beste Lloyd-Variante). Das
   ist **weniger**, als die Theorie für Gauß-Daten nahelegt (Lloyd-Max gegen den besten
   Uniform-Quantizer mit 16 Stufen: +0,88 dB; auf 32-Wert-Blöcken ist die *freie* Form sogar
   8 dB besser als jede feste Form — der transductive Anpassungseffekt kleiner Blöcke).
2. **Der affine Shear ist nicht „gut genug", er ist besser** (+0,86 dB gegen `lloyd16+LS`) — **aber
   auf anderen Bytes** (5,0 statt 4,5 bit/Zelle; mit einem 5-Bit-Nullpunkt 4,66) **und mit einem
   Kernel-Term** (K: die Query-Gruppensummen, je Kopf einmal — billig; V: eine gewichtete Summe je
   Gruppe — ein zusätzlicher Reduktionsschritt).
3. **Keine End-to-End-Aussage.** Die Block-MSE übersetzt sich nicht 1:1 in nats; die einzige
   vermessene Brücke der Kampagne ist ein Datenpunkt (1,10 dB Block ↔ +0,00400 nats, der
   Level-Split, t = +2,99). **Eine Beobachtung, kein Gesetz.**
4. **Der Nullpunkt ist ein Codec, kein Fast-Path.** Er braucht ein eigenes Byte-Format (ein
   Nullpunkt je 32-Wert-Block) und einen End-to-End-Arm — er gehört in eine **eigene Stufe**, nicht
   in S6.

### Und die Verwandtschaft mit der Literatur

Der per-Block-Nullpunkt ist genau das, was die Literatur „Zero-Point" nennt (OffQ, KIVI) — nur
**pro Block** statt pro Kanal. Und er ist die **angewandte Mathematik**, die die Anregung
gemeint hat: die Lloyd-Tabelle ist der optimale Compander (Max 1960), und ein **verschobenes
uniformes Gitter** ist die **Ingenieursform** des Companders (μ-Law, A-Law) — die Frage „wie viel
vom Lloyd-Max-Vorteil bleibt bei pro-Block-Affinität" ist damit **nicht** mit „nichts" beantwortet,
sondern mit **„bei gleichem Budget 0,43 dB Verlust, bei einem Extra-Halbit 0,86 dB Gewinn"**.

**Nebenbefund (ein Werkzeugfehler, im Protokoll als Abweichung 17):** die Option `--layers` in
`tools/kv_codec_mse.py` filtert mit `range(N)`; die Layer des Dumps sind aber **3, 7, 11, … 47**,
also lieferte `--layers 1` „no records". Das ist der Grund, warum der Vergleichslauf der Kampagne
notgedrungen auf „Layer 3+7" stand.


## 3d. **Was der Nullpunkt tut** — die Analyse, gemessen (2026-11-11)

Dieser Abschnitt beantwortet die Rückfrage „macht die gescherte Variante nicht auch eine Mittelwert- oder
Center-Korrektur?" — und die Antwort ist **nein, nicht dieselbe**. Die Messung: dieselben Zellen wie
oben (Layer 3, je 256 Zellen → 16.384 Blöcke), die K- und die V-Seite getrennt.

### Der Nullpunkt ist eine **Form**-Korrektur, keine Mittelwert-Korrektur

| | K | V |
|---|---:|---:|
| Korrelation: **Nullpunkt** ↔ **Blockmittelwert** | +0,51 | +0,63 |
| Korrelation: **Rest** (Nullpunkt − Mittelwert) ↔ **Blockschiefe** | **+0,89** | **+0,89** |
| Median \|Blockmittelwert\| / Streuung des Blocks | 0,11 | 0,14 |
| Median Schiefe des Blocks | +0,035 | +0,044 |
| Median Nullpunkt (in Code-Einheiten; die Kernel-Konvention ist 8) | 7,44 | 7,35 |
| Median Beitrag des Blockmittelwerts zum Nullpunkt | **0,03 Schritte** | — |

**Der Blockmittelwert ist nahezu null** (Median −0,0026 auf der K-Seite, 0,11 Streuungen) und trägt
im Median **0,03 Schritte** zum gefitteten Nullpunkt bei. Was den Nullpunkt bestimmt, ist die **Schiefe**
der Blockverteilung (r = +0,89 mit dem Rest): die Verteilung liegt dichter auf einer Seite, und ein um
null zentriertes Gitter gibt Range für die andere Seite aus.

### Die Verschiebung ist klein — und trotzdem wirkt sie

| | Median \|Fehler\| im Bulk | geclippt | Stufenweite \|s\| | Anteil der Clips am Fehler |
|---|---:|---:|---:|---:|
| `uniform+LS` (Nullpunkt 0) | 0,0868 | 5,84 % der Werte | 0,367 | 4,8 % |
| `uniform+LS+z` (geschert) | **0,0744** | 8,18 % der Werte | 0,349 | 4,8 % |
| `lloyd16+max` (die Engine-Tabelle) | 0,0795 | — | — | — |

**Zwei Erklärungen, die man naheliegend annimmt, sind falsch** und wurden geprüft: die Stufenweite ist
**fast gleich** (0,367 gegen 0,349), und das Clipping ist **nicht** die Geschichte (die Clips tragen in
beiden Fällen 4,8 % des Gesamtfehlers; die gescherte Variante clippt sogar *mehr*). Was passiert: die
gescherte Variante holt im **Bulk** 14 % (Medianfehler 0,0868 → 0,0744) und zahlt das mit dem Tail,
dessen Fehleranteil klein bleibt.

### Es ist eine **dritte** Art, den Bias zu entfernen

| | **was** | **wo** | **Wirkung** (Block-MSE) | **Preis** |
|---|---|---|---:|---|
| **Kalibrierung** | ein **Vektor** je (Layer, Kopf, Kanal) | **vor** der Rotation, für alle Blöcke | **2,00 dB** | 12 KB je Modell, Basisdisziplin im Snapshot, V-Add-Back |
| **Nullpunkt** | die **Form/Lage** je Block | am Encoder, **ohne** Tabelle | **1,16 dB** | **0,5 bit/Zelle** + eine Kernel-Konstante je Block |
| **Rotation** | verteilt nur (‖H·μ‖ = ‖μ‖) | — | 0 | — |

**Sie sind additiv, nicht substituierend:** 2,00 + 1,16 = **3,18 dB** (gemessen: roh 8,443299e-03 →
kalibriert 5,326535e-03 → kalibriert **und** geschert 4,059400e-03; bis 2026-11-13 standen hier die
fünffach zu kleinen Werte, siehe die Korrektur in §3c). Der Kanaloffset ist ein fester
Vektor, der über alle Blöcke gleich ist; der Blockanteil schwankt von Block zu Block.

### Korrektur (2026-11-13): wie viel davon ist wirklich der **Offset**?

Die Zeile „Nullpunkt 1,16 dB" beschreibt den **Shear** (Codes **und** Schritt **und** Offset
gemeinsam neu gefittet). Die zerlegende Messung steht in
`bench/results/2026-11-13-kv-offset-tail/` (nicht in diesem PR enthalten; 12.574.464 Blöcke, dieselben Zellen):

* Der **reine Restfehler-Offset** bei **festen** Codes und fester Skala
  (`a = mean(x − s·c)`, `r' = r + a`) holt **3,16 %** der Block-MSE zurück — **12,4 %** des
  Shear-Gewinns. Die anderen **88 %** sind die Neuplatzierung des Gitters.
* **Sechs Bit** speichern `a` fast verlustfrei (96,8–98,3 % des exakten Gewinns bei 1/32 Schritt).
* Eine **Kanal-Tabelle** (Layer × Seite × Kopf × Blockindex, also der „statische" Anteil) holt
  **nichts** zurück (≈ 0 %); je 2 Blöcke geteilt verliert die Hälfte, je 4 Blöcke drei Viertel.
* Im **1-%-Tail** der Positionen ist der Anteil genauso groß wie im Bulk (3,1 % gegen 3,1 %) — der
  Offset ist **kein** Schwerpositions-Hebel.

**Präzisierung der Formulierung „Formkorrektur, keine Mittelwertkorrektur":** der Offset korrigiert
den Mittelwert des **Rekonstruktionsfehlers**, nicht den Mittelwert der **Eingangsdaten**; das ist
auch bei fast zentrierten Daten relevant (der Blockmittelwert ist nahezu null, der Fehlermittelwert
nicht). Die Schiefe-Korrelation r = +0,89 sagt, **dass** die Größen zusammen schwanken — sie beweist
**keinen** Mechanismus. Der Mechanismus, den die Zerlegung zeigt, ist: das Gitter wird neu platziert,
und der DC-Anteil davon ist klein.

**Für die Brücke Block → nats heißt das:** von den 1,16 dB (≈ 0,00193 nats) trägt der billige
Fixed-Code-Offset allein **0,14 dB ≈ 0,0002 nats** — ein Zehntel des Shear-Effekts und weit unter
der Auflösung des Harness (0,00200 nats).

### Die Regel, die für den Plan daraus folgt

**Wo man die Korrektur anbringt, entscheidet, was sie kostet.** Dieselbe Sache — den verschobenen
Block richtig legen — ist **im Modell** billig (12 KB einmal, für alle Blöcke, keine Cache-Bits) und
**im Cache** teuer (0,5 bit *pro Zelle* plus Kernel-Arbeit). Die Kalibrierung ist die eine Variante,
der Nullpunkt die andere. Deshalb ist die Reihenfolge der Eingriffe im Pipeline eine **Designfrage**,
und deshalb kann die Frage „welche Center-Korrektur?" nicht ohne die Frage „wo?" gestellt werden.

### Was daraus **nicht** folgt

- **Keine End-to-End-Aussage.** Der Blockfehler übersetzt sich nicht 1:1 in nats (die einzige
  vermessene Brücke ist ein Punkt: 1,10 dB ↔ +0,00400 nats, der Level-Split).
- **Der Nullpunkt ist kein Ersatz für die Kalibrierung** (1,16 gegen 2,00 dB, und additiv).
- **Der Nullpunkt ist ein Codec, kein Fast-Path:** er braucht ein eigenes Byte-Format (ein
  Nullpunkt je 32-Wert-Block, +2 Bytes je 32 Werte → 5,0 bit/Zelle; ein 8-Bit-Nullpunkt gäbe 4,66)
  und eine End-to-End-Messung, bevor er über seine Qualität entscheidet.

## 4. RDNA4: die Matrix-Core-Achse ist **nicht** unser Fast-Path — und die Community misst sie uneinheitlich

### 4a. Unsere eigene Messung (der Kern)

`bench/results/2026-11-08-wmma-int8/` (`int8`, 4k/16k-Prompts, Commit `c7be7d4b7`; Rohdaten, außerhalb des
Repos gehalten),
`STRATA_HIP_WMMA=1` gegen aus:

| | Prompt 4k | Prompt 16k | Decode (short / 4k / 16k) | Sanity |
|---|---:|---:|---|---|
| WMMA **aus** | 479,4 tok/s | 565,4 tok/s | 47,3 / 42,4 / 42,2 | 12/12 ✓ |
| WMMA **an** | 478,0 tok/s | 565,5 tok/s | 45,7 / 45,3 / 42,1 | 12/12 ✓ |

**Die Lesart: der WMMA-Pfad ist bei uns ein Decode-Hebel, kein Prompt-Hebel.** Die Prompt-Raten sind
unverändert; der einzige Unterschied liegt im Decode — im 4k-Fall +6,9 %, im 16k-Fall ±0, im
Short-Fall −3,4 %, mit **überlappenden Spannen** (also kein gesicherter Durchbruch; „durchgängig" ist
er bisher nicht belegt).

**Die entscheidende Konsequenz für den Plan: unsere 2,25× sind ein *Prompt*-Preis — der WMMA-Pfad
kann sie auf dieser Karte nicht bezahlen.** Die Stage, die ich ursprünglich vorgeschlagen hatte (ein
32K-Nadel mit/ohne WMMA, um den Prompt-Gewinn zu suchen) ist damit **hinfällig**: an der Stelle, wo
unser Preis liegt, liefert der Schalter nachweislich nichts. **Der Fast-Path ist die Struktur
(Abschnitt 1), nicht die Matrix-Cores.**

### 4b. Die Community-Benchmarks für unsere Kartenfamilie (RDNA4)

| Bericht | Karte | WMMA | Was er zeigt |
|---|---|---|---|
| `bench/results/2026-10-10-community-rx-9060-xt/` (**unser**) | RX 9060 XT, gfx1200, **Windows**, Ryzen 7 9800X3D | **an** (`STRATA_HIP_WMMA=1`) | Prompt-Rate skaliert mit dem Prefill-Chunk: 346,3 (2.048) → 593,3 (4.096) → **879,4 tok/s** (8.192); Decode flach 52,7–57,9. **Kein WMMA-A/B im Bericht** — der Schalter gehört zur Baseline. |
| `bench/results/2026-10-03-community-r9700-windows/` | R9700, gfx1201, Windows | (der Bericht verweist auf den AMD-Bericht #499 / RX 9070 XT, Prompts bis 31.059 Token) | der Windows-Fall von RDNA4 ist community-validiert |
| `bench/results/2026-10-08-community-r9700-linux/` | R9700, gfx1201, **Linux** | **aus** (`STRATA_HIP_WMMA=0`), dafür `STRATA_PREFILL_MMQ=1` (der **Prompt-GEMM**-Pfad, nicht die Attention) | **die Community-Praxis auf RDNA4-Desktop dreht den Matrix-Core-Attention-Pfad nicht an** |
| PR **#5535** | RX 7900 XTX, gfx1100 (RDNA3) | int8-WMMA-Tier | −32,1 % **Kernel**-Zeit, aber **kein End-to-end-Gewinn** (0,995, CI [0,994; 1,004]) → dieselbe Aussage wie bei uns; und: „RDNA4 ships the same instruction family and already defaults to the f16 WMMA prompt path — the layout re-proof on an RDNA4 card is one probe run" |
| PR **#5725** | gfx1152 iGPU | WMMA-Pfad freigeschaltet | **32K-Prefill 79-80 → 194-201 tok/s (2,4×)**, 4K 84-86 → 97-101 (1,17×), 131K 179,9. Das Rezept: **zwei Hälften, beide nötig** — compile-time der Arch-String in die WMMA-/dp4a-Device-Guards (9 Dateien), runtime in die **zwei** Host-Gates (`gfx_arch_is_gfx11_wmma()`, `igpu_prompt_defaults_arch()`) |

**Das Muster:** Auf den **iGPUs** ist der Matrix-Core-Prompt-Pfad ein **2,4×-Prefill**-Hebel bei 32K
(bei 4K nur 1,17×). Auf den **Desktop-RDNA4-Karten** (gfx1200/1201) liefert er **keinen** Prompt-Gewinn
(bei uns gemessen; auf RDNA3 derselbe Befund) und wird von der Community **uneinheitlich** benutzt
(Windows-Bericht: an, Linux-Bericht: aus). „Wo der Hebel liegt", hängt an der Karte — und
**auf unserer** Karte liegt er (wenn überhaupt) im **Decode**, nicht im Prompt.

### 4c. Ein Nebenbefund aus unserem eigenen Community-Bericht, der größer ist als der WMMA-Schalter

Der 9060-XT-Bericht misst den **Prefill-Chunk** als Konfigurationshebel: **2.048 → 8.192 verdoppelt
die Prompt-Rate** (346,3 → 879,4 tok/s, **2,5×**), 16.384 bringt nichts mehr. Das ist auf unserer Karte
**größer als jeder Schalter**, und es ist eine *Konfiguration*, kein Kernel. (Für den 2,25×-Vergleich
zwischen `q4_0` und dem Lloyd-Arm ist der Chunk irrelevant — beide Arme liefen mit derselben
Geometrie; aber als * absoluter* Prompt-Hebel gehört er in dieselbe Liste.)

## 5. Was ich daraus für den Arbeitsplan vorschlage

1. **S6 wird umgestellt und verkleinert:** der 2,25×-Preis ist ein **Struktur**-Preis (Mode 4 vs.
   Decode-Kern), **kein** Matrix-Core-Preis. Die Matrix-Core-Achse ist (a) kartenspezifisch, (b) auf
   unserer Karte ein **Decode**-Hebel ohne gesicherten Durchbruch und (c) für den Prompt-Preis
   irrelevant. Sie gehört **nicht** in die Fast-Path-Stufe, sondern als eigener, kleiner Punkt
   („Decode-Hebel des WMMA-Pfads: durchgängig belegen oder streichen").
2. **Der Test, der die Architektur entscheidet (Host, Minuten):** im Block-Harness die Variante
   **„uniform + per-Block-Nullpunkt"** (affin!) gegen **„lloyd16 + max"**, jeweils mit und ohne die
   per-Kanal-Kalibrierung. **Fällt der Uniformitäts-Preis klein aus, ist Variante (c) der Produktcodec
   — und der Fast-Path kostet keinen Kernel.** Fällt er groß aus, bleibt die Lloyd-Tabelle, und der
   Fast-Path wird der LUT-Weg (TurboQuant/FLUTE) — mit dem AMD-Vorbehalt: im **compute-gebundenen
   Prefill** zahlt die Dequantisierung.
3. **Die Reihenfolge:** (i) der Harness-Test des Nullpunkt-Gitters (Minuten, Host), (ii) die
   Level-Tabelle in Mode 4 (falls (i) nicht trägt), (iii) der LUT-Weg (falls (i) zu teuer ist).

**Quellen:** AMD ROCm Blog „Productionizing TurboQuant on AMD GPUs for KV-Cache-Bound LLM Inference"
(2026-06-11) · Guo et al., *Fast Matrix Multiplications for Lookup Table-Quantized LLMs*, EMNLP
Findings 2024 (FLUTE) · Zandieh et al., *TurboQuant*, ICLR 2026 (arXiv:2504.19874) · Han et al.,
PolarQuant (in TurboQuant) · *Beyond: Better-than-Full-Precision KV Caches via Learnable
Non-Uniform Quantization*, EMNLP 2026 · D'Alberto et al., arXiv:2607.09683 (unsere Rolle) · Jokar,
*Transforms for LLM Quantization*, arXiv:2608.25188 (der Zahlformat- Strang: FP4/MXFP4/NVFP4 sind
uniform → **affin** → der Fast-Path kommt dort gratis; die „Great Inversion" sagt, dass das uniforme
FP4-Gitter das Flatten weniger belohnt → **Codec- und Rotationswahl hängen aneinander**) · OffQ
arXiv:2606.07116 (Nullpunkt) · KIVI (asymmetrisch per Kanal) · die Community-Berichte
`bench/results/2026-10-10-community-rx-9060-xt/`, `.-2026-10-03-community-r9700-windows/`,
`.-2026-10-08-community-r9700-linux/` · Upstream-PRs **#5535** und **#5725** · unsere eigene
WMMA-A/B-Messung `bench/results/2026-11-08-wmma-int8/` (Rohdaten, außerhalb des Repos gehalten).
