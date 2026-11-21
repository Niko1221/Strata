# Der Matrix-Pfad für `q4u32` auf RDNA4: was die Hardware kann, was die Praxis tut, was uns trennt

Stand 2026-11-14, fortgeschrieben 2026-11-17 (§7 Laufzeitstand, §8 Literatur-Update, §9 die gfx11-Familie).
Gehört zu `bench/results/2026-11-14-q4u32-fastpath/README.md` (Serie #1, Teil B).
**Das Paper zum Produktpfad:** `docs/paper/KV-q4u32-matrix-core.pdf` (q4u32: A 4-Bit KV Cache the Matrix Core
Can Read) — es trägt die Messungen A.16–A.26 und die Ablation in der Form, in der sie veröffentlicht
werden; diese Notiz bleibt der Arbeitsstand mit den offenen Armen (A.24/A.25).
Die eigene Messung ist als **Messung** markiert, alles andere als **Quelle** oder als **Rechnung**.
Eine Rechnung ist keine Messung: sie sagt, was zu erwarten wäre, nicht was gilt.

## 1. Die Frage

`q4u32` (= `kUniformLs`: 18-Byte-Block, fp16-Skala `d` + 16 Byte Codes, Skala je 32 Werte,
`code - 8` ist der Wert in Einheiten von `d`) läuft heute über den **Decode**-Kern, weil der
Matrix-Core-Prompt-Kern den 4-Bit-Pool nicht lesen konnte. A1 hat den Preis gemessen: **263,7 tok/s
gegen 634,1 tok/s für int8** (32K-Nadel, 3 Tiefen, Median). B2 hat den Kern gebaut, der den Pool liest:
**97,3 → 19,0 ms/Chunk, 5,12×** (Messung, Paritäts-Harness). Er hängt aber im echten Lauf.

*(Stand 17.11.2026: der Hänger ist weg — A.16 war das rote Gate, A.17 der grüne Durchgang, A.23 der Default.
§7 sagt, was gemessen ist; §1 bleibt der Stand vom 14.11., weil §5 die Reihenfolge der nächsten Schritte begründet.)*

Damit sind es drei verschiedene Fragen, die man nicht vermischen darf:

1. **Läuft er?** (Hänger — offen, blockiert alles)
2. **Was kostet er danach?** (A2 — erst nach 1 messbar)
3. **Wie schnell könnte er sein?** (Matrix-Rate — die Frage dieses Dokuments)

## 2. Was gfx12 an Instruktionen anbietet

RDNA4 hat WMMA in wave32. Die Rate kommt aus der Zyklustabelle von AMDs eigener Referenz
(**Quelle**: ROCm Optimization Guide, „RDNA4 dense WMMA builtins"):

| Builtin | Ops je Instruktion | Zyklen | MACs/Zyklus je CU | gegen fp32-VALU |
|---|---:|---:|---:|---:|
| `wmma_f32_16x16x16_f16` (unser Kern) | 8192 | 16 | 256 | 2× |
| `wmma_i32_16x16x16_iu8` | 8192 | 8 | 512 | 4× |
| `wmma_i32_16x16x16_iu4` | 8192 | 8 | 512 | 4× |
| `wmma_i32_16x16x32_iu4` | 16384 | 8 | 1024 | 8× |
| `wmma_f32_16x16x16_fp8_fp8` | 8192 | 8 | 512 | 4× |

Die „MACs/Zyklus"-Spalte ist meine Umrechnung (Ops/2 = MACs, ÷ Zyklen), nicht AMDs Spalte.
**INT8 ist auf RDNA4 doppelt so schnell wie FP16** — das ist der einzige echte 2×-Schritt, den die
Karte für Q·K hergibt. INT4 verdoppelt noch einmal, zwingt aber **beide** Operanden in 4 Bit.

Drei Nebenbedingungen, die den Schritt teurer machen, als die Tabelle aussieht:

- **Beide Operanden müssen Integer sein.** `iu8` hat `a_neg`/`b_neg` getrennt (u8×s8 geht), aber kein
  fp16×int8. Für Q·K heißt das: **Q muss quantisiert werden.**
- **Die Fragmente sind kleiner.** Unser fp16-Fragment liest 8 Halfs = 128 Bit je Lane; `iu8` liest
  8 Bytes = **64 Bit** (ROCm-Doku; GPUOpen, „WMMA guide part 2"). AMD empfiehlt dort ausdrücklich,
  zwei WMMA zu einem K=32-Schritt zu fusionieren, damit wieder 128-Bit-Ladungen entstehen. Unser K
  ist heute 16 (`PA_KL`) — für int8 wäre der Doppel-K-Schritt also Pflicht, nicht Kür.
- **Das Lane-Mapping ist ein anderes als bei NVIDIA.** gfx12 ist *lane-group-split*: die beiden
  Lanes-Gruppen teilen sich die K-Achse (Gruppe 0: K {0–3, 8–11}, Gruppe 1: K {4–7, 12–15} bei fp16),
  **ohne Replikation**. Die K-Verschränkung ist für `iu8`/`iu4` **nicht dokumentiert** (ROCm-Issue
  #6025 nennt genau das; für `iu4` ist auch die Nibble-Reihenfolge offen). Wer den int8-Pfad baut,
  muss dieses Mapping zuerst mit einer Identitäts-Sonde empirisch festnageln — sonst sind die Tiles
  still falsch, und der Fehler taucht erst viele Kernel-Schritte später als NaN auf.

**rocWMMA hilft nicht weiter:** int8 ist auf gfx1200/1201 unterstützt, aber nur als 16×16×**16**
(BlockK=16, keine 32+). Es ist eine Fragment-Bibliothek über denselben Instruktionen — sie erspart
das Mapping nicht, sie versteckt es.

## 3. Der Preis, der bleibt: Descaling

Das ist der Punkt, an dem die 2× verschwinden — und er ist bei uns strukturell schlechter als in
einem GEMM.

Unser Block ist 32 Werte breit, also **8 Skalen je 256-dim-Head**. Ein Q·K-Ergebnis (eine Query ×
eine Zelle) summiert über 256 Dimensionen, also über **8 Blöcke mit 8 verschiedenen `d`**. Es gibt
zwei Wege, die Skala anzubringen, und sie kosten um Größenordnungen verschieden viel:

| | Skala sitzt auf | Arbeit je 16-Zeilen-Tile |
|---|---|---|
| **fp16 (heute)** | dem **Operanden** K, vor dem MMA: `code * d` → fp16 | 512 Werte × ~2 Ops = 1024 Ops, **einmal**, für alle 16 Query-Zeilen |
| **int8 (Vorschlag)** | dem **Akkumulator**, nach dem MMA: `int32 * (d · q_scale)` | 16 Zeilen × 16 Zellen × 8 Blöcke = 2048 Ops, **je Ausgabe**, nicht je Eingabe |

Die int8-Rechnung tauscht also 1 Op je Eingabewert (durch 16 Zeilen geteilt) gegen 1 Op je Ausgabe
je Block. **Rechnung** (keine Messung, mit den Raten aus §2 und 128 fp32-Ops/Zyklus/CU): fp16-Pfad
16 Zyklen MMA + 8 Zyklen Dequant = 24; int8-Pfad 8 Zyklen MMA + ~4–6 Zyklen Descaling/Konvertierung
= 12–14. Das wären **1,7–2,0×** — und genau hier widerspricht die Fremdmessung.

**Quelle** (zolotukhin.ai, 12.07.2026, RX 9070 XT, Q4_K-GEMM): mit int8-WMMA halbiert sich der Matmul,
aber „ein fp32-Descale je 32 Werte kostet dieselben Zyklen" → **1,20× statt 2×**; mit ganzzahligen
Sub-Skalen, weitem K und gepacktem Entpacken **1,61×**. Und: WMMA und Vektor-Instruktionen teilen sich
auf RDNA4 **einen** Issue-Port, sie können sich also nicht gegenseitig verdecken.

Die ehrliche Lesart: **die 2× kommen nie ganz an, und ob wir bei 1,2× oder 1,7× landen, ist eine
Messung an unserem Kern, keine Ableitung.** Was die Rechnung sicher sagt: der Descaling-Aufwand hängt
daran, ob die Skala auf dem Operanden oder dem Akkumulator sitzt — und bei einem 16-Zeilen-Tile ist
das ein Faktor 16 zugunsten des heutigen fp16-Wegs.

## 4. Was die Praxis mit 4-Bit-KV auf Matrix-Cores macht

Alle drei Lager lösen **nicht** das Descaling-Problem, sondern das Materialisierungs-Problem: sie
vermeiden die fp16-Kopie des Caches. Das ist bei uns schon so (wir dequantisieren im Register/smem),
also ist der Teil bereits erledigt — die Zahlen zeigen aber, wo die Gewinne herkommen.

| Ansatz | Was er tut | Zahl (aus der Quelle) |
|---|---|---|
| llama.cpp **PR #31802** | fusioniert q8_0/q4_0-Dequant **in** die `fattn-mma-f16`-Kerne, statt vorher fp16-Scratch zu schreiben | Decode +17,6 % (32k) … +79 % (171k); ~0,5–1 GiB/GPU Scratch gespart; Prompt innerhalb +2 % |
| **torad-labs/llama.cpp** `60feea0` | liest q4_0 nativ im Tensor-Core-FA: q4-Blöcke → smem → dort fp16 („same values, same math"), **Q als int8 mit einer Skala je Spalte**, Q·K als s8×u8-MMA, V weiter fp16 | Perplexität 4,5065 → 4,5074 (**+0,02 %**); ab 4.096 Kontext schneller als der Vektor-Kern; **braucht `--kv-mean-center`** für brauchbaren 4-Bit-K-Cache |
| **SageAttention** PR #368 (gfx12) | nativer RDNA4-Backend: **int8-WMMA für QK**, fp8-WMMA für PV | Diffusion ~**42 %** schneller; ISA enthält `v_wmma_i32_16x16x16_iu8` und `v_wmma_f32_16x16x16_fp8_fp8` |
| llama.cpp `0bf1eef` | **verwirft** dp4a auf RDNA4 (q8_1-Aktivierungsquantisierung + Durchsatz) | 101 t/s statt 135 t/s auf RX 9070 XT → RDNA4 geht auf den Skalar-Half-Kern zurück |

Drei Schlüsse daraus:

1. **Der int8-QK-Pfad ist auf RDNA4 der Stand der Technik** (SageAttention baut ihn genau so, und der
   llama.cpp-Fork zeigt, dass er mit 4-Bit-KV funktioniert). Der Qualitätspreis der Q-Quantisierung
   ist dort **+0,02 % Perplexität** — klein, aber es ist ein Preis, den wir heute nicht zahlen.
2. **`--kv-mean-center` taucht als Voraussetzung auf.** Ein fremder, unabhängiger Fork kommt aus der
   Praxis zu derselben Bedingung, die unsere Kalibrierung begründet: ohne Mean-Center ist 4-Bit-K auf
   langem Kontext nicht brauchbar. Das ist eine Stütze der These, kein Beweis.
3. **Nicht jede Integer-Instruktion ist ein Gewinn.** `sudot4` (`v_dot4_i32_iu8`) existiert auf RDNA4
   (1/clk/SIMD, gemischt signed/unsigned; `sdot4`/`udot4` sind CDNA-only) — und llama.cpp hat den
   dp4a-Weg auf genau dieser Karte **zurückgebaut**. Ein Dot-Produkt-Pfad ist keine Abkürzung.

## 5. Was das für „voll einsatzbereit und optimiert" heißt

Die Reihenfolge ergibt sich aus den Zahlen, nicht aus der Eleganz:

1. **Der Hänger ist die ganze Lücke.** Solange er offen ist, ist der Fast Path nicht Default-fähig,
   und keine Rate hilft. (B3, das Fenster, ist die zweite Funktionslücke — der Produktarm darf den
   Pfad nicht verlieren.)
2. **A2 entscheidet, ob überhaupt noch etwas fehlt.** B2 hat den Kern 5,12× schneller gemacht. Ob der
   Arm damit in die Nähe von int8 kommt, ist **ungemessen** — jede Optimierung davor wäre Raten.
3. **int8-WMMA ist der zweite Schritt, nicht der erste.** Obergrenze ~2×, realistisch 1,2–1,7×
   (Fremdmessung), Preis: Q in int8 (+0,02 % PPL beim Nachbarn, bei uns ungemessen), ein neues
   K-Mapping (undokumentiert für `iu8`), Doppel-K gegen die 64-Bit-Ladungen. Das ist ein eigenes
   Vorhaben mit eigener Messung — kein Anhängsel an B2.
4. **Was wir *nicht* brauchen:** eine LUT/Codebook-Stufe (die Level-Tabelle ist in A1 als
   Preistreiber ausgeschlossen: `q4lm32` 268,4 ≈ `q4u32` 263,7 tok/s) und Scratch-Vermeidung
   (machen wir schon).

**Was diese Notiz nicht sagt:** dass int8-WMMA sich für uns lohnt. Sie sagt, was es kostet, es
herauszufinden (eine Q-Quantisierung, ein Mapping-Probe, eine A/B-Messung), und dass der heutige
5,12×-Gewinn aus der **Kernel-Struktur** kommt, nicht aus der Matrix-Rate.

## 6. Quellen

- AMD, ROCm Optimization Guide: „RDNA4 dense WMMA builtins" (Raten, Signatur, `a_neg`/`b_neg`,
  Lane-Group-Split) — rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/…/rdna4-dense-wmma-builtins.html
- AMD, GPUOpen: „WMMA guide for AMD RDNA 4 GPUs, part 2" (K-major, 64-Bit-Ladungen für int8/fp8,
  Doppel-K-Empfehlung) — gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/
- ROCm Issue #6025: „Documentation Gap: WMMA Output Lane Mapping for gfx12" (Spalten-Verteilung,
  unbestätigte `iu4`-Nibble-Reihenfolge, unbestätigte K-Verschränkung für Integer)
- rocWMMA: Data-type support (i8 nur 16×16×16 auf gfx1200/1201), CHANGELOG 2.2.0
- ggml-org/llama.cpp PR #31802: fused q8_0/q4_0-Dequant in den MMA-FA-Kernen
- torad-labs/llama.cpp `60feea0`: q4_0 nativ + int8-Q im Tensor-Core-FA, `--kv-mean-center`
- thu-ml/SageAttention PR #368: nativer gfx12-Backend, int8-QK + fp8-PV
- ggml-org/llama.cpp `0bf1eef`: dp4a auf RDNA4 verworfen (101 vs 135 t/s)
- zolotukhin.ai, 12.07.2026: „int8 WMMA doubles RDNA4's matrix rate, and Q4_K's block scales take half
  of it back" (1,20× / 1,61×, Issue-Port-Teilen)
- AMD, „Integer dot product builtins" (Verfügbarkeit von `sudot4`/`sdot4` je Architektur)
- Eigene Messungen: A1 und B2, `bench/results/2026-11-14-q4u32-fastpath/README.md)

**Quellen des Updates vom 17.11.2026 (§8, §9):**

- AMD, ROCm Optimization Guide: „RDNA3 dense WMMA builtins" (gfx1100–gfx1103, gfx1150–gfx1153; f16/bf16/i8/i4,
  32/32/32/**16** Zyklen) — rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/…/rdna3-dense-wmma-builtins.html
- AMD, „Matrix Instruction Calculator" (der `rdna3`-Block enthält nur `v_wmma_*`: **kein MFMA auf gfx11**;
  `iu4` = 16 Zyklen) — github.com/ROCm/amd_matrix_instruction_calculator
- AMD, RDNA3 Shader ISA Reference (Feb 2023), Kapitel 7.9 „Wave Matrix Multiply Accumulate (WMMA)" — kein
  Matrix-Kapitel
- AMD, rocWMMA „Data type support" (fp8/BF8 nur gfx1200/gfx1201; int4 nicht in der öffentlichen API) und
  „Installation" (gfx1100/1101/1102 unterstützt)
- AMD-Blog 16.10.2026: „FP16 GEMM Optimization on AMD RDNA 3 and RDNA 4" (gemessen: W7800 70,92 gegen
  56,86 TFLOPS/s eigener WMMA-Kern gegen hipBLASLt; 8060S 40,63; 9070 XT 126,17) —
  rocm.blogs.amd.com/software-tools-optimization/rdna-fp16-gemm/
- ggml-org/llama.cpp PR #31234 (native quantisierte KV im Tile-Kern; R9700 q8_0 1,85× / q4_0 1,39×) mit der
  Kommentar-Messung vom 16.11.2026 auf gfx1151 (8060S, ROCm 7.2, Win11): Decode +3 … +6 %
- ggml-org/llama.cpp PR #30792 (natives q8_0 im Tile-Kern; gfx1151: Decode +19,0 % (16k) bis +61,9 % (63k))
- ggml-org/llama.cpp PR #31323, **gemergt 14.11.2026 22:36 UTC** (Vulkan-Flash-Attention für RDNA3 und RDNA4)
- ggml-org/llama.cpp PR #30738 (Vulkan: Q4_0-KV-Dequant + Transpose im FA)
- stew675/llama-cpp-rdna-boosts PR #199, 21.10.2026 (gfx1100, head-512-WMMA; 7900 XTX: Prompt +34 … +75 %;
  **KLD 0,0293 / Top-p 92,59 %** gegen den Tile-Prefill)
- vLLM Issue #52585, 17.08.2026 (gfx1100, `BLOCK_M=64`: 1,1–1,3× TTFT, 7900 XTX, bitidentische Ausgaben)
- zolotukhin.ai, 14.08.2026: „Sub-8-Bit KV Cache Quantization Is Not a Free Lever" (KIVI-Achse: K per-Channel,
  V per-Token)
- PrismML-Eng/llama.cpp PR #51/#53, Juli 2026 (KLD-Tabelle des Mean-Centerings; keine AMD-Messung)
- LLVM PR #164036, Okt. 2026: „Fix GFX11 WMMA intrinsic lowering regression" (vor dem Fix skalar, ~100 GFLOPS;
  nach dem Fix 10–16×)
- ROCm 10.1.0 (05.10.2026), rocWMMA 2.2.1 — rocm.docs.amd.com/en/latest/release/versions.html
- Eigene Messungen: A.16–A.23, `bench/results/2026-11-14-q4u32-fastpath/README.md`; 780M-Report
  `docs/RADEON_780M.md:82–105`

## 7. Laufzeitstand (17.11.2026): aus dem Hänger ist ein Default geworden

**Messung** (Windows, gfx1201, dieser Build; alle Zahlen aus
`bench/results/2026-11-14-q4u32-fastpath/README.md`):

| Arm | Ergebnis |
|---|---|
| A.16 | Das Fenster-Gate ist **rot**: der Prompt-Kern hing im echten Lauf. |
| A.17 | Der Fenster-Pass steht, das Gate ist **grün**: `FAILURES: 0`, Parität **5,7e-06 … 8,3e-06** gegen FP64 auf einer Skala von ~7,5 — so gut wie der Decode-Kern (4,7e-06 … 7,2e-06). Die Nicht-Fenster-Fälle bleiben **bitgleich**. |
| A.18 | Die Route stimmt: Matrix-Kern-Zeile in allen drei q4-Läufen, **16/16 Nadeln**. Die Prompt-Rate ist auf dieser Maschine **nicht auflösbar** (Zweimoduseffekt — der schnelle Modus fällt mit der Engine-Startzeit zusammen). |
| A.19 | Die Cache-Größe erklärt den Vorsprung **nicht**; im Motor sind die Arme gleich. |
| A.20 | `q4_0` kostet **5–7 %**, `int8` und `q4u32` sind gleichauf. |
| A.21 | Ohne den Schalter ist `q4u32` **2,5× langsamer**: 264,3 gegen ~663 tok/s (32K-Nadel, Median). |
| A.22 | Der gekoppelte Schalter trägt ohne Flag: **−1,8 %** gegen `int8` (Mittel), **−2,1 %** (Median). |
| A.23 | Ohne `--kv` wählt der Motor `q4u32` (**3/3** Starts), die Experten-Slots steigen **+18,7 %** (3428,7 gegen 2888,5), die Prompt-Rate sinkt **−0,7 %** (Median −0,1 %). |

**Was das für die drei Fragen aus §1 heißt:**

1. *Läuft er?* — **Ja**, mit Parität gegen FP64 und ohne Hänger (A.17, A.18).
2. *Was kostet er danach?* — **Nichts messbar**: gegen den int8-Kern −1,8 % (A.22), gegen den
   4-Bit-Pool ohne Kern 2,5× schneller (A.21).
3. *Wie schnell könnte er sein?* — **weiter offen**. Der 5,12×-Gewinn aus B2 kommt aus der
   **Kernel-Struktur** (ein Kern statt zweier), nicht aus der Matrix-Rate; §2/§3 sagen, wie viel die
   Matrix-Rate theoretisch trägt und warum das Descaling davon frisst.

**Die Lehre aus A.17, die über den 4-Bit-Pool hinausgeht:** der int8-Prompt-Kern fiel von **8,07 auf
17,4 ms je Chunk** (Faktor 2,16), weil ein 32-Register-Akkumulator nur unter einer Laufzeitbedingung
geschrieben wurde — die Ausgabe blieb bitgleich. **Die Occupancy entscheidet hier über den Faktor 2,
nicht die Rechenarbeit.** Jede weitere Kernvariante (gfx11, int8-WMMA, `iu4`) ist zuerst eine
Register- und Occupancy-Frage.

## 8. Literatur-Update (Recherche 17.11.2026): was seit dem 14.11. dazugekommen ist

**Quelle** (AMD, ROCm Optimization Guide, Seite „RDNA3 dense WMMA builtins"): **gfx11 hat WMMA.**
Die Seite nennt gfx1100–gfx1103 und gfx1150–gfx1153 mit den Fragment-Layouts und Zyklusraten:

| Builtin (wave32) | Ops | Zyklen |
|---|---:|---:|
| `wmma_f32_16x16x16_f16_w32` (unser Kern auf gfx11) | 8192 | 32 |
| `wmma_i32_16x16x16_iu8_w32` | 8192 | 32 |
| `wmma_i32_16x16x16_iu4_w32` | 8192 | **16** |

Das sind **AMD-Dokumentationsangaben**, keine Messung. Zwei Folgen: (a) die 2×-Rate für 4-Bit gilt auf
RDNA3 genauso wie auf RDNA4 — **ohne** fp8, das es dort nicht gibt (rocWMMA: „F8 and BF8 data type
support requires the gfx1200 or gfx1201"); (b) **MFMA gibt es auf gfx11 nicht** (Matrix Instruction
Calculator: der `rdna3`-Block enthält nur `v_wmma_*`; die RDNA3-ISA-Reference hat Kapitel 7.9 „WMMA"
und kein Matrix-Kapitel).

**Quelle** (AMD-Blog 16.10.2026, „FP16 GEMM Optimization on AMD RDNA 3 and RDNA 4"): **gemessen**,
FP16-GEMM M=N=K=8192, ROCm 10:

| Karte | eigener WMMA-Kern | hipBLASLt |
|---|---:|---:|
| Radeon PRO W7800 (gfx1100) | 15,50 ms → **70,92 TFLOPS/s** | 19,34 ms → 56,86 TFLOPS/s |
| Radeon 8060S (gfx1151) | 27,06 ms → 40,63 TFLOPS/s | 28,14 ms → 39,07 TFLOPS/s |
| RX 9070 XT (gfx1201) | 8,71 ms → 126,17 TFLOPS/s | 8,05 ms → 136,58 TFLOPS/s |

Zwei Lesarten daraus: auf RDNA3 ist ein **eigener** WMMA-Kern dem BLAS überlegen (+25 % auf gfx1100) —
und der Akkumulator ist auf gfx11 **doppelt so breit** wie auf gfx12, weshalb die Tiles dort kleiner
bleiben müssen. Beides betrifft einen möglichen gfx11-Arm, nicht unseren jetzigen.

**Quelle** (llama.cpp, Stand 17.11.2026, abgerufen per GitHub-API):

| PR | Stand | Inhalt | Zahl (aus der Quelle) |
|---|---|---|---|
| **#31234** | offen, aktiv 16.11. | native quantisierte + bf16 KV im Tile-Kern | R9700: q8_0 **1,85×**, q4_0 **1,39×**; eine Kommentar-Messung vom **16.11.2026** auf **gfx1151** (Radeon 8060S, ROCm 7.2, Win11): Decode q8_0 **+6,1 / +5,1 / +5,2 %** (8k/32k/64k), q4_0 **+4,4 / +3,0 / +3,3 %** |
| **#30792** | offen, aktiv 16.11. | natives q8_0 im Tile-Kern, Dequant in Registern | **gfx1151**: Decode **+19,0 %** (16k) bis **+61,9 %** (63k) für q8_0, +22,4/+77,1 % für f16-K+q8_0-V |
| **#31323** | **gemergt 14.11.2026 22:36 UTC** | Vulkan-Flash-Attention für RDNA3 **und** RDNA4 (neuer „flipped FA"-Shader: 16 Q-Zeilen je Subgroup, Ausgabe im coopmat-Akkumulator) | — |
| **#30738** | offen, aktiv 04.11. | Vulkan: Q4_0-KV-Dequant + Transpose im FA | Kommentar (7900 XTX, RADV 26.2.5): großer Prompt-Gewinn, Decode neutral |
| **#31802** | offen, aktiv 15.11. | fused q8_0/**q4_0**/f8_e4m3-Dequant im MMA-FA-Kern | CUDA: Decode +17,6 % (32k) … +79 % (171k), VRAM −496 MiB/GPU |

**Was das für uns heißt — und was nicht:**

* **Wir haben keine Konkurrenz im Kern.** **Quelle:** in **keinem** gefundenen Projekt liest ein
  4-Bit-KV-Cache **direkt** in ein Matrix-Core-Register — auch nicht auf gfx12. Upstream dequantisiert
  in f16 (in Registern oder in einen Scratch). Unser `q4u32` im Kern ist der unbesetzte Platz.
* **Der Decode-Gewinn auf RDNA3 ist kleiner als auf RDNA4.** Die gfx1151-Messung (+3 … +6 %) gegen die
  R9700-Zahl (1,39–1,85×) sagt: der Decode-Bottleneck bleibt die Bandbreite, nicht die Matrix-Einheit.
  Wer „4 Bit im KV" auf RDNA3 verspricht, verspricht **KV-Speicher**, nicht Tempo.
* **Neu im Upstream (14.11.):** der Vulkan-FA-Kern für RDNA3/RDNA4 (#31323) — der beste verfügbare
  gfx11-Referenzpunkt, und ein Warnsignal: er ist **Vulkan**, nicht HIP.
* **Quelle** (stew675/llama-cpp-rdna-boosts PR #199, 21.10.2026): eigene head-512-WMMA-Zeilen für
  gfx1100, **gemessen** auf 7900 XTX: Prompt +34 … +75 % (d32768: 1908 → 2549 tok/s; d65536:
  1209 → 1757). **Aber:** der WMMA-Prefill gegen den Tile-Prefill ergibt **KLD 0,0293** und gleiches
  Top-p **92,59 %** (Grenze war 0,005 / 98 %) — **die Prefill-Logits verschieben sich messbar**.
  Ein eigener Logit-Gate ist damit Pflicht, nicht Kür.
* **Quelle** (vLLM Issue #52585, 17.08.2026): auf gfx1100 `BLOCK_M=64` statt 16 → **1,1–1,3×** TTFT
  bei 8192 Token (7900 XTX), bitidentische Ausgaben. **Ohne** Matrix-Core. Die 2×-Lücke bei
  Triton-Unified-Attention ist also ein **Tile**-Problem, kein Matrix-Problem.
* **Quelle** (zolotukhin.ai, 14.08.2026): „Sub-8-Bit KV Cache Quantization Is Not a Free Lever" — unter
  8 Bit bricht die Qualität ein; die KIVI-Achse ist **K per-Channel, V per-Token**. Unser `q4u32`
  ist eine **per-Token**-Gruppierung (Skala je 32 Werte entlang head_dim) — genau die Achse, vor der
  die Arbeit warnt. Das ist die stärkste theoretische Rückmeldung auf unser Design und der nächste
  Prüfpunkt (§10).
* **Quelle** (PrismML-Eng/llama.cpp #51/#53, Juli 2026): die KLD-Tabelle des Mean-Centerings
  (Rotation + passende Basis **0,00111**, Rotation allein 0,00144, Centering allein 0,00149). Stand
  unverändert; **eine AMD-Messung dazu gibt es nicht** (nicht gefunden).
* **Was sich seit dem 14.11. im AMD-Umfeld nicht getan hat:** im ROCm Optimization Guide kein
  inhaltlicher Commit (letzte Änderung 18.09.2026), aktueller Stand ROCm 10.1.0 (05.10.2026),
  rocWMMA 2.2.1. **Achtung, LLVM PR #164036** (Okt. 2026): vor diesem Fix zerfiel `__builtin_amdgcn_wmma_*`
  auf gfx11 in hunderte skalare `v_fma_*` (~100 GFLOPS), nach dem Fix **10–16×** mehr. **Bevor wir
  gfx11-Zyklen messen, ist zu prüfen, ob unsere Toolchain den Fix enthält.**
* **Nicht gefunden:** ein gfx11-Backend für SageAttention (PR #368 ist weiter offen), eine AMD-Messung
  zu `kv-mean-center`, ein Projekt mit `iu4`-WMMA für einen KV-Cache, SGLang- oder Mamba/SSM-Arbeit zu
  RDNA3.

## 9. Die gfx11-Familie: wie viel wirklich fehlt

**Kurzantwort: der Code ist schon da. Was fehlt, ist die Voreinstellung — und die ist eine
Default-Änderung, die eine Messung auf einer gfx11-Karte braucht.**

*(Statische Analyse des Baums, 17.11.2026; Datei:Zeile-Verweise aus dem Repo, nichts gemessen.)*

**Belegt im Code:**

* `engine/src/kernels/cuda/qsa_prompt_attn.cu:1897–1907` — der WMMA-Kern hat eine **eigene gfx11-Variante**
  (`STRATA_PA_GFX11`), mit eigenem Builtin (`:1941–1950`: gfx11
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, gfx12 `…_w32_gfx12`) und eigenem Fragment-Layout
  (`:1908–1916`: gfx11 `PA_KL = 16`, `PA_KOFF(half) = 0`, `PA_ROW(i, half) = 2·i + half`; gfx12 `PA_KL = 8`,
  `PA_KOFF = half·8`, `PA_ROW = half·8 + i`).
* `engine/src/kernels/cuda/qsa_prompt_attn.cu:2503–2513` (`hip_wmma_usable`): `STRATA_HIP_WMMA=1` **schaltet
  auf gfx11 bereits alles frei** — inklusive des 4-Bit-Pools (`:2723–2729` `q4_prompt_on`,
  `:2812–2826` die 4-Bit-Routen). Für gfx1103 (780M) ist das sogar **Default**
  (`engine/src/core/arch_defaults.cpp:163`).
* **Gemessen (780M, gfx1103, Linux):** Prompt 1K **98,5 → 157,3 tok/s**, 4K **115,1 → 233,3 tok/s**
  (`docs/RADEON_780M.md:82–105`) — der Matrix-Core-Prompt-Pfad auf gfx11 funktioniert und wirkt.
* `engine/src/kernels/qsa_prompt_attn_parity.cpp:460–479` — der Paritätstest **akzeptiert gfx11 und gfx12**
  (`:466–469`), setzt `STRATA_HIP_WMMA=1` und `STRATA_KV_Q4_PROMPT=1` und fährt den **4-Bit-Fall**
  (`:487`). Der Nachweis „4-Bit rechnet auf gfx11 richtig" ist also **ein Lauf** auf einer
  gfx1100/1101/1102-Karte, kein neues Testprogramm.

**Was fehlt (und wie schwer es ist):**

| Punkt | Datei:Zeile | Art |
|---|---|---|
| Der Motor darf auf gfx11 **selbst entscheiden** (KV-Format `q4u32` statt `fp16`) | `engine/src/program/generate.cpp:2547` (`kv_default_mode`) | **mechanisch**, aber eine **Default-Änderung** → braucht Messung |
| Das Terminal darf auf gfx11 das Format **nicht** festnageln | `terminal/strata_control/kv_default.py:19` (`GFX12_ARCHS`), `setup.py:11177` | mechanisch, folgt der Engine-Entscheidung |
| `STRATA_HIP_WMMA=auto` auf gfx11 | `engine/src/kernels/cuda/qsa_prompt_attn.cu:2497–2502` (`hip_wmma_auto_pick`: `if (arch != 2) return false;`) | mechanisch; die **Schwelle** (64 Queries) ist auf gfx1201 gemessen → **Messung** |
| Die gfx12-Schnellschalter-Tabelle | `engine/src/core/arch_defaults.cpp:27–62` | optional; die Schalter sind „exakt" behauptet → **ohne Messung nicht setzbar** |

**Statische Nachrechnung des 4-Bit-Pfades auf gfx11** (die Code-Lesung, kein Messwert): der
32-Wert-Block trägt 16 Byte Nibbles (Wert *j* im niedrigen, *j+16* im hohen Nibble). Mit `PA_KOFF = 0`
und `PA_KL = 16` liegt jeder 16-Dim-Schritt **innerhalb eines** 32-Wert-Blocks (16 teilt 32), die
Nibble-Expansion passt also auch auf gfx11. Der Fenster-Pfad liest entsprechend 32 statt 16 Byte
(`h8_win`, `:1959–1965`). **Statisch konsistent, nicht gemessen.**

**Was statisch nicht ausgeschlossen bleibt:** Fragment-Layout (ein Fehler fällt nicht auf, er rechnet
falsch), andere Rundung der inneren Summe (auf gfx11 über 16 k, auf gfx12 über zwei 8-k-Hälften),
**Registerdruck** (`PA_KL = 16` ⇒ das A-Fragment kostet 8 statt 4 VGPRs, `qh[4] + ql[4]` 64 statt 32 —
Spill wäre ein Tempo-, kein Korrektheitsproblem, cf. A.17) und die Frage, ob der 4-Bit-Pfad auf
gfx11 **langsamer** wäre als int8 (auf gfx12 war er ohne Matrix-Kern 2,5× langsamer).

**Zwei Doku-Fehler, die die Analyse gefunden hat (und die wir gleich richtigstellen):**
`docs/AMD_HIP.md:162` behauptet, der Matrix-Core-Pfad sei auf Windows **opt-in** — er ist seit dem
gfx12-Default (A.17–A.23) **an**; `cmake/strata_gpu_engine.cmake:805` sagt, der Paritätstest
„skips off gfx12" — er überspringt erst ab gfx1030.

## 9a. Der APU-Fall: Strix Halo (gfx1151) und die 780M (gfx1103)

**Kurzantwort: Auf den APU-Karten ist der Matrix-Core-Prompt-Pfad schon an — es fehlt nur der
4-Bit-KV-Pool. Und für „ältere gfx11-Karten" (7900 XTX/7800 XT/7600 XT) ist es derselbe Fall wie
A.24: der Pfad läuft, wenn man ihn anfordert.**

**Belegt und gemessen (der APU nimmt das Matrix-Set schon per Default):**
`engine/src/core/arch_defaults.cpp:129–137` und `:172–191` — auf Windows bekommt **gfx1151 (Strix Halo)** per
Default `STRATA_HIP_WMMA=1`, `PF_GEMM`, `PA_FAST`, `HC_UPMIX`, `PF_FUSED`, `SELECT_WMMA`, `PF_F16`,
`PF_DM16`. **Gemessen auf dem Strix-Box** (Qwen3.8-Flash-Next IQ3_S, Engine 0.1.52, Prompts von 4.095
und 8.191 Token, verschränkte Paare, eine Engine zur Zeit): **4K 690 → 1.090 tok/s** und **8K
785 → 1.183 tok/s** (×1,58 / ×1,51); der erste Token KL **0,0088** (erlaubt 0,0228), Top-1 überall
gleich. Der 780M (gfx1103, Linux) nimmt dasselbe Set (`arch_defaults.cpp:160–168`): **109,5 → 214,7
tok/s** (`docs/RADEON_780M.md:82–105`). Auch der Block-Scorer läuft dort auf Matrix-Cores
(`SELECT_WMMA`: ×1,008 … ×1,077 bei 8K … 64K).

→ **Auf den iGPUs ist die offene Frage nicht „geht der Matrix-Core?", sondern „was macht der 4-Bit-Pool
dort?"** Das ist eine andere Frage als auf einer dGPU mit eigenem VRAM, und sie hat eine eigene
Antwort (unten).

**Was q4u32 auf einem APU anders macht (statische Analyse; die Zahlen sind gerechnet, nicht gemessen):**

1. **Der Speicher ist geteilt (UMA): KV und Experten konkurrieren um denselben Pool.** Der Gewinn
   „KV-Platz frei → mehr Experten-Slots" (auf gfx12: **+18,7 %**, A.23) fällt auf einem APU **größer**
   aus, nicht kleiner — dort ist der Engpass die Arena, nicht die Karte: auf der Strix-Box sind
   **24.576 Slots** gemessen, weil die Arena *nicht* beim Treiber registriert wird
   (`arch_defaults.cpp:143–147`; mit Registrierung: 0 Slots und der Start bricht ab).
2. **Ab 64K Kontext streamt der KV** (`setup_config.py:329–330`: `--kv-resident 32768`; der KV liegt im
   RAM, nur 32K bleiben im Pool). Dann spart 4 Bit **RAM** statt Pool: **13,7 → 7,5 KB je Token**
   (`setup_config.py:299`) und etwas am residenten Fenster. Der Slot-Gewinn schrumpft entsprechend —
   **zwei getrennte Messungen** (unter 64K ohne Streaming, über 64K mit Streaming).
3. **Der APU hat eigene, gemessene Defaults** (PF_FUSED, PF_F16, PF_DM16, SELECT_WMMA, ein eigener
   Prompt-Chunk 4096 statt 6144, `STRATA_GFX1151_CHUNK`, `generate.cpp:3911–3926`). Die Kombination
   „q4u32 × diese Schalter" ist **ungemessen** — genau die Art Wechselwirkung, die A.17 (Register und
   Occupancy) und A.18 (Zweimoduseffekt) beide gezeigt haben.
4. **Der Decode liest den 4-Bit-Pool über einen anderen Pfad.** Auf gfx12 ist er **ohne** Matrix-Kern
   **2,5× langsamer** (A.21: 264 gegen 663 tok/s). Auf einem APU ist der Decode ohnehin
   bandbreitenlimitiert; ob der 4-Bit-Pool dort gewinnt oder verliert, ist **offen**.

**Was heute schon geht (ohne jede Code-Änderung):** auf einer Strix-Halo-Box `--kv q4u32` setzen —
der Matrix-Kern ist per Default an, der 4-Bit-Pool ist der einzige Schalter. **Ungemessen** auf dieser
Hardware (die Parität ist auf gfx1201 gemessen, A.17), aber der Weg ist derselbe wie der int8-Pfad.

**Die Antwort auf die Frage in der Kürze:** Ja — dieselbe Engine, dasselbe 4-Bit-Format, und der Motor
entscheidet je Karte (die Logik aus A.23). Auf den älteren dGPUs (gfx1100/1101/1102) ist das der Fall
von **A.24**; auf den APU-Karten (gfx1103/gfx1150/gfx1151) der Fall von **A.25**. Beide brauchen eine
Messung auf der jeweiligen Karte, sonst bleibt es bei statischer Analyse.

## 9b. Rezept: was für gfx11 (und die APUs) noch anzupassen ist

**Kurzfassung: am Kernel ist nichts zu tun. Es sind drei Schalter, die in dieser Reihenfolge fallen
müssen — und jeder ist eine Default-Änderung, also messpflichtig.**

| # | Schalter | Wo | Art | Was fehlt |
|---|---|---|---|---|
| **S1** | **WMMA auf gfx11-dGPUs** (7900 XTX, 7800 XT, 7600 XT) | `engine/src/kernels/cuda/qsa_prompt_attn.cu:2503–2513` (`hip_wmma_usable`, unset-Fall `arch == 2`), `:2497–2502` (`hip_wmma_auto_pick`: `if (arch != 2) return false;`), `engine/src/core/arch_defaults.cpp:151–193` (die iGPU-Tabelle) | Default-Änderung | Auf gfx11-dGPUs ist WMMA **aus**, solange niemand `STRATA_HIP_WMMA=1` setzt; die iGPU-Tabelle deckt nur **integrierte** Karten (`arch_defaults.cpp:153`). `=auto` ist auf gfx11 wirkungslos (Zeile 2498), und die Schwelle (64 Queries) ist eine **gfx1201**-Messung. |
| **S2** | **Der 4-Bit-Pool auf gfx11** | `engine/src/program/generate.cpp:2547` (`kv_default_mode`), `terminal/strata_control/kv_default.py` (Arch-Liste), `setup.py:11175` (`_gfx12`) | Default-Änderung | Der Pool läuft nur über den Matrix-Kern. **Ohne S1 ist er auf gfx11 der langsame Pfad** (auf gfx12 2,5× langsamer, A.21) — der 4-Bit-Default darf also **erst** nach S1 kommen. |
| **S3** | **Der KV-Default auf gfx11** | dieselben drei Stellen | Default-Änderung | Der Motor darf dort selbst wählen; das Terminal darf das Format **nicht** festnageln. |

**Was **nicht** angepasst werden muss (das ist die gute Nachricht):**

* **Der WMMA-Kern** hat die gfx11-Variante schon (`qsa_prompt_attn.cu:1897–1916`, `:1941–1950`), der
  4-Bit-Pfad ist arch-generisch, die Nibble-Expansion passt auf gfx11 statisch (16 teilt 32).
* **Der Build**: `STRATA_PA_GFX11` entsteht aus den `__gfx11xx__`-Makros; die Windows-rdna3-Pakete
  (gfx1100/1101/1102/1151) gibt es schon (`docs/AMD_HIP.md:57–63`).
* **Der Paritätstest**: `hip_prompt_attn_wmma` akzeptiert gfx11 und fährt den 4-Bit-Fall
  (`engine/src/kernels/qsa_prompt_attn_parity.cpp:466–469`, `:487`) — **ein Lauf** auf der Zielkarte liefert
  Parität *und* Timing.

**Die Regel, die man beim Umbau einhalten muss (sonst wird es still falsch):** Die Bedingung im
Motor muss **genau** die Bedingung sein, unter der der Matrix-Kern wirklich läuft — also
`hip_wmma_usable()`. Deren drei Fälle (`:2503–2513`): `=0` → aus; `=1` → an (das ist auch der Weg,
über den die iGPU-Defaults den gfx1151/gfx1103 einschalten); `=auto` → nur gfx12 heute. Der
`unset`-Fall ist der gfx12-Default (Zeile 2512). Ein `kv_default_mode()`, das die Karte prüft, aber
nicht die Env, würde auf einem `STRATA_HIP_WMMA=0` den 4-Bit-Pool wählen — **genau das, was der
gemessene Aus-Schalter in A.21 zeigt** (264 gegen 663 tok/s).

**Die APUs (gfx1103/gfx1150/gfx1151) brauchen zusätzlich drei Dinge, die auf einer dGPU nicht
vorkommen:**

1. **Die Arena ändert sich mit.** Auf der Strix-Box sind **24.576 Slots** gemessen, weil die Arena
   *nicht* beim Treiber registriert wird (`arch_defaults.cpp:143–147`; mit Registrierung: 0 Slots und
   der Start bricht ab). Diese Zahl ist mit dem bisherigen KV-Maß gemessen — ein kleinerer KV
   vergrößert die Arena und damit den Slot-Zähler. **Neu messen**, nicht hochrechnen.
2. **Ab 64K streamt der KV** (`setup_config.py:329–330`: `--kv-resident 32768`). Dann spart 4 Bit
   **RAM** (13,7 → 7,5 KB je Token, `setup_config.py:299`) und nur den residenten Fensterteil — der
   Slot-Gewinn schrumpft. Zwei Messungen: unter 64K (ohne Streaming) und über 64K (mit Streaming).
3. **Die APU-Defaults sind eigene, gemessene Schalter** (`PF_FUSED`, `PF_F16`, `PF_DM16`,
   `SELECT_WMMA`, ein eigener Prompt-Chunk 4096 statt 6144, `generate.cpp:3911–3926`). Ihre
   Kombination mit dem 4-Bit-Pool ist **ungemessen** — dieselbe Art Wechselwirkung wie Register und
   Occupancy in A.17.

**Und die Reihenfolge, in der gemessen wird:** erst der Paritätslauf (der billigste, er beweist, dass
der 4-Bit-Pfad auf der Karte rechnet), dann der Nadel-Lauf mit und ohne Matrix-Kern, dann das
**Logit-Gate** (KL ≤ 0,005, Top-p ≥ 98 %), dann die Speicher- und Slotzahlen, zuletzt der
Default-Wechsel. Vorregistriert als **A.24** (dGPU) und **A.25** (APU) im
`bench/results/2026-11-14-q4u32-fastpath/README.md`.



1. **Die Architekturentscheidung hält.** Der 4-Bit-Pool im Matrix-Kern ist der unbesetzte Platz
   (§8), und die eigene Messung sagt: kostet nichts gegen int8 (−1,8 %, A.22) und spart den
   KV-Speicher für den Experten-Cache (+18,7 % Slots, A.23).
2. **Der nächste Hebel ist nicht int8-WMMA, sondern die Skalierung.** Die Literatur warnt
   (zolotukhin.ai, 14.08.2026): unter 8 Bit bricht die Qualität ein, und die empfohlene Achse ist
   **K per-Channel**. Unsere Skala sitzt **je 32 Werte entlang head_dim** (per-Token). Das ist eine
   Frage an die Geometrie, nicht an den Kernel — und sie ist **billig zu prüfen** (Perplexität/KL auf
   einem festen Prompt, ohne Engine-Änderung).
3. **Die gfx11-Erweiterung ist leicht und rückwärtskompatibel** — der Code ist da, es fehlen die
   Voreinstellung und `=auto`. Sie ist aber eine **Default-Änderung** und damit nach
   `CONTRIBUTING.md` nur mit Messung zulässig. **Vorregistriert als A.24** (siehe
   `bench/results/2026-11-14-q4u32-fastpath/README.md`): ein Paritätslauf auf einer gfx11-Karte
   (`hip_prompt_attn_wmma`, der 4-Bit-Fall), dann ein Nadel-Lauf mit und ohne Matrix-Kern. **Ohne eine
   gfx11-Karte hier ist das offen, nicht erledigt.**
4. **`iu4`-WMMA ist der interessanteste neue Hebel** (16 Zyklen auf gfx11 und gfx12 statt 32 für
   f16/int8) — aber er verlangt, dass **beide** Operanden 4 Bit sind, also auch die Query, und die
   K-Verschränkung für Integer ist auf gfx12 bis heute **undokumentiert** (ROCm Issue #6025). Auf
   gfx11 kommt die **Layout-Regel** „matrix replication" dazu (Lanes 0–15 und 16–31 tragen
   identische A/B-Fragmente). Das ist ein eigenes Vorhaben mit eigener Messung — kein Anhängsel.

