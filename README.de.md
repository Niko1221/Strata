<h1 align="center">Strata</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · **Deutsch** · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>Ein KI-Modell mit 125 Milliarden Parametern auf deinem eigenen Gaming-PC</b><br>
NVIDIA- oder AMD-Grafikkarte (ab 12 GB) · Windows oder Linux · kostenlos und Open Source</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="Ein Voxel-Pagodengarten, den das Modell in Strata geschrieben hat, läuft im Browser"></a><br>
<sub>Ein Voxel-Pagodengarten aus einem einzigen Prompt, auf einer RTX 5070 mit Strata (IQ3_S, 128K Kontext) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">ganzes Video (49 s)</a></sub></p>

Strata lässt **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** auf einem normalen PC laufen.
Das ist ein großes, kluges KI-Modell, das sonst meist einen Server braucht. Es chattet, schreibt Code, versteht
Bilder und arbeitet mit deinen Apps und Coding-Agents zusammen. Nichts verlässt deinen PC.

## Wie schnell ist es?

Wir haben es auf zwei ganz normalen Gaming-PCs gemessen. Ein Token ist etwa ¾ eines Wortes.

- **Schreibt Antworten:** wie schnell die Antwort in einem kurzen Chat erscheint. 60 Tokens pro Sekunde sind
  schneller, als du lesen kannst.
- **Liest deinen Prompt:** wie schnell es aufnimmt, was du schickst (hier ein Dokument, Code oder Chatverlauf mit
  32K Tokens).

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>
<tr><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **Q2_0** | 94 Tokens/s | 2,650 Tokens/s |
| **IQ2_XS** | 79 Tokens/s | 2,090 Tokens/s |
| **IQ3_XXS** | 62 Tokens/s | 1,750 Tokens/s |
| **IQ3_S** | 53 Tokens/s | 1,620 Tokens/s |
| **Coder** | 55 Tokens/s | 2,180 Tokens/s |

</td><td>

| Größe | Schreibt Antworten | Liest deinen Prompt |
| --- | ---: | ---: |
| **Q2_0** | 60 Tokens/s | 1,160 Tokens/s |
| **IQ2_XS** | 52 Tokens/s | 1,110 Tokens/s |
| **Coder** | 44 Tokens/s | 1,420 Tokens/s |

</td></tr>
</table>

NVIDIA: Q2_0 mit Engine 0.1.36, die anderen Zeilen mit 0.1.26 (4K Antworten, 32K Prompts). Die vollständigen
Tabellen stehen in [DETAILS.md](docs/DETAILS.md#speed-measured). Eine Karte mit mehr VRAM ist schneller: Eine
RTX 3090 (24 GB) sollte etwa 100-140 Tokens pro Sekunde schreiben. Lange Chats und andere Karten:
[Tempo jedes Modells](docs/MODELS.md#how-fast-is-each-size), [Ergebnisse aus der Community](docs/COMMUNITY_BENCHMARKS.md).

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>Strata ist kostenlos. Wenn es auf deinem PC gut läuft, hält ein Kaffee die Arbeit daran am Laufen.</sub></p>

## Was du brauchst

| | |
| --- | --- |
| **Grafikkarte** | **NVIDIA** GeForce RTX 20, 30, 40 oder 50 Serie, oder **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 oder RX 6800 / 6900 Serie. Sie braucht **12 GB VRAM oder mehr**. |
| **RAM** | 32 GB oder mehr. Dein RAM entscheidet, [welches Modell](#welches-modell-soll-ich-nehmen) passt. Mit 64 GB laufen alle Größen. |
| **Festplatte** | Etwa 80 GB frei. Nimm wenn möglich eine SSD: Der erste Start geht dann viel schneller. |
| **System** | Windows 10 / 11 oder Linux und ein aktueller Grafiktreiber von NVIDIA oder AMD. |

Alles andere richtet der Installer ein. Zwei oder drei Karten können sich das Modell teilen ([Multi-GPU](docs/MULTI_GPU.md)).

Experimentell, von Community-Mitgliedern auf ihren eigenen Rechnern geschrieben und getestet:

- **Ältere Grafikkarten** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [Ältere GPUs](docs/OLDER_GPUS.md).
- **Intel Arc**, unter Linux aus dem Quellcode gebaut: [Intel Arc](docs/INTEL_ARC.md).
- **AMD Ryzen AI Max (Strix Halo)**, unter Linux aus dem Quellcode gebaut: [Strix Halo](docs/STRIX_HALO.md).
- **Ältere Prozessoren ohne AVX2**: Sie funktionieren, aber langsam. [Ältere CPUs](docs/INSTALL.md#older-cpus-experimental).

Die vollständige Liste: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Installieren

### Lass deine KI es einrichten

Nutzt du einen KI-Coding-Assistenten (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Füge dort diesen Text ein:

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

Er prüft deine Grafikkarte, deinen RAM und deine Festplatte und wählt das passende Modell. Dann installiert und
startet er es und sagt dir, wie du deine Apps verbindest. KI-Tools können Strata auch über seinen
[MCP-Server](docs/MCP_SERVER.md) installieren, starten und stoppen.

### Oder mach es selbst

[Lade Strata herunter](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) und entpacke es (oder nutze
`git clone`). **Windows:** Doppelklick auf **`START-HERE.bat`**. **Linux:** Führe im Strata-Ordner **`./setup.sh`** aus.

Die Schritte sind für NVIDIA und AMD gleich. Der Installer erkennt deine Karte und richtet die passende Engine
dafür ein. Er stellt dir ein paar Fragen:

- welches Modell und welche Größe,
- wie viel Kontext (wie viel Text sich das Modell merkt),
- ob es Bilder lesen soll.

Drück jedes Mal Enter, um die empfohlene Antwort zu nehmen. Dann lädt er das Modell herunter (etwa 70 GB) und
startet es. Bricht der Download ab, starte ihn einfach neu: Er macht dort weiter, wo er aufgehört hat. Dein Browser
öffnet die Strata-App unter `http://127.0.0.1:8080`.

> **Während das Modell startet, kann dein PC 1-3 Minuten langsam sein oder nicht reagieren** (beim ersten Mal am
> längsten). Strata lädt 35-55 GB in deinen RAM und reserviert einen Teil davon für die Grafikkarte. Das ist normal.
> Warte und schließ das Fenster nicht. Im Fenster siehst du, was Strata gerade tut.

**Beim nächsten Mal** startest du wieder `START-HERE.bat` (oder `./setup.sh`). Es startet sofort und lädt nichts
doppelt herunter. Schließ das Fenster, um das Modell zu stoppen. `UPDATE.bat` (`./update.sh`) aktualisiert Strata,
ohne es zu starten. Updates, Docker, mehrere Karten, wo die Dateien landen und alle Optionen:
[docs/INSTALL.md](docs/INSTALL.md).

## Welches Modell soll ich nehmen?

Der Installer empfiehlt eins passend zu deinem RAM. Dasselbe Modell gibt es in mehreren Größen, mehr oder weniger
stark komprimiert. Kleinere Größen sind schneller. Größere sind etwas klüger.

| Dein RAM | Nimm | Warum |
| --- | --- | --- |
| **32 GB** | **Coder** | passt in 32 GB und ist für Code gemacht (mit einer 24-GB-Karte laufen auch Q2_0 und IQ2_XS) |
| **48 GB** | **IQ2_XS** (oder Q2_0, das schnellste) | die größeren Größen passen nicht |
| **64 GB** | **IQ2_XS** (empfohlen) oder IQ3_XXS / IQ3_S | alle Größen passen; IQ3_S ist das beste und das langsamste |
| **96 GB oder mehr** | **IQ3_S** oder Unsloths UD-IQ4_XS (~4 Bit) | Platz für die größten Größen, auch wenn alles andere offen ist |

- **[Coder](docs/MODELS.md#coder):** eine Version für Code, bei der die Hälfte der Experten entfernt wurde. Sie
  erreicht 91 % des SWE-bench-Verified-Werts des vollen Modells (von ihren Autoren gemessen) und passt in 32 GB RAM.
  Außerhalb von Code ist sie schwächer, auch bei chinesischem und anderem CJK-Text (#438). Nimm dafür Q2_0, IQ2_XS
  oder IQ3_S, die alle Experten behalten.
- **[Swift 1.5](docs/MODELS.md#swift-15):** ein Fine-Tune, das viel kürzer nachdenkt, bevor es antwortet. Du
  bekommst die Antwort früher, bei etwa gleicher Qualität.
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs):** die ~4-Bit-Version von Unsloth, in der Qualität
  zwischen IQ3_S und UD-Q4_K_XL. 94 GB Download. Mit weniger als ~80 GB RAM liest Strata einen Teil davon während
  der Antwort von der SSD, dort ist es also langsamer (eine NVMe-SSD hilft).
- **[Unsloth UD-Q4_K_XL](docs/MODELS.md#unsloth-ud-q4_k_xl-experimental)** (experimentell): am nächsten am vollen
  Modell. Aber Strata liest das meiste davon während der Antwort von der SSD, deshalb schreibt es auf einem PC mit
  64 GB nur 7-8.5 Tokens/s.
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs):** Das richtest du von Hand
  ein. Es steht nicht im Menü des Installers.

Größen, Downloads und was wohin passt: [docs/MODELS.md](docs/MODELS.md). Um später ein weiteres Modell
hinzuzufügen, starte `SETUP.bat` (Linux: `./setup.sh --setup`).

## So benutzt du es

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="Der Monitor-Tab der Strata-App neben einem Coding-Agent"><br>
<sub>Der <b>Monitor</b> der Strata-App (links), während ein Coding-Agent den Pagodengarten aus dem Video schreibt (rechts)</sub></p>

- **Im Browser:** Öffne `http://127.0.0.1:8080`. Dort gibt es **Chat**, einen Live-**Monitor** für das Modell und
  deine GPU/CPU/RAM und **About** mit den Einstellungen und Adressen.
- **Deine Apps und Coding-Agents:** Füge einen „OpenAI-kompatiblen“ Anbieter mit der Basis-URL
  **`http://127.0.0.1:8080/v1`** hinzu. Jeder API-Key und jeder Modellname funktioniert.
  - Apps, die die API von Anthropic nutzen: `http://127.0.0.1:8080/v1/messages` (Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
  - Codex CLI und andere Apps, die die OpenAI Responses API nutzen: `/v1/responses`
    ([Einrichtung](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Nachdenken:** Wähle **aus, niedrig, mittel oder hoch** im Chat-Menü oder unter „Reasoning Effort“ in deiner
  App. Aus ist am schnellsten. Hoch ist am besten für schwierige Fragen.
- **Bilder:** Antworte im Setup mit Ja auf „Images?“. Klick dann im Chat auf **Picture** oder hänge Bilder in deiner
  App an. AMD-Karten lesen Bilder unter Linux über den Prozessor; unter Windows geht das noch nicht.
- **Vom Handy oder einem anderen PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Setz immer einen
  Key.
- **Eine Anfrage nach der anderen:** Standardmäßig beantwortet Strata eine Anfrage, die anderen warten. Um mehrere
  gleichzeitig zu beantworten, setz `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)). Auf einer 12-GB-Karte wird
  dadurch jede Antwort langsamer.
- **Lange Prompts:** Strata liest die erste Nachricht eines Chats komplett, etwa 1 Minute pro 30,000 Tokens.
  Folgenachrichten starten in Sekunden.

Mehr: [wo deine Chats gespeichert werden](docs/INSTALL.md#where-things-are-stored), [die API](docs/DETAILS.md#using-it).

## Etwas ist schiefgelaufen?

- **Mein PC ist beim ersten Start von Strata eingefroren.** Das ist normal, während das Modell lädt. Warte und
  schließ das Fenster nicht. Nach 10 Minuten immer noch eingefroren? Starte den PC neu, schließ andere Programme und
  versuch es noch einmal, oder nimm eine kleinere Größe.
- **Es hat beim Herunterladen oder Installieren aufgehört.** Starte `START-HERE.bat` (oder `./setup.sh`) noch
  einmal. Es macht dort weiter, wo es aufgehört hat.
- **Es ist sehr langsam und die Festplattenlampe blinkt ständig, oder es meldet „the engine stopped
  unexpectedly“.** Dein PC hat nicht genug freien RAM. Schließ andere Programme (Browser brauchen viel) oder nimm
  eine kleinere Größe (Q2_0 oder IQ2_XS).
- **Es meldet, dass Port 8080 schon belegt ist.** Strata läuft bereits. Such nach seinem Fenster.

Weitere Probleme und ihre Lösungen: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Kommst du nicht weiter?
Öffne ein [Issue](https://github.com/Niko1221/Strata/issues) und hänge `strata-<model>.log` aus dem Strata-Ordner
an. Du hast ein Sicherheitsproblem gefunden? Melde es vertraulich: [SECURITY.md](SECURITY.md).

## Wie funktioniert es?

Modelle wie dieses laufen normalerweise auf Servern mit Hunderten Gigabyte Grafikspeicher. Deine Grafikkarte hat
12-24 GB. Strata bringt das Modell trotzdem unter, indem es **die Arbeit auf deinen ganzen PC verteilt**. Stell dir
eine Küche vor: Was du ständig brauchst, bleibt auf der Arbeitsfläche, der Rest wartet in der Vorratskammer.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="Die 24,576 Experten des Modells: die meistgenutzten auf der Grafikkarte, alle im RAM, eine Nachschlagetabelle auf der SSD"></p>

- **Das Modell ist ein Team aus 24,576 kleinen Spezialisten („Experten“).** Jedes Wort braucht nur 10 davon.
- **Deine Grafikkarte** hält die paar tausend Experten, die am häufigsten gebraucht werden. **Dein RAM** hält alle,
  und **dein Prozessor** rechnet gleichzeitig am Rest. **Deine SSD** hält eine große Nachschlagetabelle.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="Ein kleiner Helfer rät die nächsten Wörter; das große Modell prüft alle auf einmal und behält die richtigen"></p>

- **Raten, dann prüfen:** Ein kleiner Helfer rät die nächsten paar Wörter. Das große Modell prüft sie alle auf
  einmal. Du bekommst dieselbe Antwort, nur 1.6-1.8x schneller.
- **Lange Texte werden in großen Stücken gelesen** (bis zu 8,192 Tokens auf einmal), mit über 1,000 Tokens pro
  Sekunde.

Die längere Erklärung: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Jeder Teil und seine Zahlen:
[die Details](docs/DETAILS.md#how-it-works) und das [Paper](docs/paper/Strata-Paper.pdf).

## Danksagung und Lizenz

Das Modell ist [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) vom Qwen-Team. Komprimiert
wurde es von [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5)
und Unsloth. Strata nutzt Teile von [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Alle Danksagungen:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata ist Open Source unter der [MIT-Lizenz](LICENSE). Ein
paar Teile und alle Modelle haben eigene Lizenzen ([welche](docs/HOW_IT_WORKS.md#license)).

## Strata unterstützen

Strata ist kostenlos und Open Source. Wenn es dir nützt, kannst du die Entwicklung unterstützen:

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>
