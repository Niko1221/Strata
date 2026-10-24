<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="380">
  </picture>
</p>

<p align="center"><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <b>Deutsch</b> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a></p>

<h3 align="center">Ein Spitzenmodell auf deinem eigenen Gaming-PC</h3>

<p align="center">Unterstützt NVIDIA, AMD, Intel · Windows, Linux, Mac · kostenlos und Open Source</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?include_prereleases&label=release" alt="Neueste Version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT-Lizenz"></a>
  <a href="https://github.com/Niko1221/Strata/stargazers"><img src="https://img.shields.io/github/stars/Niko1221/Strata?style=flat" alt="GitHub-Sterne"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?logo=buymeacoffee&logoColor=black" alt="Spendier mir einen Kaffee"></a>
</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="Ein Voxel-Pagodengarten, den das Modell von Strata geschrieben hat, im Browser"></a><br>
<sub>Ein Voxel-Pagodengarten aus einem einzigen Prompt, auf einer RTX 5070 mit Strata (IQ3_S, 128K Kontext) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">ganzes Video (49 s)</a></sub></p>

<p align="center">
  <a href="#installation"><b>Installation</b></a> ·
  <a href="#modelle">Modelle</a> ·
  <a href="#wie-schnell-ist-es">Tempo</a> ·
  <a href="#benutzung">Benutzung</a> ·
  <a href="#wie-funktioniert-das">Wie es funktioniert</a> ·
  <a href="#etwas-ist-schiefgegangen">Hilfe</a> ·
  <a href="#strata-unterstützen">Unterstützen</a>
</p>

Strata lässt **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** laufen, ein Modell mit 125
Milliarden Parametern, für das man sonst einen Server braucht, auf einem normalen PC. Ab 0.2.4 läuft im Terminal auch
**DeepSeek V4.1-Flash**. Es chattet, schreibt Code, liest Bilder und arbeitet mit deinen Apps und Coding-Agents zusammen.
Deine Chats und Dateien bleiben auf deinem PC.

- **Läuft auf einem Gaming-PC.** 32 GB Arbeitsspeicher (RAM) und Grafikspeicher zusammen reichen für den Anfang.
- **Passt zu deinen Werkzeugen.** Claude Code, Codex CLI, OpenCode und andere Agents verbinden sich über die OpenAI- und die Anthropic-API.
- **Liest Bilder.** Das ist standardmäßig eingeschaltet.
- **Kostenlos und privat.** MIT-Lizenz, kein Konto, keine Cloud, keine Telemetrie ([was es herunterlädt und wann](docs/SAFETY.md)).

## Was du brauchst

**Du brauchst mindestens 32 GB Speicher: Arbeitsspeicher (RAM) und Grafikspeicher (VRAM) zusammen.** Bei einem Mac oder einem
Ryzen AI Max ist das der gemeinsame Speicher. Ein PC mit 32 GB und einer Karte mit 12-16 GB führt die kleineren Größen aus,
mit 64 GB laufen alle Größen.

| Läuft auf | Anleitung |
| --- | --- |
| **NVIDIA** GeForce RTX 20, 30, 40 und 50 Serie, 12 GB oder mehr | [Installation](#installation) |
| **AMD** Radeon RX 6800 und neuer, Radeon AI PRO R9700 | [AMD-Karten](docs/AMD_HIP.md) |
| **Intel** Arc | [Intel Arc](docs/INTEL_ARC.md) |
| **Apple-Silicon-Mac**, M1 oder neuer, macOS 15 oder neuer | [Mac](docs/MAC.md) |
| **AMD Ryzen AI Max (Strix Halo)** und andere PCs mit gemeinsamem Speicher | [Strix Halo](docs/STRIX_HALO.md) |
| Ältere Karten und Prozessoren | [Ältere GPUs](docs/OLDER_GPUS.md) |

Außerdem brauchst du etwa 80 GB freien Speicherplatz (eine SSD), Windows 10 / 11, Linux oder macOS und einen aktuellen
Grafiktreiber. Zwei oder drei Karten desselben Herstellers können sich ein Modell teilen ([Multi-GPU](docs/MULTI_GPU.md)).

## Installation

### 1. Die Strata-App (am einfachsten)

Lade die Datei für deinen Computer herunter, öffne sie und drück auf Installieren beim empfohlenen Modell.

**Windows 10 / 11**

[![Windows, NVIDIA, Treiber 580 oder neuer](https://img.shields.io/badge/Windows-NVIDIA%20%C2%B7%20driver%20580%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-cuda13.exe)
[![Windows, NVIDIA, älterer Treiber](https://img.shields.io/badge/Windows-NVIDIA%20%C2%B7%20older%20driver-4C7A00?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-cuda12.exe)
[![Windows, AMD RX 9000 oder R9700](https://img.shields.io/badge/Windows-AMD%20RX%209000%20%C2%B7%20R9700-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-hip.exe)
[![Windows, AMD RX 7000 oder Ryzen AI Max](https://img.shields.io/badge/Windows-AMD%20RX%207000%20%C2%B7%20Ryzen%20AI%20Max-B3131B?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-hip-rdna3.exe)

**Linux**

[![Linux, NVIDIA, Treiber 580 oder neuer](https://img.shields.io/badge/Linux-NVIDIA%20%C2%B7%20driver%20580%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-nvidia-cuda13.AppImage)
[![Linux, NVIDIA, älterer Treiber](https://img.shields.io/badge/Linux-NVIDIA%20%C2%B7%20older%20driver-4C7A00?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-nvidia-cuda12.AppImage)
[![Linux, AMD RX 9000 oder R9700](https://img.shields.io/badge/Linux-AMD%20RX%209000%20%C2%B7%20R9700-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-rx9070.AppImage)
[![Linux, AMD RX 7000](https://img.shields.io/badge/Linux-AMD%20RX%207000-B3131B?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-rx7900.AppImage)
[![Linux, AMD Ryzen AI Max](https://img.shields.io/badge/Linux-Ryzen%20AI%20Max-8A0E15?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-strixhalo.AppImage)

**Mac** (Apple Silicon, macOS 15 oder neuer)

[![Mac, Apple Silicon](https://img.shields.io/badge/Mac-Apple%20Silicon-333333?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-mac.dmg)

Dein NVIDIA-Treiber ist älter als 580 (`nvidia-smi` zeigt ihn an)? Nimm die Datei für ältere Treiber. Eine AMD-RX-6000-Karte
nimmt die RX-7000-Datei, sie holt die richtige Engine von selbst. Alle Dateien stehen auf der
[Releases-Seite](https://github.com/Niko1221/Strata/releases/latest).

| Erster Start | Was du tust |
| --- | --- |
| **Windows** | Starte die `.exe`. Du brauchst keine Administratorrechte. Wenn Windows "Der Computer wurde durch Windows geschützt" meldet, wähle **Weitere Informationen** und dann **Trotzdem ausführen**. |
| **Linux** | Mach die Datei ausführbar (`chmod +x`) und starte sie. Sie braucht FUSE 2: `sudo apt install libfuse2t64` (Ubuntu 24.04 und neuer) oder `libfuse2` (22.04). |
| **Mac** | Öffne die `.dmg` und zieh Strata in den Ordner "Programme". |

Das Modell ist ein großer Download (40-70 GB). Wenn er abbricht, geht er dort weiter, wo er aufgehört hat.

### 2. Das Terminal (eine Zeile)

Der Befehl `strata`, für Skripte, Server und KI-Agents. Damit läuft auch DeepSeek.

Linux und Mac:

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

Windows (PowerShell):

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

Dann startest du es. Es stellt dir eine Frage nach der anderen (Modell, Kontext, Bilder, wer sich verbinden darf), lädt,
was es braucht, und beginnt zu laufen:

```sh
strata launch
```

Du brauchst keine Administratorrechte. Die Installation prüft die SHA-256-Prüfsumme von allem, was sie herunterlädt. Öffne
unter Windows nach der Installation ein neues Terminalfenster. Alle Bildschirme und Flags: [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md).

### 3. Deine KI richtet es ein

Füge das in Claude Code, Cursor, Codex, GitHub Copilot oder einen anderen Coding-Assistenten ein:

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

Der Assistent prüft deine Grafikkarte, deinen Arbeitsspeicher und den Speicherplatz, wählt das passende Modell, installiert
und startet es und sagt dir, wie du deine Apps verbindest. KI-Werkzeuge können Strata auch über den
[MCP-Server](docs/MCP_SERVER.md) installieren, starten und stoppen.

Du willst es lieber selbst bauen? Siehe [Aus dem Quellcode bauen](#aus-dem-quellcode-bauen).

## Modelle

Die App und `strata launch` empfehlen eine Größe passend zu deinem Speicher. Kleinere Größen sind schneller, größere
Größen sind etwas klüger.

### Wenig Speicher? Probier Gyro

[Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) von agentionai ist die kleinste Möglichkeit, Qwen3.8-Flash-Next
auszuführen. Es läuft auf NVIDIA, AMD, Intel Arc und Apple Silicon.

| Größe | Benötigter Speicher (RAM + VRAM) | Download | Geeignet für |
| --- | ---: | ---: | --- |
| **Gyro-S** | 27,6 GB | 58,5 GB | PCs, auf denen nichts anderes läuft. 9 Tokens/s auf einem Mac mini mit 32 GB. |
| **Gyro-M** | 35 GB | 92,0 GB | Ungefähr die Qualität von IQ3_XXS bei 20 % weniger GPU-Speicher. |

Starte es mit `strata launch --model gyro-s`. Details: [docs/GYRO.md](docs/GYRO.md).

### Die Größen von Qwen3.8-Flash-Next

| Dein Speicher (RAM + VRAM) | Nimm | Warum |
| --- | --- | --- |
| **32 GB** | **Gyro-S**, oder den **[Coder](docs/MODELS.md#coder)** für Code | Der Coder behält die Experten, die Code braucht |
| **40 GB** | **[Swift 1.5](docs/MODELS.md#swift-15)** IQ2_XS (39,2 GB) oder **Gyro-M** | Swift denkt kürzer nach, bevor es antwortet |
| **48 GB** | **IQ3_XXS** (47,0 GB) | Langsamer, besser |
| **64 GB** | **IQ3_S** (54,8 GB) | Erreicht in den veröffentlichten Tests das volle Modell |
| **96 GB oder mehr** | **UD-IQ4_XS** bis **Q8_0** von Unsloth | Am nächsten am vollen Modell |

Auch mit weniger RAM, als eine Zeile nennt, kann es funktionieren, nur langsamer: Strata wechselt von selbst in seinen
[Modus für wenig RAM](docs/MODELS.md#a-big-graphics-card-and-little-ram). Außerdem gibt es:
[NVFP4](docs/NVFP4.md) für NVIDIA-Karten, [Qwen3.6-35B-A3B](docs/QWEN36.md) (kleiner, mehrere Anfragen gleichzeitig) und
[OrcaRouters Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs). Alle Größen und was wo läuft:
[docs/MODELS.md](docs/MODELS.md).

### DeepSeek V4.1-Flash (Terminal, ab 0.2.4)

Ein deutlich größeres Modell, das auf einer Grafikkarte mit 12 GB und 46 bis 64 GB RAM läuft. Strata hält die meistgenutzten
Teile im Speicher und liest den Rest von deiner SSD. [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
hat die Sets gebaut. Es läuft auf NVIDIA-RTX-Karten der Serien 30 bis 50 und auf der Radeon AI PRO R9700. Wähle es in
`strata launch`. Die Desktop-App bietet es noch nicht an.

| Set | Download | Qualität | Schreibt Code mit |
| --- | ---: | --- | --- |
| **IQ1_M** (1,75 Bit) | 239 GB | Das kleinste, schwächer bei einfacher Prosa | auf einem PC noch nicht gemessen |
| **IQ2_XXS** (2.00 Bit) | 257 GB | Gute Balance | 10 Tokens/s (RTX 3060), 5 (RTX 5070), 48 (4x R9700) |
| **IQ2_XS** (2.25 Bit) | 274 GB | Am nächsten am Original | 15 Tokens/s (eine R9700), 80 (4x R9700) |
| **IQ4_NL** (Engram-Tabelle) | in jedem Download oben enthalten | Die Nachschlagetabelle, die alle drei Sets teilen | |

Starte es mit `strata launch --model ist-das-2.00bpw` (oder `ist-das-1.75bpw`, `ist-das-2.25bpw`). Qualitätswerte,
Geschwindigkeiten je PC und die manuelle Installation: [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md).

## Wie schnell ist es?

Gemessen mit einer Grafikkarte mit 12 GB (RTX 5070, 64 GB RAM). Ergebnisse anderer Karten:
[Tempo jeder Größe](docs/MODELS.md#how-fast-is-each-size) und [Ergebnisse der Community](docs/COMMUNITY_BENCHMARKS.md).

| Modell | Antworten erscheinen mit | Liest deinen Prompt mit |
| --- | --- | --- |
| **Q2_0** | `████████████` 94 Tokens/s | `████████████` 2.650 Tokens/s |
| **IQ2_XS** | `██████████░░` 79 Tokens/s | `█████████░░░` 2.090 Tokens/s |
| **IQ3_XXS** | `████████░░░░` 62 Tokens/s | `████████░░░░` 1.750 Tokens/s |
| **IQ3_S** | `███████░░░░░` 53 Tokens/s | `███████░░░░░` 1.620 Tokens/s |
| **Coder** | `███████░░░░░` 55 Tokens/s | `██████████░░` 2.180 Tokens/s |

Auf einer AMD-Karte mit 16 GB (RX 9070 XT, 47 GB RAM):

| Modell | Antworten erscheinen mit | Liest deinen Prompt mit |
| --- | --- | --- |
| **Q2_0** | `████████░░░░` 60 Tokens/s | `█████░░░░░░░` 1.160 Tokens/s |
| **IQ2_XS** | `███████░░░░░` 52 Tokens/s | `█████░░░░░░░` 1.110 Tokens/s |
| **Coder** | `██████░░░░░░` 44 Tokens/s | `██████░░░░░░` 1.420 Tokens/s |

Ein Token ist etwa ¾ eines Wortes, 60 Tokens pro Sekunde sind also schneller, als du lesen kannst. "Liest deinen Prompt"
beschreibt, wie schnell Strata ein langes Dokument, Code oder einen Chatverlauf aufnimmt.

## Benutzung

### In der App

Ein Chat, ein **Monitor** für das Modell sowie für GPU, CPU und RAM, eine Seite **Modelle** und die **Einstellungen**. Der
Chat kann in einem Ordner auf deinem PC arbeiten. Im Modus **Manuell** fragt er nach, bevor er eine Datei ändert oder einen
Befehl ausführt. Er kann das Web durchsuchen (DuckDuckGo oder Bing, ohne Konto). Die Suchbegriffe sind das Einzige, was
deinen PC verlässt.

<p align="center"><img src="docs/media/app-chat.png" width="900" alt="Die Strata-App: ein Chat, in dem der Agent einen kleinen Voxel-Pagodengarten gebaut hat, die Seite ist rechts in der Vorschau geöffnet"><br>
<sub>Der Agent hat einen Voxel-Pagodengarten als eine HTML-Datei geschrieben, und die Seite öffnet sich in der Vorschau neben dem Chat</sub></p>

### Im Terminal

<p align="center"><img src="docs/media/terminal-monitor.png" width="900" alt="Die Übersicht des Strata-Terminal-Monitors, während das Modell eine Antwort schreibt"><br>
<sub><code>strata monitor</code>, während das Modell antwortet: Geschwindigkeit, GPU, Speicher und die letzten Anfragen</sub></p>

```sh
strata launch                              # guided: model, context, pictures, access, then serve
strata launch --model q2_0 --yes --no-tui  # no questions: serve and print the address
strata launch --status                     # what is being served now
strata launch --stop                       # stop it
strata monitor                             # watch speed, GPU, memory and requests
strata update                              # get the newest engine
strata doctor                              # check the installed files
```

Um Strata von deinem Handy oder einem anderen PC aus zu erreichen, nutze `strata launch --access lan`. Der Befehl erzeugt
einen API-Schlüssel für dich. `strata --help` listet alles auf, und [docs/TERMINAL.md](docs/TERMINAL.md) erklärt es.

### Mit KI-Agents und anderen Apps

Ein Agent installiert Strata mit dem Einzeiler oben und führt `strata launch --model NAME --yes --no-tui` aus. Die Schritte,
Flags und Exit-Codes stehen in [docs/AI_SETUP.md](docs/AI_SETUP.md). Die Regeln für das Repository stehen in [AGENTS.md](AGENTS.md).

| Deine App spricht | Basis-URL |
| --- | --- |
| OpenAI-API (die meisten Apps und Agents) | `http://127.0.0.1:8080/v1` (beliebiger API-Schlüssel, beliebiger Modellname) |
| Anthropic-API (Claude Code) | `http://127.0.0.1:8080`, ohne `/v1`. Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` |
| OpenAI Responses API (Codex CLI) | `http://127.0.0.1:8080/v1/responses` ([Einrichtung](docs/DETAILS.md#the-responses-api-and-codex-cli)) |

`strata connect` gibt die genauen Einstellungen für OpenCode, Codex CLI, Claude Code, oh-my-pi, pi, Hermes Agent und OpenClaw
aus, und `strata connect opencode --write` schreibt sie für dich (mit einer Sicherung).

### Gut zu wissen

- **Bilder** sind für jedes Modell eingeschaltet, das sie lesen kann. Hänge ein Bild im Chat an oder füge es ein, oder sende es aus deiner App. Details: [docs/IMAGES.md](docs/IMAGES.md).
- **Das Denkniveau** kann aus, niedrig, mittel oder hoch sein, im Chat-Menü oder unter "reasoning effort" in deiner App. Aus ist am schnellsten.
- **Die erste Nachricht** eines Chats wird komplett gelesen, etwa 1 Minute pro 30.000 Tokens. Folgenachrichten starten in Sekunden.
- **Eine Anfrage nach der anderen** ist die Standardeinstellung. Für mehrere nutze `--parallel N` ([BATCHING.md](docs/BATCHING.md)). Auf einer Karte mit 12 GB wird dadurch jede Antwort langsamer.

## Aus dem Quellcode bauen

<details>
<summary>Windows, Linux und Mac aus dem Repository</summary>

[Lade Strata herunter](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) und entpacke es, oder nutze `git clone`.
**Windows:** Doppelklick auf `START-HERE.bat`. **Linux:** Führe `./setup.sh` im Strata-Ordner aus.

**Mac (Apple Silicon, macOS 15 oder neuer):** Installiere einmal die Command Line Tools von Apple mit `xcode-select --install`
(Xcode selbst brauchst du nicht). Öffne das Terminal im Strata-Ordner und führe `./setup.sh` aus. Das Skript baut die
Mac-Engine auf deinem Mac (etwa 7 Minuten), stellt dieselben Fragen, lädt das Modell herunter und startet es. Beim nächsten
Mal startet `./setup.sh` sofort. Welche Größe zu deinem Mac passt: [docs/MAC.md](docs/MAC.md).

Der Installer erkennt deine Karte und richtet die passende Engine ein. Er fragt, welches Modell und welche Größe du willst,
wie viel Kontext (wie viel Text sich das Modell merkt) und ob es Bilder lesen soll. Drück Enter für die empfohlene Antwort.
Dann lädt er das Modell herunter (etwa 70 GB) und startet es. Wenn der Download abbricht, führe das Skript erneut aus, dann
geht es weiter. Dein Browser öffnet die Strata-Seite unter `http://127.0.0.1:8080`.

> Während das Modell startet, kann dein PC 1-3 Minuten langsam sein oder nicht reagieren (beim ersten Mal am längsten). Strata
> lädt 35-55 GB in deinen RAM und sperrt einen Teil davon für die Grafikkarte. Warte ab und schließe das Fenster nicht.

Starte beim nächsten Mal wieder `START-HERE.bat` (oder `./setup.sh`). Es startet sofort und lädt nichts doppelt herunter.
Schließ sein Fenster, um das Modell zu beenden. `UPDATE.bat` (`./update.sh`) aktualisiert Strata, ohne es zu starten.
Aktualisieren, Docker, mehrere Karten, wohin die Dateien kommen und alle Optionen: [docs/INSTALL.md](docs/INSTALL.md).

Um deinen PC zu messen, führe `BENCH.bat` (`./bench.sh`) aus. Das dauert 5-10 Minuten und schreibt einen Bericht, den du teilen
kannst ([so geht's](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)).

**Optional: mehr Tempo auf einer Karte, wenn das ganze Modell in den RAM passt.** Zwei Einstellungen zum Einschalten von
RubenTorney (#1904, #1905). Sie helfen, wenn du eine Grafikkarte nutzt und dein RAM alle Experten aufnimmt, zum Beispiel bei
Q2_0 oder IQ3_XXS auf einem PC mit 64 GB. Öffne `strata-<model>.json` im Strata-Datenordner ([wo](docs/INSTALL.md#where-things-are-stored)) und ergänze:

- `"--adapt-async", "1"` in der Liste `"args"`. Der Expertencache wird dann in einem Hilfsthread aktualisiert, statt die
  Antwort anzuhalten. Gemessen auf einer RTX 5070 (Windows, Q2_0): +4-5 % beim Decodieren, +7-10 % bei Code und langen
  Dokumenten. Die Antworten können sich leicht von der Standardeinstellung unterscheiden und bleiben von Lauf zu Lauf gleich.
- `"STRATA_PIN_BY_NEED": "1"` im Block `"env"` (Windows). Windows kann nur etwa die Hälfte des RAM für die Grafikkarte sperren,
  und diese Einstellung sperrt zuerst die Experten, die die Karte am dringendsten braucht. Gemessen auf der RTX 5070: +4,4 % beim Lesen des Prompts.

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

Starte Strata nach der Änderung neu. Zum Rückgängigmachen entfernst du die beiden Einträge.

</details>

## Wie funktioniert das?

Modelle wie dieses laufen normalerweise auf Servern mit Hunderten Gigabyte Grafikspeicher. Deine Grafikkarte hat 12-24 GB.
Strata bringt das Modell auf deinen PC, indem es die Arbeit auf den ganzen Rechner verteilt. Stell dir eine Bibliothek vor.

- Das Modell ist ein Team aus 24.576 kleinen Spezialisten ("Experten"), wie 24.576 Nachschlagewerke. Für jedes Wort braucht es nur 10 davon.
- **Deine Grafikkarte ist der Schreibtisch.** Er ist klein und schnell. Die etwa 4.000 am häufigsten genutzten Bücher liegen
  dort aufgeschlagen.
- **Dein RAM sind die Regale im Lesesaal.** Alle 24.576 Bücher stehen dort. Braucht ein Wort ein Buch, das nicht auf dem
  Schreibtisch liegt, holt dein Prozessor es sofort aus dem Regal, während die Karte weiterarbeitet.
- **Deine SSD ist das Archiv.** Sie enthält eine Nachschlagetabelle von 29 GB. Das Modell liest pro Wort nur wenige Zeilen
  daraus, deshalb muss sie nicht im RAM liegen.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="Die 24.576 Experten des Modells: die meistgenutzten auf der Grafikkarte, alle im RAM, eine Nachschlagetabelle auf der SSD"></p>

Ein kleiner Helfer rät außerdem die nächsten Wörter. Das große Modell prüft sie alle auf einmal und behält die richtigen. Du
bekommst dieselbe Antwort, 1,6-1,8-mal früher. Lange Texte werden in großen Stücken gelesen, bis zu 8.192 Tokens auf einmal,
mit über 1.000 Tokens pro Sekunde.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="Ein kleiner Helfer rät die nächsten Wörter, das große Modell prüft sie alle auf einmal und behält die richtigen"></p>

Die ausführliche Erklärung: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Jeder Teil mit seinen Zahlen:
[die Details](docs/DETAILS.md#how-it-works) und das [Paper](docs/paper/Strata-Paper.pdf).

## Etwas ist schiefgegangen?

- **Mein PC ist beim ersten Start von Strata eingefroren.** Strata lädt das Modell in den RAM. Warte ab und schließe das
  Fenster nicht. Nach 10 Minuten immer noch eingefroren? Starte den PC neu, schließe andere Programme und versuch es noch
  einmal, oder wähle eine kleinere Größe.
- **Es ist beim Herunterladen oder Installieren stehen geblieben.** Starte die App oder `strata launch` noch einmal. Es
  macht dort weiter, wo es aufgehört hat.
- **Es ist sehr langsam, oder es meldet "the engine stopped unexpectedly".** Dein PC hat nicht genug freien RAM. Schließe
  andere Programme (Browser brauchen viel) oder wähle eine kleinere Größe wie Q2_0 oder IQ2_XS.
- **Es meldet, dass es keine Engine für diesen PC gibt.** Öffne in der App **Einstellungen, Engine** und drücke **Engine
  aktualisieren**. Führe im Terminal `strata update` aus.
- **Es meldet, dass Port 8080 schon belegt ist.** Strata läuft bereits. Such das Fenster.

Weitere Lösungen: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Du kommst nicht weiter? Öffne ein
[Issue](https://github.com/Niko1221/Strata/issues) und hänge die Protokolle an: in der App unter **Einstellungen, Engine,
Diagnose, Protokolle speichern**, im Terminal die Ausgabe von `strata doctor`. Ein Sicherheitsproblem meldest du bitte
privat: [SECURITY.md](SECURITY.md).

## Danksagung und Lizenz

Das Modell ist [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) vom Qwen-Team. Komprimiert wurde es von
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5), agentionai (Gyro)
und Unsloth. ISTA-DASLab hat auch die Sets von DeepSeek V4.1-Flash gebaut. Strata nutzt Teile von
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Alle Danksagungen: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits).
Strata ist Open Source unter der [MIT-Lizenz](LICENSE). Einige Teile und alle Modelle haben eigene Lizenzen
([welche](docs/HOW_IT_WORKS.md#license)).

## Mitmachen

Du hast einen Fehler gefunden oder willst eine Änderung oder einen Benchmark einsenden? Lies zuerst [CONTRIBUTING.md](CONTRIBUTING.md).
Dort steht, was dazugehört, damit andere deine Arbeit nachprüfen können.

## Strata unterstützen

Strata ist kostenlos und steht unter der MIT-Lizenz. Ich baue es und teste es auf jeder Hardware, die ich bekommen kann. Wenn es auf deinem PC gut läuft, hilft mir ein Kaffee,
weiter neue Karten zu testen und das zu beheben, was ihr meldet. Es besteht kein Druck. Strata zu nutzen und zu melden, was dir auffällt, hilft genauso viel.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>

Du möchtest Strata sponsern? Schreib mir über [GitHub](https://github.com/Niko1221).
