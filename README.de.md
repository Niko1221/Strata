# Hakua / Strata

[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)

Lokale KI unter Windows für Chats, Agentenaufgaben und Recherche mit Quellen. Hakua ist der unabhängig gepflegte zapabob-Fork von Niko1221/Strata und keine vom Originalprojekt bestätigte Distribution.

## Version und Pakete

<!-- release-060-candidate -->
Dieser Quellstand ist der Integrationskandidat 0.6.0. App, Terminal, Control und nativer Quellcode tragen 0.6.0. Build, Installation und Veröffentlichung werden noch geprüft; eine Versionsnummer belegt keine verfügbare Datei. Die verifizierte frühere [Version 0.5.9](https://github.com/zapabob/Strata/releases/tag/v0.5.9) bleibt erhalten. Der Auslieferungsnachweis nennt den tatsächlichen Stand von 0.6.0.

[0.6.0](https://github.com/zapabob/Strata/releases/tag/v0.6.0) · [Strata-0.6.0-windows-x64-setup.exe](https://github.com/zapabob/Strata/releases/download/v0.6.0/Strata-0.6.0-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.6.0/Strata-0.6.0-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.0/SHA256SUMS.txt)

## Installation und Update

Verwenden Sie das Windows-x64-NSIS-Paket dieses Forks. Es installiert pro Benutzer und enthält eigenes Node, Python und Control; Entwicklungswerkzeuge sind nicht nötig. Die signierte native CUDA-Engine ist eine separate Komponente. Das tar.7z-Paket enthält denselben Installer, Komponenten, Anleitung und Prüfsummen, jedoch keine Modellgewichte oder persönlichen Daten. Zuerst 7z, dann tar entpacken und BUNDLE-SHA256SUMS.txt prüfen.

Aufträge abschließen, Modelle entladen, App-Daten sichern und Strata einschließlich Tray beenden. In den bisherigen Ordner installieren. Modelle, Chats, Speicher, Werkzeuge, Schlüssel und Freigaben erhalten. Die komprimierte Engine über die Option für eine vorhandene Datei importieren, ein kompatibles Modell oder lokales GGUF wählen und laden. Ein Modellname beweist keine Engine-Kompatibilität. [Installation und Rückkehr zur Vorversion](docs/HAKUA_INSTALL.md).

## Erhaltene Funktionen

`/btw` startet einen echten werkzeuglosen Nebenfragen-Agenten mit eigener Stoppfunktion. `/research` und `/deepresearch` sammeln Quellen. Die Websuche kann das mitgelieferte RustXNG Full verwenden. Suchanfragen gehen an aktivierte externe Anbieter; die Suche ist daher nicht vollständig offline. Quellenzahlen garantieren keine korrekte Antwort.

Das Lernen von Workflow-Skills und deren automatische Anwendung sind getrennte Optionen, bei neuen Profilen beide aus. Bestehende Einstellungen, private editierbare Skills, Revisionen und Rücknahme bleiben erhalten. Ebenso bleiben Dateioperationen, begrenzter Chat- und Speicherzugriff, kompatible PrismML-/TurboQuant-Pfade, begrenzter Expertenspeicher, Pack-Integrität und Diagnosefunktionen erhalten. [Skills](docs/WORKFLOW_SKILLS.md) · [Suche](docs/BROWSER_SEARCH.md).

## Originalprojekt und Kompatibilität

Grundlage ist der offizielle Commit [09b6038869](https://github.com/Niko1221/Strata/commit/09b603886925994d0d4461857b01321ef3aa431c) aus der Generation 0.2.14 / Engine 0.1.60: Netzwerk, Drafting-Auswahl, Prompt-Kernel, Speicherverwaltung und Plattformkorrekturen. Fork-Versionen sind unabhängig. Entfernte DeepSeek-Pfade bleiben als Erweiterung erhalten; ohne passende dedizierte Laufzeit muss das Laden scheitern. Dieses Windows-Paket belegt weder native Bildeingabe noch eine ausgelieferte native DeepSeek-Engine oder geprüfte Linux-/Mac-Inferenz.

## Schutz und Messgrenzen

Die Grenzen bleiben bei 3 GiB freiem RAM, 4 GiB Commit-Spielraum, mindestens 768 MiB GPU-Reserve und Mindestplatz auf den Laufwerken. Der Windows-RAM-Schutz überwacht auch Laden und Generierung. 32 GB RAM und 16 GB VRAM garantieren nicht jedes Modell oder Kontextfenster. Volle Kontextlänge, allgemeine Modellqualität und parallele GPU-Nutzung sind nicht qualifiziert: releaseQualified=false. Der Installer hat keine Windows-Authenticode-Signatur; Komponenten- und Feed-Signaturen sind davon getrennt. Frühere Messungen stehen mit ihren Originalversionen im [Auslieferungsnachweis](docs/HAKUA_RELEASE.md).

## Dokumentation und Entwicklung

[Install](docs/HAKUA_INSTALL.md) · [Release evidence](docs/HAKUA_RELEASE.md) · [API and hardware](docs/AI_REFERENCE.md) · [Terminal](docs/TERMINAL.md) · [Source build](docs/SOURCE_INSTALL.md) · [Packaging](packaging/README.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [License](LICENSE)

```sh
python -m pytest tests/terminal
```

```sh
cd apps/host
npm test
npm run build
```

### Niko1221/Strata — upstream Terminal reference

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

[Niko1221/Strata](https://github.com/Niko1221/Strata) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) · [Quellen und Hinweise Dritter](third_party/)
