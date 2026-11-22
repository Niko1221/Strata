## Was sich in 0.7.6 ändert

Hinweis: Die Werte DS4 2,84x / FLA 4,86x / Sakura 22,44 t/s stammen aus den Antigravity-Entwicklungsprotokollen und wurden in dieser Build-Sitzung nicht unabhängig reproduziert. Diese Sitzung maß 9,09 t/s Prefill / 11,78 t/s Decode mit dem kanonischen GGUF bei 32K.

Diese Version behält alle Fork-Eigenschaften und fügt die nativen Optimierungen vom 22./23.11. hinzu: Produkt **0.7.6**, Engine-Familie **0.6.7**, Upstream v0.2.19. RTX 5060 Ti (sm_120a)/CUDA 13.5: DS4-9B Prefill 2,84x, E2E +22,7 %, FLA GDN + Spin(8) Prefill 4,86x, Sakura Inkling-2 65K-98K (22,44 t/s Decode). Ryu's (kuraneko1, MIT) SM x VRAM-Profile in der offiziellen CUDA-Inkling-2 (PR #6571). Signaturprüfung entfernt, SHA-256 bleibt. `releaseQualified=false`.

## Version und Downloads

<!-- release-062-stable -->
Aktuelles App-Release: [0.7.6](https://github.com/zapabob/Strata/releases/tag/v0.7.6) (App `0.7.3`; Terminal, Control `0.6.3`; native Engine-Familie `0.6.5`) — dies ist GitHub Latest. Die Windows-App liefert die 0.6.5-Engine-Familie (neu gebautes Q38 27B+9B, offizielles Hy4-Flash 0.1.64) als getrennte optionale Komponenten. Die Signaturprüfung wurde entfernt (kein Schlüssel); SHA-256-Prüfsummen bleiben die Integritätsprüfung. `releaseQualified=false` bleibt; Vollfenster-Inferenz und Modellqualität sind nicht qualifiziert. Eine dritte optionale Engine, `strata-inkling2` (Inkling-2 Mini, GGUF-Arch `inkling2`), wurde aus kuraneko1/Strata (MIT) von kuraneko1 (_ryu15_) mit Quellenangabe portiert.

[0.7.6](https://github.com/zapabob/Strata/releases/tag/v0.7.6) · [Setup-EXE](https://github.com/zapabob/Strata/releases/download/v0.7.6/Strata-0.7.6-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.7.6/Strata-0.7.6-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.7.6/SHA256SUMS.txt) · [BUNDLE-SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.7.6/BUNDLE-SHA256SUMS.txt)

![Hakua / Strata real-model Japanese chat on Qwen3.8-27B](docs/media/native-qwen38-japanese-chat.png)
*Hakua / Strata Desktop mit `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf` über `strata-q38.exe` (CUDA 13 / RTX 5060 Ti 16 GB).*

## Empfohlenes GGUF für den Qwen3.8-Prototyp

Für lokalen Textchat auf einer RTX 5060 Ti mit 16 GB VRAM empfehlen wir das getestete [gemischt quantisierte GGUF mit 3,69 bpw von soyaakinohara](https://huggingface.co/soyaakinohara/qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf). Danke an **soyaakinohara** für die Quantisierung und Veröffentlichung und an **AEON-7** für die Ausgangsgewichte. Das Modell wird separat heruntergeladen. Beginnen Sie mit 4.096 Kontext-Tokens; der Dateiname garantiert keinen Betrieb mit 12 GB VRAM.

Die [signierte native CUDA-13.5-Vorschau](https://github.com/zapabob/Strata/releases/tag/v0.7.2) bestand in App 0.6.3 zwei japanische Chat-Runden, das Stoppen einer Antwort und das Entladen. In zwei Tests mit der ersten Anfrage sank die F16-Prefill-Zeit für kurze Eingaben gegenüber dem bisherigen nativen Pfad um 35–43 %. Prism (llama.cpp) dekodierte unter gleichen Bedingungen schneller. [Einrichtung, öffentlicher Schlüssel und Grenzen](docs/Q38_NATIVE_PREFILL_PREVIEW.md).

## Installation und Aktualisierung

Verwenden Sie den Windows-x64-NSIS-Installer dieses Forks. Er installiert pro Benutzer und enthält eigenes Node, Python und Control; System-Entwicklungstools sind nicht erforderlich. Die native CUDA-Engine ist eine separate Komponente. Das `tar.7z`-Paket enthält Installer, CUDA-Engine-Paket, Control-Paket, Anleitungen und Prüfsummen ohne Modellgewichte oder persönliche Daten.

Beenden Sie laufende Anfragen, entladen Sie Modelle, sichern Sie die App-Daten und schließen Sie die App einschließlich des Tray-Symbols. Führen Sie den Installer im bestehenden Installationsordner aus, um per Überschreib-Installation zu aktualisieren, während Modellordner, Chats, Speicher, Tools, API-Schlüssel und Freigabe-Einstellungen erhalten bleiben. Importieren Sie die komprimierte CUDA-Komponente über *Ich habe die Datei bereits*, wählen Sie ein kompatibles Modell oder eine lokale GGUF-Datei und laden Sie es. Weitere Details unter [Installation und Rollback](docs/HAKUA_INSTALL.md).

## Neuerungen in 0.6.2 und Fork-Funktionen

0.6.2 erweitert den [lokalen Qwen3.8-27B-Dense-GGUF-Runner (`strata-q38.exe`)](docs/QWEN38_DENSE_NATIVE.md) unter Windows CUDA 13 / Blackwell (SM120):
- **Tensor-spezifischer Quantisierungs-Polymorphismus (`StrataQuantTensor`)**: unterstützt 11 GGML/Strata-Typen (`F32`, `F16`, `BF16`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ3_S`, `IQ2_S`, `IQ4_XS` sowie PrismML `PQ2_0=142`) pro Tensor.
- **Tensor-Core-Batched-Prefill + Multi-Token-GDN + Fused Spin8 Attention + CUDA Graph Decode (CUDA 13.5 SM120)**:
  - 140-Token-Prefill (`--kv spin8_k8v4 --ctx-size 4096`): von `36.760,7 ms` (`3,81 tok/s`, `--conservative-sync`) auf **`1.558,3 ms` (`89,8 tok/s` warm, 23,6x schneller)**.
  - Stationäres Decode (`GEN 32`, 31 Schritte, gepacktes GEMV + 8-Layer-Fence CUDA Graph): von `206,4 ms/Schritt` auf **`85,13 ms/Schritt` (`11,75 Schritte/s`, 2,42x schneller)**.
  - **Langkontext-Messung (`--kv spin8_k8v4`, RTX 5060 Ti 16 GB)**: **65K (`65.536`)** bestanden (1.024-Token-Prefill `10.813,5–11.220,7 ms`, `91,3–94,7 tok/s`; Decode `670,4 ms` / 7 Schritte); **96K (`98.304`)** bestanden (`required_with_reserve=15,09 GB`, 140-Token-Prefill `1.968,7 ms`); **131K (`131.072`)** durch die 768-MiB-GPU-Reserve geschützt abgelehnt (`required=15,97 GB > available=15,91 GB`).
- **Echte MTP-`K=3`-Draft-Verifikation (`blk.64.nextn.*`)**: unterstützt `--spec-draft-n-max 0..3` mit `KvCache::rewind` und GDN-Zustands-Rollback.
- **KV-Cache & Japanisch-Profil**: `F16`, `Spin8 K8V4` und `F16-K / Turbo2/3/4-V` (`auto`-Reihenfolge: `f16` → `f16_turbo4` → `spin8_k8v4`) sowie optionales Japanisch-Ausgabeprofil.

`/btw` startet einen werkzeugfreien Nebenfragen-Subagenten mit eigener Stopp-Steuerung. `/research` und `/deepresearch` sammeln Quellen (mit integriertem RustXNG Full; Anfragen gehen an aktivierte externe Anbieter). Workflow-Skill-Lernen und automatische Anwendung sind getrennt schaltbar und in neuen Profilen deaktiviert. Siehe [Skills](docs/WORKFLOW_SKILLS.md) und [Suche](docs/BROWSER_SEARCH.md).

## Upstream-Integration und Kompatibilität

Die fixierte Upstream-Basis ist [6e38f588](https://github.com/Niko1221/Strata/commit/6e38f588a646c0ee6948a74afbe670eb75896106) (entspricht Upstream 0.2.15 / Engine 0.1.60). Fork-Versionen sind unabhängig. DeepSeek-Kompatibilität bleibt als Erweiterung erhalten; fehlt eine dedizierte Laufzeit, wird das Laden blockiert. Dieses Windows-Paket impliziert keine native Bildeingabe, keine verteilte native DeepSeek-Engine und keine zertifizierte Linux/Mac-Inferenz.

## Schutzgrenzen und gemessene Limits

Halten Sie die Untergrenzen ein: 3 GiB freier RAM, 4 GiB Commit-Reserve, mindestens 768 MiB GPU-Reserve und Festplatten-Untergrenzen. 32 GB RAM und 16 GB VRAM garantieren nicht, dass jedes Modell oder jeder Kontext passt. `releaseQualified=false` gilt weiterhin; der Installer hat keine Windows-Authenticode-Signatur. Historische Messwerte bleiben in [Release-Nachweise](docs/HAKUA_RELEASE.md) erhalten.

## Dokumentation und Entwicklung

[Installation](docs/HAKUA_INSTALL.md) · [Release-Nachweise](docs/HAKUA_RELEASE.md) · [API & Hardware](docs/AI_REFERENCE.md) · [Terminal](docs/TERMINAL.md) · [Quellcode-Build](docs/SOURCE_INSTALL.md) · [Paketierung](packaging/README.md) · [Mitwirken](CONTRIBUTING.md) · [Sicherheit](SECURITY.md) · [Lizenz](LICENSE)

```sh
python -m pytest tests/terminal
```

```sh
cd apps/host
pnpm test
pnpm run build
```

### Niko1221/Strata — Upstream-Terminal-Referenz

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

[Niko1221/Strata](https://github.com/Niko1221/Strata) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) · [Hinweise zu Drittanbietern](third_party/)

## Änderungen in 0.6.3

- Das native Q38-Prefill nutzt cuBLAS-Tensor-Core-GEMM mit tokenweise skalierten FP16-Aktivierungen und einem 20 MiB L2-persistenten Gewichts-Tile. Decode verwendet formatabhängiges ausgerichtetes Q8_1-dp4a-GEMV und einen LayerFence(8)-CUDA-Graph.
- Die Zulassung langer Kontexte wählt die Prefill-Chunk-Größe anhand einer gemessenen VRAM-Leiter. Die GPU-Reserve von 768 MiB sowie die RAM-Untergrenze von 3 GiB und die Commit-Untergrenze von 4 GiB bleiben unverändert.
- Numerische Fixtures normalisieren synthetische gepackte Gewichte auf realistische Amplituden, ohne die strengen `_exact_q8`-Toleranzen zu lockern. Nicht ausgerichtetes GEMV greift auf einen Byte-Lade-Scalarkernel zurück.
- Verifikation: 17/17 native Tests bestanden; compute-sanitizer meldet bei memcheck/synccheck/racecheck 0 Fehler und 0 Hazards. Allgemeine Modellqualität, Inference über das gesamte Fenster und GPU-Parallelbetrieb bleiben nicht qualifiziert (`releaseQualified=false`).
- Forschungsgrundlage: NVIDIA CUDA 13.5 Release Notes (Korrektur für cuBLASLt-Algorithmus 66 bei Tensor Memory) und der Qwen3.8-Next-Architekturbericht (3:1-GDN/Global-Attention-Hybrid, QSA mit r=4-Kompression und K=2048-Token-Budget, gated residual, host-vorgeladene n-Gramm-Einbettung).
