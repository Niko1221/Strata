# Hakua / Strata

[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)

Lokale KI für Windows mit Chat, Agenten-Workflows, Recherche mit Quellenangaben und einer Auswahl kompatibler Modelle. Hakua ist der zapabob-Fork von Niko1221/Strata; er wird unabhängig gepflegt und ist keine offiziell empfohlene Distribution des Upstream-Projekts.

## Version und Downloads

<!-- release-061-stable -->
Aktuelles Release: [0.6.1](https://github.com/zapabob/Strata/releases/tag/v0.6.1) (App `0.6.1`, Native Q38 `0.6.1` SM120, Control `0.6.0`). `releaseQualified=false` gilt weiterhin; Volllängen-Inferenz und allgemeine Modellqualität sind nicht zertifiziert.

[0.6.1](https://github.com/zapabob/Strata/releases/tag/v0.6.1) · [Strata-0.6.1-windows-x64-setup.exe](https://github.com/zapabob/Strata/releases/download/v0.6.1/Strata-0.6.1-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.6.1/Strata-0.6.1-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.1/SHA256SUMS.txt)

![Hakua / Strata real-model Japanese chat on Qwen3.8-27B](docs/media/native-qwen38-japanese-chat.png)
*Hakua / Strata Desktop mit `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf` über `strata-q38.exe` (CUDA 13 / RTX 5060 Ti 16 GB).*

## Installation und Aktualisierung

Verwenden Sie den Windows-x64-NSIS-Installer dieses Forks. Er installiert pro Benutzer und enthält eigenes Node, Python und Control; System-Entwicklungstools sind nicht erforderlich. Die native CUDA-Engine ist eine separate Komponente. Das `tar.7z`-Paket enthält Installer, CUDA-Engine-Paket, Control-Paket, Anleitungen und Prüfsummen ohne Modellgewichte oder persönliche Daten.

Beenden Sie laufende Anfragen, entladen Sie Modelle, sichern Sie die App-Daten und schließen Sie die App einschließlich des Tray-Symbols. Führen Sie den Installer im bestehenden Installationsordner aus, um per Überschreib-Installation zu aktualisieren, während Modellordner, Chats, Speicher, Tools, API-Schlüssel und Freigabe-Einstellungen erhalten bleiben. Importieren Sie die komprimierte CUDA-Komponente über *Ich habe die Datei bereits*, wählen Sie ein kompatibles Modell oder eine lokale GGUF-Datei und laden Sie es. Weitere Details unter [Installation und Rollback](docs/HAKUA_INSTALL.md).

## Neuerungen in 0.6.1 und Fork-Funktionen

0.6.1 erweitert den [lokalen Qwen3.8-27B-Dense-GGUF-Runner (`strata-q38.exe`)](docs/QWEN38_DENSE_NATIVE.md) unter Windows CUDA 13 / Blackwell (SM120):
- **Tensor-spezifischer Quantisierungs-Polymorphismus (`StrataQuantTensor`)**: unterstützt 11 GGML/Strata-Typen (`F32`, `F16`, `BF16`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ3_S`, `IQ2_S`, `IQ4_XS` sowie PrismML `PQ2_0=142`) pro Tensor.
- **Tensor-Core-Batched-Prefill + `spin8_triality.cuh` Batch-KV (`KvCache::enqueue_batch`)**: ersetzt sequenzielles GEMV durch doppelt gepuffertes `cublasGemmEx` (`prefill_gemm=cublas_tensor_op`, optional `STRATA_FP8_PREFILL=1`). Im 140-Token-Prefill + 8-Token-Decode-Benchmark (`--kv spin8_k8v4 --ctx-size 4096`) auf einer RTX 5060 Ti SM120 sank die Prefill-Zeit von `36.760,7 ms` (`3,81 tok/s`, `--conservative-sync`) auf `11.387,4 ms` (`12,29 tok/s`, **3,23x schneller**).
- **Echte MTP-`K=3`-Draft-Verifikation (`blk.64.nextn.*`)**: unterstützt `--spec-draft-n-max 0..3` mit `KvCache::rewind` und GDN-Zustands-Rollback.
- **KV-Cache & Japanisch-Profil**: `F16`, `Spin8 K8V4` und `F16-K / Turbo2/3/4-V` (`auto`-Reihenfolge: `f16` → `f16_turbo4` → `spin8_k8v4`) sowie optionales Japanisch-Ausgabeprofil.

`/btw` startet einen werkzeugfreien Nebenfragen-Subagenten mit eigener Stopp-Steuerung. `/research` und `/deepresearch` sammeln Quellen (mit integriertem RustXNG Full; Anfragen gehen an aktivierte externe Anbieter). Workflow-Skill-Lernen und automatische Anwendung sind getrennt schaltbar und in neuen Profilen deaktiviert. Siehe [Skills](docs/WORKFLOW_SKILLS.md) und [Suche](docs/BROWSER_SEARCH.md).

## Upstream-Integration und Kompatibilität

Die fixierte Upstream-Basis ist [c775728069](https://github.com/Niko1221/Strata/commit/c77572806996349b49995e4f98d41b5294d36a21) (entspricht Upstream 0.2.15 / Engine 0.1.60). Fork-Versionen sind unabhängig. DeepSeek-Kompatibilität bleibt als Erweiterung erhalten; fehlt eine dedizierte Laufzeit, wird das Laden blockiert. Dieses Windows-Paket impliziert keine native Bildeingabe, keine verteilte native DeepSeek-Engine und keine zertifizierte Linux/Mac-Inferenz.

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
