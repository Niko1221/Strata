# Hakua / Strata

[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)

IA locale pour Windows avec chat, travail d'agent, recherche sourcée et choix de modèles compatibles. Hakua est le fork par zapabob de Niko1221/Strata ; il est maintenu indépendamment et n'est pas une distribution approuvée par le projet amont.

## Version et téléchargements

<!-- release-061-stable -->
Version actuelle : [0.6.1](https://github.com/zapabob/Strata/releases/tag/v0.6.1) (App `0.6.1`, Native Q38 `0.6.1` SM120, Control `0.6.0`). `releaseQualified=false` reste explicite ; l'inférence pleine fenêtre et la qualité générale des modèles ne sont pas qualifiées.

[0.6.1](https://github.com/zapabob/Strata/releases/tag/v0.6.1) · [Strata-0.6.1-windows-x64-setup.exe](https://github.com/zapabob/Strata/releases/download/v0.6.1/Strata-0.6.1-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.6.1/Strata-0.6.1-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.1/SHA256SUMS.txt)

![Hakua / Strata real-model Japanese chat on Qwen3.8-27B](docs/media/native-qwen38-japanese-chat.png)
*Application Hakua / Strata exécutant `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf` via `strata-q38.exe` (CUDA 13 / RTX 5060 Ti 16 Go).*

## Installation et mise à jour

Utilisez l'installateur NSIS Windows x64 de ce fork. Il s'installe par utilisateur et embarque Node, Python et Control privés ; aucun outil de développement système n'est requis. Le moteur CUDA natif est un composant séparé. L'archive `tar.7z` contient l'installateur, le paquet moteur CUDA, le paquet Control, les instructions et les sommes de contrôle, sans poids de modèles ni données personnelles.

Terminez les requêtes en cours, déchargez les modèles, sauvegardez les données, puis quittez l'application y compris l'icône de la barre des tâches. Lancez l'installateur dans le dossier existant pour mettre à jour par écrasement tout en conservant modèles, discussions, mémoire, outils, clés et autorisations. Importez le composant CUDA compressé via *J'ai déjà le fichier*, choisissez un modèle compatible ou un GGUF local, puis chargez-le. Voir [installation et restauration](docs/HAKUA_INSTALL.md).

## Nouveautés 0.6.1 et fonctionnalités du fork

La version 0.6.1 améliore le [runner local GGUF dense Qwen3.8-27B (`strata-q38.exe`)](docs/QWEN38_DENSE_NATIVE.md) sous Windows CUDA 13 / Blackwell (SM120) :
- **Polymorphisme de quantification par tenseur (`StrataQuantTensor`)** : prend en charge 11 types GGML/Strata (`F32`, `F16`, `BF16`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ3_S`, `IQ2_S`, `IQ4_XS` et PrismML `PQ2_0=142`) par tenseur.
- **Prefill par lots Tensor Core + KV par lots `spin8_triality.cuh` (`KvCache::enqueue_batch`)** : remplace le GEMV séquentiel par `cublasGemmEx` à double tampon (`prefill_gemm=cublas_tensor_op`, plus `STRATA_FP8_PREFILL=1` en option). Lors d'un benchmark de 140 tokens de prefill + 8 tokens de décodage (`--kv spin8_k8v4 --ctx-size 4096`) sur RTX 5060 Ti SM120, le prefill est passé de `36 760,7 ms` (`3,81 tok/s`, `--conservative-sync`) à `11 387,4 ms` (`12,29 tok/s`, **accélération 3,23x**).
- **Vérification réelle MTP `K=3` (`blk.64.nextn.*`)** : prend en charge `--spec-draft-n-max 0..3` avec `KvCache::rewind` et restauration d'état GDN.
- **Cache KV et profil japonais** : modes `F16`, `Spin8 K8V4` et `F16-K / Turbo2/3/4-V` (ordre `auto` : `f16` → `f16_turbo4` → `spin8_k8v4`) ainsi qu'un profil de sortie japonais optionnel.

`/btw` lance un sous-agent sans outils pour les questions annexes avec son propre contrôle d'arrêt. `/research` et `/deepresearch` collectent des sources avec RustXNG Full intégré (les requêtes sont envoyées aux fournisseurs externes activés). L'apprentissage et l'application automatique de compétences de workflow sont des options distinctes, désactivées par défaut dans les nouveaux profils. Voir [compétences](docs/WORKFLOW_SKILLS.md) et [recherche](docs/BROWSER_SEARCH.md).

## Intégration officielle et compatibilité

La base officielle épinglée est [c775728069](https://github.com/Niko1221/Strata/commit/c77572806996349b49995e4f98d41b5294d36a21), associée à la version officielle 0.2.15 / moteur 0.1.60. Les versions du fork sont indépendantes. La compatibilité DeepSeek est conservée comme extension et bloque le chargement en l'absence d'un runtime dédié. Ce paquet Windows n'implique pas d'entrée d'image native, de moteur DeepSeek natif distribué ni d'inférence qualifiée sous Linux/Mac.

## Protections et limites mesurées

Conservez les planchers de 3 Gio de RAM disponible, 4 Gio de marge de commit, au moins 768 Mio de réserve GPU et les seuils disque. 32 Go de RAM et 16 Go de VRAM ne garantissent pas l'adéquation de tout modèle ou contexte. `releaseQualified=false` reste en vigueur et l'installateur n'a pas de signature Authenticode Windows. Les mesures historiques sont conservées dans [preuves de version](docs/HAKUA_RELEASE.md).

## Documentation et développement

[Installation](docs/HAKUA_INSTALL.md) · [Preuves de version](docs/HAKUA_RELEASE.md) · [API et matériel](docs/AI_REFERENCE.md) · [Terminal](docs/TERMINAL.md) · [Compilation](docs/SOURCE_INSTALL.md) · [Empaquetage](packaging/README.md) · [Contribuer](CONTRIBUTING.md) · [Sécurité](SECURITY.md) · [Licence](LICENSE)

```sh
python -m pytest tests/terminal
```

```sh
cd apps/host
pnpm test
pnpm run build
```

### Niko1221/Strata — référence Terminal amont

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

[Niko1221/Strata](https://github.com/Niko1221/Strata) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) · [Sources et mentions tierces](third_party/)
