# Hakua / Strata

[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)

IA locale pour Windows avec chat, travail d'agent, recherche sourcée et choix de modèles compatibles. Hakua est le fork par zapabob de Niko1221/Strata ; il est maintenu indépendamment et n'est pas une distribution approuvée par le projet amont.

## Nouveautes de 0.6.9

Cette version porte les travaux Q38 sur la vitesse, les formats et les vecteurs de contrôle vers la nouvelle disposition `terminal/` pour Strata officiel 0.2.16. Qwen3.8-27B utilise désormais une attention fusionnée GQA : un test A/B à chaud en ordre inversé a mesuré un **décodage 6,0 % plus rapide avec des tokens de sortie identiques** (référence 555,8 ms -> GQA 522,6 ms, médianes, invite de 10 tokens, ctx 512, KV F16). Le chemin 9B ajoute une attention FP16 groupée avec cache softmax partagé et la suppression des memset redondants/RoPE (5,4 % plus rapide, tokens bit à bit identiques). La prise en charge des formats FP4/FP8-NVFP4 et les vecteurs de contrôle sont inclus. Auparavant : les écritures KV F16 groupées ont réduit le coût KV du préfill de 27 % avec des logits octet pour octet identiques, le décodage est au niveau de Prism llama.cpp (tous deux DP4A), et le squeeze RAM 32 Go avec avertissement de limite de commit de Sync-Strata est inclus. Compilé et vérifié avec CUDA 13.5 / SM120. La génération sur fenêtre complète, la sémantique d'attention creuse QSA et la qualité générale du modèle restent non qualifiées (`releaseQualified=false`).

## Version et téléchargements

<!-- release-062-stable -->
Version actuelle de l'application : [0.6.9](https://github.com/zapabob/Strata/releases/tag/v0.6.9) (App `0.6.9` ; Terminal, Control et moteur natif `0.6.3`). Cette version Windows limitée à l'application n'est volontairement pas GitHub Latest ; Latest reste `0.6.3`. Le moteur GPU reste le composant signé séparément [`v0.6.4-preview.1`](https://github.com/zapabob/Strata/releases/tag/v0.6.4-preview.1). `releaseQualified=false` reste explicite ; l'inférence pleine fenêtre et la qualité générale des modèles ne sont pas qualifiées.

[0.6.9](https://github.com/zapabob/Strata/releases/tag/v0.6.9) · [EXE d'installation](https://github.com/zapabob/Strata/releases/download/v0.6.9/Strata-0.6.9-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.6.9/Strata-0.6.9-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.9/SHA256SUMS.txt) · [BUNDLE-SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.9/BUNDLE-SHA256SUMS.txt)

![Hakua / Strata real-model Japanese chat on Qwen3.8-27B](docs/media/native-qwen38-japanese-chat.png)
*Application Hakua / Strata exécutant `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf` via `strata-q38.exe` (CUDA 13 / RTX 5060 Ti 16 Go).*

## GGUF recommandé pour le prototype Qwen3.8

Pour essayer le chat textuel local sur une RTX 5060 Ti dotée de 16 Go de VRAM, nous recommandons le [GGUF à quantification mixte de 3,69 bpw de soyaakinohara](https://huggingface.co/soyaakinohara/qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf) utilisé dans nos essais. Merci à **soyaakinohara** pour la quantification et sa publication, et à **AEON-7** pour les poids d’origine. Téléchargez le modèle séparément et commencez avec un contexte de 4 096 tokens. Le nom du fichier ne garantit pas le fonctionnement avec 12 Go de VRAM.

L’[aperçu natif CUDA 13.5 signé](https://github.com/zapabob/Strata/releases/tag/v0.6.4-preview.1) a réussi deux tours en japonais, l’arrêt d’une réponse en cours et le déchargement dans l’application 0.6.3. Sur deux essais de première requête, le temps de préremplissage F16 court a baissé de 35 à 43 % par rapport au moteur natif sans l’option ; Prism (llama.cpp) décodait toujours plus vite. [Configuration, clé publique et limites](docs/Q38_NATIVE_PREFILL_PREVIEW.md).

## Installation et mise à jour

Utilisez l'installateur NSIS Windows x64 de ce fork. Il s'installe par utilisateur et embarque Node, Python et Control privés ; aucun outil de développement système n'est requis. Le moteur CUDA natif est un composant séparé. L'archive `tar.7z` contient l'installateur, le paquet moteur CUDA, le paquet Control, les instructions et les sommes de contrôle, sans poids de modèles ni données personnelles.

Terminez les requêtes en cours, déchargez les modèles, sauvegardez les données, puis quittez l'application y compris l'icône de la barre des tâches. Lancez l'installateur dans le dossier existant pour mettre à jour par écrasement tout en conservant modèles, discussions, mémoire, outils, clés et autorisations. Importez le composant CUDA compressé via *J'ai déjà le fichier*, choisissez un modèle compatible ou un GGUF local, puis chargez-le. Voir [installation et restauration](docs/HAKUA_INSTALL.md).

## Nouveautés 0.6.2 et fonctionnalités du fork

La version 0.6.2 améliore le [runner local GGUF dense Qwen3.8-27B (`strata-q38.exe`)](docs/QWEN38_DENSE_NATIVE.md) sous Windows CUDA 13 / Blackwell (SM120) :
- **Polymorphisme de quantification par tenseur (`StrataQuantTensor`)** : prend en charge 11 types GGML/Strata (`F32`, `F16`, `BF16`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ3_S`, `IQ2_S`, `IQ4_XS` et PrismML `PQ2_0=142`) par tenseur.
- **Prefill par lots Tensor Core + GDN multi-token + Attention Spin8 fusionnée + Décodage CUDA Graph (CUDA 13.5 SM120)** :
  - Prefill de 140 tokens (`--kv spin8_k8v4 --ctx-size 4096`) : passé de `36 760,7 ms` (`3,81 tok/s`, `--conservative-sync`) à **`1 558,3 ms` (`89,8 tok/s` à chaud, accélération 23,6x)**.
  - Décodage en régime permanent (`GEN 32`, 31 étapes, GEMV vectorisé + CUDA Graph avec barrières toutes les 8 couches) : passé de `206,4 ms/étape` à **`85,13 ms/étape` (`11,75 étapes/s`, accélération 2,42x)**.
  - **Mesures en contexte long (`--kv spin8_k8v4`, RTX 5060 Ti 16 Go)** : **65K (`65 536`)** validé (prefill 1 024 tokens `10 813,5–11 220,7 ms`, `91,3–94,7 tok/s` ; décodage `670,4 ms` / 7 étapes) ; **96K (`98 304`)** validé (`required_with_reserve=15,09 Go`, prefill 140 tokens `1 968,7 ms`) ; **131K (`131 072`)** bloqué par la réserve GPU de 768 Mio (`required=15,97 Go > available=15,91 Go`).
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

## Changements dans la version 0.6.3

- Le préremplissage natif Q38 utilise le GEMM Tensor Core de cuBLAS avec des activations FP16 mises à l'échelle par jeton et une tuile de poids persistante en L2 de 20 Mio. Le décodage emploie un GEMV dp4a Q8_1 aligné selon le format et un CUDA Graph LayerFence(8).
- L'admission en contexte long choisit la taille de lot de préremplissage à partir d'une échelle VRAM mesurée. La réserve GPU de 768 Mio, le plancher RAM de 3 Gio et le plancher de commit de 4 Gio restent inchangés.
- Les fixtures numériques normalisent les poids empaquetés synthétiques à une amplitude réaliste sans assouplir les tolérances strictes de `_exact_q8`. Le GEMV non aligné bascule vers un noyau scalaire à chargement par octet.
- Vérification : 17/17 tests natifs réussis ; compute-sanitizer rapporte 0 erreur et 0 danger pour memcheck/synccheck/racecheck. La qualité générale du modèle, l'inférence sur toute la fenêtre et la concurrence GPU restent non qualifiées (`releaseQualified=false`).
- Base de recherche : notes de version NVIDIA CUDA 13.5 (correction de l'algorithme 66 de cuBLASLt avec Tensor Memory) et rapport d'architecture Qwen3.8-Next (hybride GDN/attention globale 3:1, QSA avec compression r=4 et budget K=2048, résiduel à porte, plongement de n-grammes préchargé côté hôte).
