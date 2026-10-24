<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="380">
  </picture>
</p>

<p align="center"><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <b>Français</b> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a></p>

<h3 align="center">Faites tourner un modèle de pointe sur votre propre PC de jeu</h3>

<p align="center">Compatible avec NVIDIA, AMD, Intel · Windows, Linux, Mac · gratuit et open source</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?include_prereleases&label=release" alt="Dernière version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="Licence MIT"></a>
  <a href="https://github.com/Niko1221/Strata/stargazers"><img src="https://img.shields.io/github/stars/Niko1221/Strata?style=flat" alt="Étoiles GitHub"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?logo=buymeacoffee&logoColor=black" alt="Offrez-moi un café"></a>
</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="Un jardin de pagode en voxels écrit par le modèle de Strata, qui tourne dans le navigateur"></a><br>
<sub>Un jardin de pagode en voxels, généré en une seule requête sur une RTX 5070 avec Strata (IQ3_S, contexte de 128K) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">vidéo complète (49 s)</a></sub></p>

<p align="center">
  <a href="#installation"><b>Installation</b></a> ·
  <a href="#modèles">Modèles</a> ·
  <a href="#quelle-est-sa-vitesse-">Vitesse</a> ·
  <a href="#utilisation">Utilisation</a> ·
  <a href="#comment-ça-marche-">Comment ça marche</a> ·
  <a href="#un-problème-">Aide</a> ·
  <a href="#soutenir-strata">Soutenir</a>
</p>

Strata fait tourner **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, un modèle de 125 milliards
de paramètres qui demande normalement un serveur, sur un PC ordinaire. Depuis la 0.2.4, il fait aussi tourner **DeepSeek
V4.1-Flash** dans le Terminal. Il discute, écrit du code, lit des images et fonctionne avec vos applications et vos agents
de code. Vos conversations et vos fichiers restent sur votre PC.

- **Tourne sur un PC de jeu.** 32 Go de RAM et de mémoire graphique réunis suffisent pour commencer.
- **Fonctionne avec vos outils.** Claude Code, Codex CLI, OpenCode et d'autres agents se connectent par les API d'OpenAI et d'Anthropic.
- **Lit les images**, activé par défaut.
- **Gratuit et privé.** Licence MIT, pas de compte, pas de cloud, aucune télémétrie ([ce qu'il télécharge et quand](docs/SAFETY.md)).

## Ce qu'il vous faut

**Il vous faut au moins 32 Go de mémoire : RAM et mémoire graphique (VRAM) réunis.** Sur un Mac ou un Ryzen AI Max, c'est
la mémoire unifiée. Un PC de 32 Go avec une carte de 12 à 16 Go fait tourner les petites tailles, et 64 Go font tourner
toutes les tailles.

| Fonctionne sur | Guide |
| --- | --- |
| **NVIDIA** GeForce RTX série 20, 30, 40 et 50, 12 Go ou plus | [Installation](#installation) |
| **AMD** Radeon RX 6800 et plus récent, Radeon AI PRO R9700 | [Cartes AMD](docs/AMD_HIP.md) |
| **Intel** Arc | [Intel Arc](docs/INTEL_ARC.md) |
| **Mac Apple Silicon**, M1 ou plus récent, macOS 15 ou plus récent | [Mac](docs/MAC.md) |
| **AMD Ryzen AI Max (Strix Halo)** et autres PC à mémoire unifiée | [Strix Halo](docs/STRIX_HALO.md) |
| Cartes et processeurs plus anciens | [GPU anciens](docs/OLDER_GPUS.md) |

Il vous faut aussi environ 80 Go d'espace disque libre (un SSD), Windows 10 / 11, Linux ou macOS, et un pilote graphique
récent. Deux ou trois cartes de la même marque peuvent se partager un modèle ([multi-GPU](docs/MULTI_GPU.md)).

## Installation

### 1. L'application Strata (le plus simple)

Téléchargez le fichier pour votre ordinateur, ouvrez-le et appuyez sur Installer pour le modèle qu'elle recommande.

**Windows 10 / 11**

[![Windows, NVIDIA, pilote 580 ou plus récent](https://img.shields.io/badge/Windows-NVIDIA%20%C2%B7%20driver%20580%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-cuda13.exe)
[![Windows, NVIDIA, ancien pilote](https://img.shields.io/badge/Windows-NVIDIA%20%C2%B7%20older%20driver-4C7A00?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-cuda12.exe)
[![Windows, AMD RX 9000 ou R9700](https://img.shields.io/badge/Windows-AMD%20RX%209000%20%C2%B7%20R9700-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-hip.exe)
[![Windows, AMD RX 7000 ou Ryzen AI Max](https://img.shields.io/badge/Windows-AMD%20RX%207000%20%C2%B7%20Ryzen%20AI%20Max-B3131B?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-hip-rdna3.exe)

**Linux**

[![Linux, NVIDIA, pilote 580 ou plus récent](https://img.shields.io/badge/Linux-NVIDIA%20%C2%B7%20driver%20580%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-nvidia-cuda13.AppImage)
[![Linux, NVIDIA, ancien pilote](https://img.shields.io/badge/Linux-NVIDIA%20%C2%B7%20older%20driver-4C7A00?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-nvidia-cuda12.AppImage)
[![Linux, AMD RX 9000 ou R9700](https://img.shields.io/badge/Linux-AMD%20RX%209000%20%C2%B7%20R9700-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-rx9070.AppImage)
[![Linux, AMD RX 7000](https://img.shields.io/badge/Linux-AMD%20RX%207000-B3131B?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-rx7900.AppImage)
[![Linux, AMD Ryzen AI Max](https://img.shields.io/badge/Linux-Ryzen%20AI%20Max-8A0E15?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-strixhalo.AppImage)

**Mac** (Apple Silicon, macOS 15 ou plus récent)

[![Mac, Apple Silicon](https://img.shields.io/badge/Mac-Apple%20Silicon-333333?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-mac.dmg)

Votre pilote NVIDIA est antérieur à 580 (`nvidia-smi` l'affiche) ? Prenez le fichier pour ancien pilote. Une carte AMD RX 6000
prend le fichier RX 7000, qui récupère tout seul le bon moteur. Tous les fichiers sont sur la
[page des versions](https://github.com/Niko1221/Strata/releases/latest).

| Premier démarrage | Ce qu'il faut faire |
| --- | --- |
| **Windows** | Lancez le `.exe`. Pas besoin de droits d'administrateur. Si Windows affiche « Windows a protégé votre ordinateur », choisissez **Informations complémentaires**, puis **Exécuter quand même**. |
| **Linux** | Rendez le fichier exécutable (`chmod +x`) et lancez-le. Il a besoin de FUSE 2 : `sudo apt install libfuse2t64` (Ubuntu 24.04 et plus récent) ou `libfuse2` (22.04). |
| **Mac** | Ouvrez le `.dmg` et faites glisser Strata dans Applications. |

Le téléchargement du modèle est gros (40 à 70 Go). S'il s'arrête, il reprend là où il s'était arrêté.

### 2. Le Terminal (une ligne)

La commande `strata`, pour les scripts, les serveurs et les agents d'IA. C'est aussi le moyen de faire tourner DeepSeek.

Linux et Mac :

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

Windows (PowerShell) :

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

Puis lancez-le. Il pose une question à la fois (modèle, contexte, images, qui peut se connecter), télécharge ce dont il a
besoin et commence à servir :

```sh
strata launch
```

Pas besoin de droits d'administrateur. L'installation vérifie l'empreinte SHA-256 de tout ce qu'elle télécharge. Sous
Windows, ouvrez une nouvelle fenêtre de terminal après l'installation. Tous les écrans et toutes les options : [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md).

### 3. Laissez votre IA l'installer

Collez ceci dans Claude Code, Cursor, Codex, GitHub Copilot ou un autre assistant de programmation :

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

L'assistant vérifie votre carte graphique, votre RAM et votre disque, choisit le modèle adapté, l'installe, le démarre et
vous explique comment connecter vos applications. Les outils d'IA peuvent aussi installer, démarrer et arrêter Strata via son
[serveur MCP](docs/MCP_SERVER.md).

Vous préférez le compiler vous-même ? Voir [Compiler depuis les sources](#compiler-depuis-les-sources).

## Modèles

L'application et `strata launch` recommandent une taille selon votre mémoire. Les petites tailles sont plus rapides. Les
grandes sont un peu plus intelligentes.

### Peu de mémoire ? Essayez Gyro

[Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) d'agentionai est la façon la plus légère de faire tourner
Qwen3.8-Flash-Next. Il fonctionne sur NVIDIA, AMD, Intel Arc et Apple Silicon.

| Taille | Mémoire nécessaire (RAM + VRAM) | Téléchargement | Pour quoi |
| --- | ---: | ---: | --- |
| **Gyro-S** | 27,6 Go | 58,5 Go | Les PC où rien d'autre ne tient. 9 tokens/s sur un Mac mini de 32 Go. |
| **Gyro-M** | 35 Go | 92,0 Go | À peu près la qualité d'IQ3_XXS avec 20 % de mémoire GPU en moins. |

Lancez-le avec `strata launch --model gyro-s`. Détails : [docs/GYRO.md](docs/GYRO.md).

### Les tailles de Qwen3.8-Flash-Next

| Votre mémoire (RAM + VRAM) | Prenez | Pourquoi |
| --- | --- | --- |
| **32 Go** | **Gyro-S**, ou le **[Coder](docs/MODELS.md#coder)** pour le code | Le Coder garde les experts dont le code a besoin |
| **40 Go** | **[Swift 1.5](docs/MODELS.md#swift-15)** IQ2_XS (39,2 Go) ou **Gyro-M** | Swift réfléchit moins longtemps avant de répondre |
| **48 Go** | **IQ3_XXS** (47,0 Go) | Plus lent, meilleur |
| **64 Go** | **IQ3_S** (54,8 Go) | Égale le modèle complet dans les tests publiés |
| **96 Go ou plus** | **UD-IQ4_XS** à **Q8_0** d'Unsloth | Au plus près du modèle complet |

Avoir moins de RAM que ce qu'indique une ligne peut encore fonctionner, en plus lent : Strata passe tout seul en
[mode peu de RAM](docs/MODELS.md#a-big-graphics-card-and-little-ram). Il existe aussi :
[NVFP4](docs/NVFP4.md) pour les cartes NVIDIA, [Qwen3.6-35B-A3B](docs/QWEN36.md) (plus petit, plusieurs requêtes à la fois) et
[l'Uncensored IQ3_XXS d'OrcaRouter](docs/MODELS.md#orcarouter-uncensored-iq3_xxs). Toutes les tailles et ce qui tient où :
[docs/MODELS.md](docs/MODELS.md).

### DeepSeek V4.1-Flash (Terminal, depuis 0.2.4)

Un modèle bien plus gros, qui tourne sur une seule carte graphique de 12 Go avec 46 à 64 Go de RAM. Strata garde en mémoire
les parties les plus utilisées et lit le reste sur votre SSD. [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
a produit les versions. Il fonctionne sur les cartes NVIDIA RTX des séries 30 à 50 et sur la Radeon AI PRO R9700. Choisissez-le
dans `strata launch`. L'application de bureau ne le propose pas encore.

| Version | Téléchargement | Qualité | Écrit du code à |
| --- | ---: | --- | --- |
| **IQ1_M** (1,75 bit) | 239 Go | La plus petite, plus faible sur la prose simple | pas encore mesuré sur un PC |
| **IQ2_XXS** (2.00 bits) | 257 Go | Bon équilibre | 10 tokens/s (RTX 3060), 5 (RTX 5070), 48 (4x R9700) |
| **IQ2_XS** (2.25 bits) | 274 Go | Au plus près de l'original | 15 tokens/s (une R9700), 80 (4x R9700) |
| **IQ4_NL** (table Engram) | compris dans chaque téléchargement ci-dessus | La table de consultation que les trois versions partagent | |

Lancez-le avec `strata launch --model ist-das-2.00bpw` (ou `ist-das-1.75bpw`, `ist-das-2.25bpw`). Chiffres de qualité,
vitesses par PC et installation manuelle : [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md).

## Quelle est sa vitesse ?

Mesuré avec une carte graphique de 12 Go (RTX 5070, 64 Go de RAM). Voir les résultats d'autres cartes :
[vitesse de chaque taille](docs/MODELS.md#how-fast-is-each-size) et [résultats de la communauté](docs/COMMUNITY_BENCHMARKS.md).

| Modèle | Les réponses s'affichent à | Lit votre prompt à |
| --- | --- | --- |
| **Q2_0** | `████████████` 94 tokens/s | `████████████` 2 650 tokens/s |
| **IQ2_XS** | `██████████░░` 79 tokens/s | `█████████░░░` 2 090 tokens/s |
| **IQ3_XXS** | `████████░░░░` 62 tokens/s | `████████░░░░` 1 750 tokens/s |
| **IQ3_S** | `███████░░░░░` 53 tokens/s | `███████░░░░░` 1 620 tokens/s |
| **Coder** | `███████░░░░░` 55 tokens/s | `██████████░░` 2 180 tokens/s |

Avec une carte AMD de 16 Go (RX 9070 XT, 47 Go de RAM) :

| Modèle | Les réponses s'affichent à | Lit votre prompt à |
| --- | --- | --- |
| **Q2_0** | `████████░░░░` 60 tokens/s | `█████░░░░░░░` 1 160 tokens/s |
| **IQ2_XS** | `███████░░░░░` 52 tokens/s | `█████░░░░░░░` 1 110 tokens/s |
| **Coder** | `██████░░░░░░` 44 tokens/s | `██████░░░░░░` 1 420 tokens/s |

Un token vaut environ ¾ de mot, donc 60 tokens par seconde, c'est plus rapide que votre lecture. « Lit votre prompt »
indique la vitesse à laquelle Strata absorbe un long document, du code ou un historique de discussion.

## Utilisation

### Dans l'application

Un chat, un **Monitor** du modèle et de votre GPU, CPU et RAM, une page **Models** et les **Settings**. Le chat peut
travailler dans un dossier de votre PC. En mode **Manual**, il demande avant de modifier un fichier ou d'exécuter une
commande. Il peut chercher sur le web (DuckDuckGo ou Bing, sans compte). Les mots recherchés sont la seule chose qui quitte
votre PC.

<p align="center"><img src="docs/media/app-chat.png" width="900" alt="L'application Strata : un chat où l'agent a construit un petit jardin de pagode en voxels, avec la page ouverte dans l'aperçu à droite"><br>
<sub>L'agent a écrit un jardin de pagode en voxels dans un seul fichier HTML, et la page s'ouvre dans l'aperçu à côté du chat</sub></p>

### Dans le Terminal

<p align="center"><img src="docs/media/terminal-monitor.png" width="900" alt="La vue d'ensemble du moniteur du Terminal Strata pendant que le modèle écrit une réponse"><br>
<sub><code>strata monitor</code> pendant que le modèle répond : vitesse, GPU, mémoire et les dernières requêtes</sub></p>

```sh
strata launch                              # guided: model, context, pictures, access, then serve
strata launch --model q2_0 --yes --no-tui  # no questions: serve and print the address
strata launch --status                     # what is being served now
strata launch --stop                       # stop it
strata monitor                             # watch speed, GPU, memory and requests
strata update                              # get the newest engine
strata doctor                              # check the installed files
```

Pour joindre Strata depuis votre téléphone ou un autre PC, utilisez `strata launch --access lan`. La commande crée une clé
d'API pour vous. `strata --help` liste tout, et [docs/TERMINAL.md](docs/TERMINAL.md) l'explique.

### Avec des agents d'IA et d'autres applications

Un agent installe Strata avec la commande en une ligne ci-dessus et lance `strata launch --model NAME --yes --no-tui`. Les
étapes, les options et les codes de sortie sont dans [docs/AI_SETUP.md](docs/AI_SETUP.md). Les règles du dépôt pour les agents : [AGENTS.md](AGENTS.md).

| Votre application parle | URL de base |
| --- | --- |
| API OpenAI (la plupart des applications et des agents) | `http://127.0.0.1:8080/v1` (n'importe quelle clé d'API, n'importe quel nom de modèle) |
| API Anthropic (Claude Code) | `http://127.0.0.1:8080`, sans `/v1`. Claude Code : `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` |
| API OpenAI Responses (Codex CLI) | `http://127.0.0.1:8080/v1/responses` ([configuration](docs/DETAILS.md#the-responses-api-and-codex-cli)) |

`strata connect` affiche les réglages exacts pour OpenCode, Codex CLI, Claude Code, oh-my-pi, pi, Hermes Agent et OpenClaw,
et `strata connect opencode --write` les écrit pour vous (avec une sauvegarde).

### Bon à savoir

- **Les images** sont activées pour tous les modèles qui savent les lire. Joignez ou collez une image dans le chat, ou envoyez-la depuis votre application. Détails : [docs/IMAGES.md](docs/IMAGES.md).
- **La réflexion** peut être désactivée, faible, moyenne ou élevée, dans le menu du chat ou dans le réglage « reasoning effort » de votre application. Désactivée est le plus rapide.
- **Le premier message** d'une discussion est lu en entier, environ 1 minute pour 30 000 tokens. Les messages suivants démarrent en quelques secondes.
- **Une requête à la fois** par défaut. Pour en traiter plusieurs, utilisez `--parallel N` ([BATCHING.md](docs/BATCHING.md)). Sur une carte de 12 Go, chaque réponse devient alors plus lente.

## Compiler depuis les sources

<details>
<summary>Windows, Linux et Mac depuis le dépôt</summary>

[Téléchargez Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) et décompressez-le, ou faites un `git clone`.
**Windows :** double-cliquez sur `START-HERE.bat`. **Linux :** lancez `./setup.sh` dans le dossier Strata.

**Mac (Apple Silicon, macOS 15 ou plus récent) :** installez une fois les Command Line Tools d'Apple avec `xcode-select --install`
(Xcode lui-même n'est pas nécessaire). Ouvrez le Terminal dans le dossier Strata et lancez `./setup.sh`. Le script compile
le moteur Mac sur votre Mac (environ 7 minutes), pose les mêmes questions, télécharge le modèle et le démarre. La fois
suivante, `./setup.sh` le démarre tout de suite. Quelle taille convient à votre Mac : [docs/MAC.md](docs/MAC.md).

L'installateur détecte votre carte et prépare le bon moteur. Il demande quel modèle et quelle taille, combien de contexte
(la quantité de texte que le modèle garde en tête) et s'il doit lire les images. Appuyez sur Entrée pour la réponse
recommandée. Puis il télécharge le modèle (environ 70 Go) et le démarre. Si le téléchargement s'arrête, relancez le script
et il reprend. Votre navigateur ouvre la page de Strata à l'adresse `http://127.0.0.1:8080`.

> Pendant le démarrage du modèle, votre PC peut être lent ou ne plus répondre pendant 1 à 3 minutes (le plus longtemps la
> première fois). Strata charge 35 à 55 Go dans votre RAM et en verrouille une partie pour la carte graphique. Patientez et
> ne fermez pas la fenêtre.

La fois suivante, relancez `START-HERE.bat` (ou `./setup.sh`). Il démarre tout de suite et ne télécharge rien deux fois.
Fermez sa fenêtre pour arrêter le modèle. `UPDATE.bat` (`./update.sh`) met Strata à jour sans le démarrer. Mise à jour,
Docker, plusieurs cartes, où vont les fichiers et toutes les options : [docs/INSTALL.md](docs/INSTALL.md).

Pour mesurer votre PC, lancez `BENCH.bat` (`./bench.sh`). Cela prend 5 à 10 minutes et produit un rapport que vous pouvez
partager ([comment](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)).

**Facultatif : plus de vitesse sur une carte quand tout le modèle tient en RAM.** Deux réglages à activer soi-même, par
RubenTorney (#1904, #1905). Ils aident quand vous utilisez une seule carte graphique et que votre RAM contient tous les
experts, par exemple Q2_0 ou IQ3_XXS sur un PC de 64 Go. Ouvrez `strata-<model>.json` dans le dossier de données de Strata ([où](docs/INSTALL.md#where-things-are-stored)) et ajoutez :

- `"--adapt-async", "1"` à la liste `"args"`. Le cache des experts se met alors à jour sur un thread auxiliaire au lieu de
  mettre la réponse en pause. Mesuré sur une RTX 5070 (Windows, Q2_0) : +4 à 5 % en décodage, +7 à 10 % sur le code et les
  longs documents. Les réponses peuvent différer légèrement de celles du réglage par défaut, et restent identiques d'une exécution à l'autre.
- `"STRATA_PIN_BY_NEED": "1"` dans le bloc `"env"` (Windows). Windows ne peut verrouiller qu'environ la moitié de la RAM
  pour la carte graphique, et ce réglage verrouille d'abord les experts dont la carte a le plus besoin. Mesuré sur la RTX 5070 : +4,4 % en lecture du prompt.

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

Redémarrez Strata après la modification. Pour annuler, supprimez les deux entrées.

</details>

## Comment ça marche ?

Les modèles de ce genre tournent d'habitude sur des serveurs qui disposent de centaines de gigaoctets de mémoire graphique.
Votre carte graphique en a 12 à 24. Strata fait tenir le modèle en répartissant le travail sur tout votre PC. Imaginez une
bibliothèque.

- Le modèle est une équipe de 24 576 petits spécialistes (les « experts »), comme 24 576 ouvrages de référence. Chaque mot n'en demande que 10.
- **Votre carte graphique est le bureau.** Il est petit et rapide. Les quelque 4 000 livres les plus utilisés y sont ouverts.
- **Votre RAM, ce sont les étagères de la salle de lecture.** Les 24 576 livres s'y trouvent. Quand un mot demande un livre
  qui n'est pas sur le bureau, votre processeur va le chercher aussitôt sur l'étagère, pendant que la carte continue de travailler.
- **Votre SSD, ce sont les archives.** Il contient une table de consultation de 29 Go. Le modèle n'en lit que quelques lignes par
  mot, elle n'a donc pas besoin de tenir en RAM.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="Les 24 576 experts du modèle : les plus sollicités sur la carte graphique, tous dans la RAM, une table de consultation sur le SSD"></p>

Un petit assistant devine aussi les mots suivants. Le grand modèle les vérifie tous d'un coup et garde les bons. Vous
obtenez la même réponse, 1,6 à 1,8 fois plus tôt. Les textes longs sont lus par gros morceaux, jusqu'à 8 192 tokens à la
fois, à plus de 1 000 tokens par seconde.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="Un petit assistant devine les mots suivants ; le grand modèle les vérifie tous d'un coup et garde les bons"></p>

L'explication plus longue : [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Chaque élément avec ses chiffres :
[les détails](docs/DETAILS.md#how-it-works) et l'[article](docs/paper/Strata-Paper.pdf).

## Un problème ?

- **Mon PC s'est figé au premier démarrage de Strata.** Il charge le modèle en RAM. Patientez et ne fermez pas la fenêtre.
  Toujours figé après 10 minutes ? Redémarrez le PC, fermez les autres programmes et réessayez, ou choisissez une taille plus petite.
- **Ça s'est arrêté pendant le téléchargement ou l'installation.** Relancez l'application ou `strata launch`. Cela reprend là
  où ça s'était arrêté.
- **C'est très lent, ou il affiche « the engine stopped unexpectedly ».** Votre PC n'a pas assez de RAM libre. Fermez
  d'autres programmes (les navigateurs en consomment beaucoup) ou choisissez une taille plus petite, comme Q2_0 ou IQ2_XS.
- **Il indique qu'il n'y a pas de moteur pour ce PC.** Dans l'application, ouvrez **Settings, Engine** et appuyez sur **Update engine**
  (mettre à jour le moteur). Dans le Terminal, lancez `strata update`.
- **Il indique que le port 8080 est déjà utilisé.** Strata tourne déjà. Cherchez sa fenêtre.

Autres solutions : [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Toujours bloqué ? Ouvrez une
[issue](https://github.com/Niko1221/Strata/issues) et joignez les journaux : dans l'application via **Settings, Engine,
Diagnostics, Save logs**, dans le Terminal la sortie de `strata doctor`. Pour un problème de sécurité, signalez-le en
privé : [SECURITY.md](SECURITY.md).

## Remerciements et licence

Le modèle est [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), de l'équipe Qwen. Il a été compressé par
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5), agentionai (Gyro)
et Unsloth. ISTA-DASLab a aussi produit les versions de DeepSeek V4.1-Flash. Strata utilise des parties de
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Tous les remerciements : [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits).
Strata est open source sous [licence MIT](LICENSE). Quelques parties et tous les modèles ont leurs propres licences
([lesquelles](docs/HOW_IT_WORKS.md#license)).

## Contribuer

Vous avez trouvé un bug, ou vous voulez proposer une modification ou un benchmark ? Lisez d'abord [CONTRIBUTING.md](CONTRIBUTING.md).
Il indique ce qu'il faut inclure pour que d'autres puissent vérifier votre travail.

## Soutenir Strata

Strata est gratuit et sous licence MIT. Je le construis et je le teste sur le matériel que je peux obtenir. S'il tourne bien sur votre PC, un café m'aide
à continuer de tester de nouvelles cartes et à corriger ce que vous signalez. Aucune obligation. L'utiliser et signaler ce que vous constatez aide tout autant.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>

Vous souhaitez sponsoriser Strata ? Contactez-moi via [GitHub](https://github.com/Niko1221).
