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
  <a href="https://github.com/Niko1221/Strata/releases"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?include_prereleases&label=release" alt="Latest release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/Niko1221/Strata/stargazers"><img src="https://img.shields.io/github/stars/Niko1221/Strata?style=flat" alt="GitHub stars"></a>
</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>Un jardin de pagode en voxels, généré en une seule requête sur une RTX 5070 avec Strata (IQ3_S, contexte de 128K) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">vidéo complète (49 s)</a></sub></p>

Strata fait tourner **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, un modèle de 125 milliards
de paramètres qui demande normalement un serveur, sur un PC ordinaire. Il discute, écrit du code, lit des images et
fonctionne avec vos applications et vos agents de code. Vos conversations et vos fichiers restent sur votre PC.

**Nouveau ici ?** Regardez [ce qu'il vous faut](#ce-quil-vous-faut), puis [installez l'application](#option-a--lapplication-strata).

- **Tourne sur un PC de jeu.** 32 Go de RAM et de mémoire graphique réunis suffisent pour commencer.
- **Fonctionne avec vos outils.** Il parle les API d'OpenAI et d'Anthropic, donc Claude Code, Codex CLI, OpenCode et d'autres agents peuvent s'y connecter.
- **Lit les images.** C'est activé par défaut.
- **Gratuit.** Licence MIT, pas de compte, pas de cloud.
- **Privé.** Aucune télémétrie. Ce que Strata télécharge, quand il utilise internet et comment il vérifie les téléchargements est expliqué dans [docs/SAFETY.md](docs/SAFETY.md).

Version 0.2.0-beta · moteur 0.1.45.3

## Ce qu'il vous faut

**Il vous faut au moins 32 Go de RAM et de mémoire graphique (VRAM) réunis.** Sur un Mac ou un Ryzen AI Max, c'est la
mémoire unifiée. Plus de mémoire permet un modèle plus gros ou plus rapide.

- Un PC de 32 Go avec une carte de 12 à 16 Go fait tourner le Coder.
- Avec 64 Go de RAM, toutes les tailles fonctionnent.

| Votre ordinateur | Détails |
| --- | --- |
| **NVIDIA** GeForce RTX série 20, 30, 40, 50 (12 Go ou plus) | Installez l'application ou le Terminal. |
| **AMD** Radeon RX 6800 et plus récent, Radeon AI PRO R9700 (12 Go ou plus) | [Cartes AMD](docs/AMD_HIP.md) |
| **Intel** Arc | [Intel Arc](docs/INTEL_ARC.md), [moteur Intel](docs/INTEL.md) |
| **Mac Apple Silicon** (M1 ou plus récent, macOS 15 ou plus récent) | [Mac](docs/MAC.md) |
| **AMD Ryzen AI Max (Strix Halo)** et autres systèmes à mémoire unifiée | [Strix Halo](docs/STRIX_HALO.md) |
| Cartes et processeurs plus anciens | [GPU anciens](docs/OLDER_GPUS.md), [CPU anciens](docs/INSTALL.md#older-cpus-experimental) |

Il faut aussi :

- environ 80 Go d'espace disque libre (un SSD accélère beaucoup le premier démarrage)
- Windows 10 / 11, Linux ou macOS
- un pilote graphique récent

Deux ou trois cartes de la même marque peuvent se partager un modèle ([multi-GPU](docs/MULTI_GPU.md)). Strata ne répartit
jamais un modèle entre des cartes NVIDIA et AMD. Liste complète : [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Installation

Choisissez une option. Elles lancent toutes le même moteur et utilisent les mêmes modèles.

### Option A : l'application Strata

Le plus simple. L'application contient le moteur : installez-la, choisissez un modèle, discutez. Téléchargez le fichier
correspondant à votre système et à votre carte graphique depuis la [page des versions](https://github.com/Niko1221/Strata/releases/latest).
Vous ne savez pas lequel prendre ? Lisez le paragraphe sous le tableau.

| Système | Votre carte graphique | Fichier à télécharger |
| --- | --- | --- |
| **Windows 10 / 11** | NVIDIA RTX, pilote 580 ou plus récent | [`Strata-0.2.0-beta-windows-cuda13.exe`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.0-beta-windows-cuda13.exe) |
| | NVIDIA, carte plus ancienne ou pilote antérieur à 580 | [`Strata-0.2.0-beta-windows-cuda12.exe`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.0-beta-windows-cuda12.exe) |
| | AMD Radeon série RX 9000 ou Radeon AI PRO R9700 | [`Strata-0.2.0-beta-windows-hip.exe`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.0-beta-windows-hip.exe) |
| **Linux** | NVIDIA RTX, pilote 580 ou plus récent | [`Strata-0.2.0-beta-linux-nvidia-cuda13.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.0-beta-linux-nvidia-cuda13.AppImage) |
| | NVIDIA, carte plus ancienne ou pilote antérieur à 580 | [`Strata-0.2.0-beta-linux-nvidia-cuda12.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.0-beta-linux-nvidia-cuda12.AppImage) |
| | AMD Radeon série RX 9000 ou Radeon AI PRO R9700 | [`Strata-0.2.0-beta-linux-amd-rx9070.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.0-beta-linux-amd-rx9070.AppImage) |
| | AMD Radeon série RX 7000 | [`Strata-0.2.0-beta-linux-amd-rx7900.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.0-beta-linux-amd-rx7900.AppImage) |
| **Mac** (Apple Silicon, macOS 15 ou plus récent) | intégrée | [`Strata-0.2.0-beta-mac.dmg`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.0-beta-mac.dmg) |

**Si vous hésitez :** pour NVIDIA, prenez `cuda13` si la version de votre pilote est 580 ou plus récente, sinon `cuda12`.
`cuda12` convient aux cartes plus anciennes et aux pilotes plus anciens. Les fichiers `hip` et `amd` sont pour les cartes
AMD. Les autres cartes AMD (RX 6000, ou RX 7000 sous Windows) n'ont pas encore d'application prête à l'emploi : voir
[Cartes AMD](docs/AMD_HIP.md).

Ensuite :

- **Windows :** lancez le `.exe`. Il s'installe pour votre utilisateur et ne demande pas de droits d'administrateur. Si
  Windows affiche « Windows a protégé votre ordinateur », choisissez **Informations complémentaires**, puis **Exécuter
  quand même** (l'installateur n'est pas encore signé).
- **Linux :** rendez le fichier exécutable (`chmod +x`) et lancez-le. Il a besoin de FUSE 2 : `sudo apt install libfuse2`
  (Ubuntu 22.04) ou `libfuse2t64` (Ubuntu 24.04 et plus récent).
- **Mac :** ouvrez le `.dmg` et faites glisser Strata dans Applications.

Sur le premier écran, l'application analyse votre PC et recommande un modèle adapté. Appuyez sur Installer. Le modèle est un
gros téléchargement (environ 40 à 70 Go) qui reprend là où il s'est arrêté en cas d'interruption. Quand vous fermez la
fenêtre, Strata et le moteur s'arrêtent.

### Option B : le Terminal (une ligne)

Le Terminal est la commande `strata`, pensée pour les scripts, les serveurs et les agents d'IA.

Linux et Mac :

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh
```

Windows (PowerShell) :

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

Le script analyse votre PC, télécharge le Terminal, vérifie son empreinte SHA-256 et le décompresse dans votre dossier
personnel. Il ne demande pas de droits d'administrateur. Sous Windows, ouvrez ensuite une nouvelle fenêtre de terminal.
Puis lancez Strata :

```sh
strata launch
```

Il pose une question à la fois : modèle, contexte, images, qui peut se connecter. Ensuite il télécharge le moteur pour votre
carte graphique (signé, et lancé à titre d'essai avant usage) ainsi que le modèle, puis commence à servir. Pour les scripts
et les agents d'IA, utilisez des options :

```sh
strata launch --model q2_0 --yes --no-tui
```

Pour lire le script d'installation avant de l'exécuter, téléchargez-le d'abord avec `curl -fsSL <address> -o install.sh`,
lisez-le, puis lancez `sh install.sh`. Tous les écrans et toutes les options : [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md).

### Option C : laissez votre IA l'installer

Collez ceci dans Claude Code, Cursor, Codex, GitHub Copilot ou un autre assistant de programmation :

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

L'assistant vérifie votre carte graphique, votre RAM et votre disque, choisit le modèle adapté, l'installe, le démarre et
vous explique comment connecter vos applications. Les outils d'IA peuvent aussi installer, démarrer et arrêter Strata via son
[serveur MCP](docs/MCP_SERVER.md).

Vous préférez le compiler vous-même ? Voir [Compiler depuis les sources](#compiler-depuis-les-sources).

## Modèles

L'application et `strata launch` recommandent un modèle selon votre mémoire. C'est un seul modèle en plusieurs tailles,
plus ou moins compressé. Les petites tailles sont plus rapides. Les grandes sont un peu plus intelligentes.

### Peu de mémoire ? Essayez Gyro

[Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) (par agentionai) est la façon la plus légère de faire tourner ce modèle.

| Taille | Mémoire nécessaire (RAM + VRAM) | Téléchargement | Ce que c'est |
| --- | ---: | ---: | --- |
| **Gyro-S** | 27,6 Go | 58,5 Go | Pour les PC où rien d'autre ne tient. La qualité est inférieure à celle d'IQ2_XS. |
| **Gyro-M** | 35 Go | 92,0 Go | À peu près la qualité d'IQ3_XXS avec environ 20 % de mémoire GPU en moins. Il écrit les réponses 17 à 21 % plus lentement que Q2_0 sur une carte de 12 Go et sur une R9700. |

Gyro fonctionne sur les cartes NVIDIA et AMD (expérimental sur AMD), pas sur Intel Arc ni sur Mac. Détails et mesures :
[docs/GYRO.md](docs/GYRO.md). Lancez-le avec `strata launch --model gyro-s`.

### Les autres tailles

| Votre mémoire (RAM + VRAM) | Prenez | Remarques |
| --- | --- | --- |
| **32 Go** | **Gyro-S**, ou le **[Coder](docs/MODELS.md#coder)** pour le code | Le Coder demande 32 Go de RAM et garde la moitié des experts, il est donc moins bon hors du code |
| **40 Go** | **[Swift 1.5](docs/MODELS.md#swift-15)** IQ2_XS (39,2 Go), ou **Gyro-M** (35 Go) | Swift réfléchit moins longtemps avant de répondre. Gyro-M approche la qualité d'IQ3_XXS, mais il écrit les réponses 17 à 21 % plus lentement que Q2_0 |
| **48 Go** | **IQ3_XXS** (47,0 Go) | Plus lent, meilleur |
| **64 Go** | **IQ3_S** (54,8 Go), l'original ou Swift 1.5 | Le plus lent, le meilleur. Il égale le modèle complet dans les tests publiés |
| **96 Go ou plus** | **UD-IQ4_XS** d'Unsloth (environ 72 Go de RAM) jusqu'à **Q8_0** (environ 136 Go), 4 à 8 bits | Plus proche du modèle complet. Avec moins de RAM, Strata lit une partie depuis le SSD, ce qui est beaucoup plus lent |

Avoir moins de RAM que ce qu'indique une ligne peut encore fonctionner : Strata passe tout seul en
[mode peu de RAM](docs/MODELS.md#a-big-graphics-card-and-little-ram). Attendez-vous à plus de lenteur, et à bien plus avec
une petite carte graphique.

Autres choix :

- **[Swift 1.5](docs/MODELS.md#swift-15)** est une version affinée qui réfléchit moins longtemps avant de répondre, pour une qualité et un besoin en RAM à peu près identiques.
- **[NVFP4](docs/NVFP4.md)** est la version 4 bits de NVIDIA pour les cartes NVIDIA. Elle demande environ 80 Go de RAM pour garder tous les experts en RAM.
- **[Qwen3.6-35B-A3B](docs/QWEN36.md)** est un modèle plus petit et différent, qui lit les images et peut répondre à plusieurs requêtes à la fois.
- **[L'Uncensored IQ3_XXS d'OrcaRouter](docs/MODELS.md#orcarouter-uncensored-iq3_xxs)** se configure à la main.

Toutes les tailles, les téléchargements et ce qui tient où : [docs/MODELS.md](docs/MODELS.md).

## Quelle est sa vitesse ?

Mesuré avec une carte graphique de 12 Go (RTX 5070, Ryzen 5 7600, 64 Go de RAM). Voir les résultats d'autres cartes :
[vitesse de chaque taille](docs/MODELS.md#how-fast-is-each-size), [résultats de la communauté](docs/COMMUNITY_BENCHMARKS.md).

| Modèle | Les réponses s'affichent à | Lit votre prompt à |
| --- | --- | --- |
| **Q2_0** | `████████████` 94 tokens/s | `████████████` 2 650 tokens/s |
| **IQ2_XS** | `██████████░░` 79 tokens/s | `█████████░░░` 2 090 tokens/s |
| **IQ3_XXS** | `████████░░░░` 62 tokens/s | `████████░░░░` 1 750 tokens/s |
| **IQ3_S** | `███████░░░░░` 53 tokens/s | `███████░░░░░` 1 620 tokens/s |
| **Coder** | `███████░░░░░` 55 tokens/s | `██████████░░` 2 180 tokens/s |

Un token vaut environ ¾ de mot, donc 60 tokens par seconde, c'est plus rapide que votre lecture. « Lit votre prompt »
indique la vitesse à laquelle Strata absorbe un long document, du code ou un historique de discussion.

Mesuré avec une carte AMD de 16 Go (RX 9070 XT, Ryzen 9 3900X, 47 Go de RAM) :

| Modèle | Les réponses s'affichent à | Lit votre prompt à |
| --- | --- | --- |
| **Q2_0** | `████████░░░░` 60 tokens/s | `█████░░░░░░░` 1 160 tokens/s |
| **IQ2_XS** | `███████░░░░░` 52 tokens/s | `█████░░░░░░░` 1 110 tokens/s |
| **Coder** | `██████░░░░░░` 44 tokens/s | `██████░░░░░░` 1 420 tokens/s |

Les barres utilisent la même échelle dans les deux tableaux. Une carte avec plus de VRAM est plus rapide. Une RTX 3090
(24 Go) devrait écrire environ 100 à 140 tokens par seconde. La version du moteur derrière chaque chiffre est indiquée dans
[DETAILS.md](docs/DETAILS.md#speed-measured).

## Utilisation

### Dans l'application

L'application comprend un **Chat**, un **Monitor** du modèle, de votre GPU, de votre CPU et de votre RAM, une page **Models** et les **Settings**. Le chat peut travailler dans un dossier de votre PC. En mode **Manual**, il demande
avant de modifier un fichier ou d'exécuter une commande. Tapez `/loop 10m <what to do>` pour répéter un message toutes les
10 minutes tant que le chat est ouvert.

Le chat peut chercher sur le web avec DuckDuckGo (ou Bing), sans compte. Vous pouvez ajouter une clé de recherche sous
**Settings, Tools**. La recherche est la seule fonction qui envoie quelque chose hors de votre PC (les mots recherchés).

Pour faire tourner deux modèles, activez **Run multiple models** (exécuter plusieurs modèles) et donnez à chacun sa propre carte de la même marque.

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="The Strata app's Monitor tab next to a coding agent"><br>
<sub>Le <b>Monitor</b> de l'application Strata (à gauche) pendant qu'un agent de code écrit le jardin de pagode de la vidéo (à droite)</sub></p>

- **Dans le navigateur :** ouvrez `http://127.0.0.1:8080`. Vous y trouvez **Chat**, un **Monitor** en direct du
  modèle et de votre GPU/CPU/RAM, et **About** avec les réglages et les adresses.
- **Vos applications et agents de code :** ajoutez un fournisseur « OpenAI-compatible » avec l'URL de base
  **`http://127.0.0.1:8080/v1`**. N'importe quelle clé API et n'importe quel nom de modèle fonctionnent.
  - Applications qui utilisent l'API d'Anthropic : `http://127.0.0.1:8080/v1/messages` (Claude Code :
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
  - Codex CLI et les autres applications qui utilisent l'API OpenAI Responses : `/v1/responses`
    ([configuration](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Réflexion :** choisissez **off, low, medium ou high** dans le menu du chat ou dans le « reasoning effort » de
  votre application. Off est le plus rapide. High est le meilleur pour les questions difficiles.
- **Images :** répondez oui à « Images? » pendant l'installation. Ensuite cliquez sur **Picture** dans le chat, ou
  joignez des images dans votre application. Les cartes AMD lisent les images sous Linux via le processeur ; sous
  Windows, pas encore.
- **Depuis votre téléphone ou un autre PC :** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>` pour IPv4,
  ou utilisez `--host ::` pour IPv6. Définissez toujours une clé. Une URL IPv6 met l'adresse entre crochets :
  `http://[2001:db8::7]:8080/v1`.
- **Une requête à la fois :** par défaut, Strata répond à une requête et les autres attendent. Pour répondre à
  plusieurs en même temps, mettez `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)). Sur une carte de 12 Go, chaque
  réponse devient alors plus lente.
- **Longs prompts :** Strata lit en entier le premier message d'une discussion, environ 1 minute pour 30,000 tokens.
  Les messages suivants démarrent en quelques secondes.

Plus d'infos : [où sont stockées vos discussions](docs/INSTALL.md#where-things-are-stored), [l'API](docs/DETAILS.md#using-it).

### Optionnel : plus de vitesse sur une seule carte quand tout le modèle tient en RAM

Deux réglages optionnels (de RubenTorney, #1904 et #1905). Ils aident si vous utilisez une seule carte graphique et que votre RAM contient tous les experts (le mode arena de Strata, par exemple Q2_0 ou IQ3_XXS sur un PC de 64 Go). Ouvrez votre configuration `strata-<model>.json` dans le dossier de données de Strata ([où](docs/INSTALL.md#where-things-are-stored)) et ajoutez :

- **Génération plus rapide :** `"--adapt-async", "1"` dans la liste `"args"`. Le cache d'experts est alors mis à jour dans un thread auxiliaire au lieu de mettre la réponse en pause. Mesuré sur une RTX 5070 (Windows, Q2_0) : +4-5 % de génération, +7-10 % sur le code et les longs documents. Les réponses peuvent légèrement différer de celles par défaut (elles restent identiques d'une exécution à l'autre).
- **Lecture du prompt plus rapide sous Windows :** `"STRATA_PIN_BY_NEED": "1"` dans le bloc `"env"`. Windows ne peut verrouiller qu'environ la moitié de la RAM pour la carte ; ce réglage verrouille d'abord les experts dont la carte a vraiment besoin. Mesuré sur la RTX 5070 : +4,4 % en lecture du prompt.

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

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

Un petit assistant devine aussi les mots suivants. Le grand modèle les vérifie tous d'un coup et garde les bons. Vous
obtenez la même réponse, 1,6 à 1,8 fois plus tôt. Les textes longs sont lus par gros morceaux, jusqu'à 8 192 tokens à la
fois, à plus de 1 000 tokens par seconde.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

L'explication plus longue : [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Chaque élément avec ses chiffres :
[les détails](docs/DETAILS.md#how-it-works) et l'[article](docs/paper/Strata-Paper.pdf).

## Un problème ?

- **Mon PC s'est figé au premier démarrage de Strata.** Il charge le modèle en RAM. Patientez et ne fermez pas la fenêtre.
  Toujours figé après 10 minutes ? Redémarrez le PC, fermez les autres programmes et réessayez, ou choisissez une taille plus petite.
- **Ça s'est arrêté pendant le téléchargement ou l'installation.** Relancez l'application ou `strata launch`, ou relancez
  `START-HERE.bat` (ou `./setup.sh`). Cela reprend là où ça s'était arrêté.
- **C'est très lent, le voyant du disque clignote sans arrêt, ou il affiche « the engine stopped unexpectedly ».** Votre PC
  n'a pas assez de RAM libre. Fermez d'autres programmes (les navigateurs en consomment beaucoup) ou choisissez une taille plus petite, comme Q2_0 ou IQ2_XS.
- **Il indique qu'il n'y a pas de moteur pour ce PC.** Dans l'application, ouvrez **Settings, Engine** et appuyez sur **Update engine** (mettre à jour le moteur). Dans le Terminal, lancez `strata update`. `strata doctor` montre quels fichiers du moteur sont installés.
- **Il indique que le port 8080 est déjà utilisé.** Strata tourne déjà. Cherchez sa fenêtre.

Autres problèmes et solutions : [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Toujours bloqué ? Ouvrez une
[issue](https://github.com/Niko1221/Strata/issues). Dans l'application, joignez le fichier obtenu par **Settings, Engine, Diagnostics, View logs, Save logs**. Dans le Terminal, collez la sortie de `strata doctor`. Si
vous avez lancé depuis les sources, joignez `strata-<model>.log` du dossier Strata. Pour un problème de sécurité,
signalez-le en privé : [SECURITY.md](SECURITY.md).

## Remerciements et licence

Le modèle est [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), de l'équipe Qwen. Il a été compressé par
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5), agentionai (Gyro)
et Unsloth. Strata utilise des parties de [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Tous les
remerciements : [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata est open source sous [licence MIT](LICENSE).
Quelques parties et tous les modèles ont leurs propres licences ([lesquelles](docs/HOW_IT_WORKS.md#license)).

## Contribuer

Vous avez trouvé un bug, ou vous voulez proposer une modification ou un benchmark ? Lisez d'abord [CONTRIBUTING.md](CONTRIBUTING.md).
Il indique ce qu'il faut inclure pour que d'autres puissent vérifier votre travail.

## Soutenir Strata

Strata est gratuit et sous licence MIT. Je le construis et je le teste sur le matériel que je peux obtenir. S'il tourne bien sur votre PC, un café m'aide
à continuer de tester de nouvelles cartes et à corriger ce que vous signalez. Aucune obligation. L'utiliser et signaler ce que vous constatez aide tout autant.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>

Vous souhaitez sponsoriser Strata ? Contactez-moi via [GitHub](https://github.com/Niko1221).
