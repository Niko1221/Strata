<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Une IA puissante sur votre propre ordinateur.</h3>
<p align="center">Discutez, programmez et travaillez avec des images, en local.</p>

<p align="center">
  <a href="#modèles">Modèles</a> · <a href="#installation"><b>Installation</b></a> ·
  <a href="#matériel">Matériel</a> · <a href="#outils">Outils</a> ·
  <a href="#aide">Aide</a> · <a href="#soutenir-strata">Soutenir</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Télécharger Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="Licence MIT"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Soutenir Strata sur Buy Me a Coffee"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <b>Français</b> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a></sub></p>

**Gratuit, open source, et à vous de l'exécuter.** Utilisez Strata sur votre bureau ou avec vos outils de programmation préférés. Aucun compte nécessaire.

> **Fork personnel (c-OOL-io/Strata-WebSearch).** Ce fork ajoute **la recherche web et la lecture complète des pages au chat intégré et au frontend serve** : la web app légère du terminal, où l'amont n'a pas de recherche (la recherche amont vit dans l'App). Construit comme le chat de Strata lui-même : **entièrement désactivé par défaut** (tant que vous ne l'activez pas, rien en rapport avec la recherche ne quitte le PC), **aucun token ajouté** pour les chats qui ne l'activent pas, et le texte de recherche est plafonné à une petite part de la fenêtre de contexte du modèle (les valeurs par défaut suivent `max_context`, les instances à petit contexte restent sûres).
>
> - **Moteurs :** DuckDuckGo par défaut (sans clé), SearXNG (votre instance auto-hébergée), Brave, Tavily, Serper (Google), Exa et Perplexity avec vos propres clés. Les clés restent dans le navigateur ; rien en rapport avec la recherche n'est stocké sur le serveur.
> - **Optimiseur de requête :** une formulation conversationnelle est réécrite en une requête de recherche propre par un appel rapide à votre propre modèle local, dans la langue où vous avez écrit. En cas d'échec, le message brut est recherché à la place. Si ce qui a été cherché diffère de ce que vous avez tapé, la ligne Sources l'affiche.
> - **Lecture des pages de résultats :** les meilleures pages de résultats sont récupérées et injectées en texte extrait (nav/pied/scripts retirés, `main`/`article` privilégiés), avec les modes de lecture Direct / Jina en repli / Jina pour toutes, et un avertissement explicite dès qu'un tiers pourrait voir vos URL.
> - **Réponses auditable :** chaque réponse reçoit une ligne `Sources` (moteur, requête exacte, liens cliquables, badges `read`) qui suit l'export Markdown, et une recherche échouée laisse un avertissement nommé par cause (clé manquante, moteur injoignable, contrôle bot) au lieu d'échouer silencieusement.
> - **Durci :** `/search` se trouve derrière la protection own-page de l'application (origines étrangères rejetées), contrôles d'URL http(s) uniquement pour que la lecture de pages ne puisse jamais devenir un lecteur de fichiers locaux, le texte web injecté est présenté comme non fiable, et le backend (`serve/websearch.py`) est stdlib uniquement, aucune nouvelle dépendance, Windows/Linux/macOS.
>

> Deux points avant tout téléchargement :
>
> 1. **Les liens de téléchargement ci-dessous sont l'app upstream d'origine** de Niko1221/Strata. Les binaires prêts à l'emploi **ne contiennent pas** la recherche de ce fork. La version avec recherche n'existe que dans le source : `git clone`, branche `web-search`, puis START-HERE.bat / setup.sh.
> 2. **Ce fork convient aussi à la route Terminal économe en ressources** : ceux qui veulent seulement le serve/CLI léger sur un serveur headless (pas d'app, pas d'UI de chat) l'obtiennent de ce même source, avec la recherche intégrée au frontend serve.
>
> **Installation via CLI** (ce fork, branche `web-search`) : clonez et installez exactement comme Strata depuis le source, avec la recherche intégrée.
> Le guide d'installation propre à ce fork (terminal / source, et mise à jour) est dans [docs/INSTALL.FORK.md](docs/INSTALL.FORK.md) :
>
> ```sh
> git clone -b web-search https://github.com/c-OOL-io/Strata-WebSearch.git && cd Strata-WebSearch
> ./setup.sh          # first run: .venv, the engine build, the model; then it serves
> ```
>
> Les résultats de bench sur RTX 4080 SUPER / Ryzen 9 7950X3D sont dans `bench/`.

<p align="center">
  <img src="docs/media/web-search-chat.png" width="360" alt="Recherche web dans le chat Strata">
  <img src="docs/media/web-search-settings.png" width="360" alt="Réglages de recherche web : moteurs et clés">
</p>
<p align="center"><sub>À gauche: chat répondu avec la ligne Sources sous la réponse (moteur utilisé, requête réellement cherchée), activable via le globe du composeur.<br>À droite: section Web search des réglages: activation, optimisation de la requête, choix du moteur (DuckDuckGo / SearXNG / Brave / Tavily / Serper / Exa / Perplexity), champ de clé, lecture des pages.</sub></p>

## Modèles

Nous recommandons **d'abord ISTA-DASLab**, puis **Unsloth** et d'autres fournisseurs, lorsqu'un modèle compatible tient en mémoire.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Fournisseur | Choix |
| --- | --- |
| **ISTA-DASLab · recommandé** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder : IQ1\_M](docs/MODELS.md#coder) pour programmer. |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Utilise moins de mémoire, mais plus lent. |
| **Autres options** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Comparer les tailles, la mémoire nécessaire et la compatibilité](docs/MODELS.md).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. Ce sont les paquets actuellement documentés pour ce modèle.

[Guide du modèle et compatibilité](docs/QWEN36.md).

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

Dans l'**application de bureau** et dans le **Terminal**. Ce qui ne tient pas en mémoire est lu depuis votre SSD : il faut donc beaucoup d'espace disque libre.

| Jeu (IST-DAS) | Téléchargement | Espace disque libre nécessaire |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | environ 245 GB |
| 2.00 bpw | 257 GB | environ 265 GB |
| 2.25 bpw | 274 GB | environ 280 GB |

Fonctionne sur les NVIDIA RTX 30, 40 et 50 (Windows et Linux) et sur l'AMD Radeon AI PRO R9700 (Linux). L'installation télécharge les fichiers et le moteur DeepSeek, puis démarre le modèle.

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

Mesuré avec le jeu 2.00 bpw sur Strata 0.2.5. « Sortie » est la vitesse de la réponse, « Prompt » la vitesse à laquelle Strata lit votre texte.

| Matériel | Sortie | Prompt |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB de RAM | 4 à 6 tokens/s | environ 110 tokens/s (4K) |
| RTX 3060 12 GB, 46 GB de RAM | 9 à 11 tokens/s | environ 155 tokens/s (4K) |
| 1x Radeon AI PRO R9700, 251 GB de RAM | 21 à 25 tokens/s | 320 à 360 tokens/s (8K) |
| 4x Radeon AI PRO R9700 | 74 à 77 tokens/s | 400 à 450 tokens/s (8K) |

[Mémoire, disque et détails](docs/DEEPSEEK41.md) · [Notes de version](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

Vous ne savez pas ce qui tient ? [Choisir un modèle pour votre ordinateur](docs/MODELS.md#pick-by-memory).

## Installation

<a id="option-a-the-strata-app"></a>

### Application de bureau

**[Télécharger Strata pour Windows, Linux ou Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Installez l'application, choisissez un modèle et commencez à discuter. Il y a un seul installateur par système. Au premier lancement, l'application télécharge le moteur de votre carte graphique. [Aide pour l'installateur](docs/INSTALL.md#the-app).

<sub>Fichiers: `Strata-0.2.6-windows-x64-setup.exe` · `Strata-0.2.6-linux-x86_64.AppImage` · `Strata-0.2.6-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Application de bureau Strata avec une discussion et un aperçu du code en direct côte à côte">
  <br><sub>Discussion, fichiers et aperçu en direct dans une seule fenêtre.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Application Terminal

Vous préférez le terminal ? Installez Strata, puis lancez **`strata launch`**.

<details open>
<summary><b>Commandes d'installation</b></summary>

**Linux / Mac**

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

Sous Windows, ouvrez ensuite un nouveau terminal. Puis :

```sh
strata launch
```

Suivez les invites pour télécharger et démarrer un modèle. [Guide du Terminal](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Application Terminal de Strata affichant l'activité du modèle, la vitesse, la mémoire et les requêtes récentes">
  <br><sub>Suivez votre modèle et votre matériel pendant qu'il tourne.</sub>
</p>

<a id="build-from-source"></a>

[Compiler depuis les sources](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Guide d'installation complet](docs/INSTALL.md)

<a id="what-you-need"></a>

## Matériel

| Matériel pris en charge | Guide |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50** | [Installation NVIDIA](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [Installation AMD](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · expérimental** | [Installation RX 6000](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Installation Intel](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 ou plus récent** | [Installation sur Mac](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Installation Strix Halo](docs/STRIX_HALO.md) |

<a id="using-it"></a>

## Outils

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, et d'autres via les API compatibles OpenAI et Anthropic.

Lancez **`strata connect`** pour voir la configuration de votre outil. [Guide de connexion](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Aide

[Tous les guides](docs) · [Dépannage](docs/TROUBLESHOOTING.md) · [Signaler un bogue](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [Comment ça marche](docs/HOW_IT_WORKS.md) · [Contribuer](CONTRIBUTING.md)

## Soutenir Strata

**Aidez à garder l'IA locale ouverte à tous.** Un café aide à financer le développement et les tests sur davantage de matériel.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee : soutenir Strata" height="48"></a>

[Licence MIT](LICENSE) · [Crédits et licences des modèles](docs/HOW_IT_WORKS.md#credits) · [Confidentialité](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>Pour les outils d'IA et les lecteurs automatiques : <a href="AGENTS.md">référence technique</a> · <a href="docs/AI_SETUP.md">instructions d'installation</a>.</sub>

**Parrainages, partenariats matériel ou collaborations :** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).
