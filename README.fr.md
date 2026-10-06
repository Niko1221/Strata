<h1 align="center">Strata</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · **Français** · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>Faites tourner un modèle d'IA de 125 milliards de paramètres sur votre propre PC de jeu</b><br>
Carte graphique NVIDIA ou AMD (12 Go ou plus) · Windows ou Linux · gratuit et open source</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="Un jardin de pagode en voxels écrit par le modèle de Strata, qui tourne dans le navigateur"></a><br>
<sub>Un jardin de pagode en voxels, généré en une seule requête sur une RTX 5070 avec Strata (IQ3_S, contexte de 128K) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">vidéo complète (49 s)</a></sub></p>

Strata fait tourner **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** sur un PC normal. C'est
un grand modèle d'IA, très capable, qui demande d'habitude un serveur. Il discute, écrit du code, lit des images et
fonctionne avec vos applications et vos agents de code. Rien ne quitte votre PC.

## Quelle est sa vitesse ?

Nous l'avons mesurée sur deux PC de jeu ordinaires. Un token correspond à environ ¾ d'un mot.

- **Écrit les réponses :** la vitesse à laquelle la réponse s'affiche dans une courte discussion. 60 tokens par seconde, c'est plus rapide que votre lecture.
- **Lit votre prompt :** la vitesse à laquelle il absorbe ce que vous envoyez (ici un document, du code ou un historique de discussion de 32K tokens).

<table>
<tr><th>NVIDIA : RTX 5070 (12 Go), Ryzen 5 7600, 64 Go de RAM</th><th>AMD : RX 9070 XT (16 Go), Ryzen 9 3900X, 47 Go de RAM</th></tr>
<tr><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| Taille | Écrit les réponses | Lit votre prompt |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA : Q2_0 avec le moteur 0.1.36, les autres lignes avec 0.1.26 (réponses de 4K, prompts de 32K). Les tableaux
complets sont dans [DETAILS.md](docs/DETAILS.md#speed-measured). Une carte avec plus de VRAM va plus vite : une
RTX 3090 (24 Go) devrait écrire environ 100-140 tokens par seconde. Longues discussions et autres cartes :
[vitesse de chaque modèle](docs/MODELS.md#how-fast-is-each-size), [résultats de la communauté](docs/COMMUNITY_BENCHMARKS.md).

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>Strata est gratuit. S'il tourne bien sur votre PC, un café aide à poursuivre le travail.</sub></p>

## Ce qu'il vous faut

| | |
| --- | --- |
| **Carte graphique** | **NVIDIA** GeForce RTX série 20, 30, 40 ou 50, ou **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 ou série RX 6800 / 6900. Il faut **12 Go de VRAM ou plus**. |
| **RAM** | 32 Go ou plus. Votre RAM décide [quel modèle](#quel-modèle-choisir-) tient. Avec 64 Go, toutes les tailles tournent. |
| **Disque** | Environ 80 Go libres. Prenez un SSD si possible : le premier démarrage est bien plus rapide. |
| **Système** | Windows 10 / 11 ou Linux, avec un pilote graphique NVIDIA ou AMD à jour. |

L'installateur s'occupe de tout le reste. Deux ou trois cartes peuvent se partager le modèle ([multi-GPU](docs/MULTI_GPU.md)).

Expérimental, écrit et testé par des membres de la communauté sur leurs propres machines :

- **Cartes graphiques plus anciennes** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT) : [Older GPUs](docs/OLDER_GPUS.md).
- **Intel Arc**, compilé depuis les sources sous Linux : [Intel Arc](docs/INTEL_ARC.md).
- **AMD Ryzen AI Max (Strix Halo)**, compilé depuis les sources sous Linux : [Strix Halo](docs/STRIX_HALO.md).
- **Processeurs plus anciens sans AVX2** : ça marche, mais lentement. [Older CPUs](docs/INSTALL.md#older-cpus-experimental).

La liste complète : [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Installation

### Laissez votre IA l'installer

Vous utilisez un assistant de code IA (Claude Code, Cursor, Codex, GitHub Copilot, ...) ? Collez-lui ceci :

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

Il vérifie votre carte graphique, votre RAM et votre disque, et choisit le modèle qui convient. Ensuite il l'installe,
le démarre et vous explique comment connecter vos applications. Les outils d'IA peuvent aussi installer, démarrer et
arrêter Strata grâce à son [serveur MCP](docs/MCP_SERVER.md).

### Ou faites-le vous-même

[Téléchargez Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) et décompressez-le (ou faites un `git clone`).
**Windows :** double-cliquez sur **`START-HERE.bat`**. **Linux :** lancez **`./setup.sh`** dans le dossier Strata.

Les étapes sont les mêmes pour NVIDIA et AMD. L'installateur détecte votre carte et installe le bon moteur pour elle.
Il vous pose quelques questions :

- quel modèle et quelle taille,
- combien de contexte (la quantité de texte que le modèle garde en tête),
- s'il doit lire les images.

Appuyez sur Entrée à chaque fois pour garder la réponse recommandée. Ensuite il télécharge le modèle (environ 70 Go)
et le démarre. Si le téléchargement s'arrête, relancez-le : il reprend là où il s'était arrêté. Votre navigateur
ouvre l'application Strata à l'adresse `http://127.0.0.1:8080`.

> **Pendant le démarrage du modèle, votre PC peut ralentir ou ne plus répondre pendant 1-3 minutes** (plus longtemps la première fois).
> Strata charge 35-55 Go dans votre RAM et en réserve une partie pour la carte graphique. C'est normal. Attendez, et
> ne fermez pas la fenêtre. La fenêtre montre ce que fait Strata.

**La fois suivante**, relancez `START-HERE.bat` (ou `./setup.sh`). Il démarre tout de suite et ne télécharge rien
deux fois. Fermez sa fenêtre pour arrêter le modèle. `UPDATE.bat` (`./update.sh`) met Strata à jour sans le démarrer.
Mises à jour, Docker, plusieurs cartes, emplacement des fichiers et toutes les options : [docs/INSTALL.md](docs/INSTALL.md).

## Quel modèle choisir ?

L'installateur en recommande un selon votre RAM. Le même modèle existe en plusieurs tailles, plus ou moins
compressées. Les petites tailles sont plus rapides. Les grandes sont un peu plus intelligentes.

| Votre RAM | Prenez | Pourquoi |
| --- | --- | --- |
| **32 Go** | **Coder** | il tient dans 32 Go et il est fait pour le code (avec une carte de 24 Go, Q2_0 et IQ2_XS tournent aussi) |
| **48 Go** | **IQ2_XS** (ou Q2_0, le plus rapide) | les plus grandes tailles ne tiennent pas |
| **64 Go** | **IQ2_XS** (recommandé), ou IQ3_XXS / IQ3_S | toutes les tailles tiennent ; IQ3_S est le meilleur et le plus lent |
| **96 Go ou plus** | **IQ3_S**, ou l'UD-IQ4_XS d'Unsloth (~4 bits) | de la place pour les plus grandes tailles, avec tout le reste ouvert |

- **[Coder](docs/MODELS.md#coder) :** une version pour le code, avec la moitié des experts en moins. Elle atteint 91 %
  du score SWE-bench Verified du modèle complet (mesuré par ses auteurs) et tient dans 32 Go de RAM. Elle est moins
  bonne en dehors du code, y compris pour le chinois et les autres textes CJK (#438). Pour ces cas, prenez Q2_0,
  IQ2_XS ou IQ3_S, qui gardent tous les experts.
- **[Swift 1.5](docs/MODELS.md#swift-15) :** un fine-tune qui réfléchit bien moins longtemps avant de répondre. Vous
  avez la réponse plus tôt, avec à peu près la même qualité.
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs)** : la version ~4 bits d'Unsloth, entre IQ3_S et
  UD-Q4_K_XL en qualité. Un téléchargement de 94 Go. Avec moins de ~80 Go de RAM, Strata en lit une partie depuis
  le SSD pendant qu'il répond, il y est donc plus lent (un SSD NVMe aide).
- **[Unsloth UD-Q4_K_XL](docs/MODELS.md#unsloth-ud-q4_k_xl-experimental)** (expérimental) : le plus proche du modèle
  complet. Mais Strata en lit la plus grande partie depuis le SSD pendant qu'il répond, donc il n'écrit que
  7-8.5 tokens/s sur un PC avec 64 Go.
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs) :** vous l'installez à la main.
  Il n'est pas dans le menu de l'installateur.

Tailles, téléchargements et ce qui tient où : [docs/MODELS.md](docs/MODELS.md). Pour ajouter un autre modèle plus
tard, lancez `SETUP.bat` (Linux : `./setup.sh --setup`).

## Utilisation

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="L'onglet Monitor de l'application Strata à côté d'un agent de code"><br>
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
- **Depuis votre téléphone ou un autre PC :** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Définissez
  toujours une clé.
- **Une requête à la fois :** par défaut, Strata répond à une requête et les autres attendent. Pour répondre à
  plusieurs en même temps, mettez `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)). Sur une carte de 12 Go, chaque
  réponse devient alors plus lente.
- **Longs prompts :** Strata lit en entier le premier message d'une discussion, environ 1 minute pour 30,000 tokens.
  Les messages suivants démarrent en quelques secondes.

Plus d'infos : [où sont stockées vos discussions](docs/INSTALL.md#where-things-are-stored), [l'API](docs/DETAILS.md#using-it).

## Un problème ?

- **Mon PC s'est figé au premier démarrage de Strata.** C'est normal pendant le chargement du modèle. Attendez, et ne
  fermez pas la fenêtre. Toujours figé après 10 minutes ? Redémarrez le PC, fermez les autres programmes et
  réessayez, ou choisissez une taille plus petite.
- **Il s'est arrêté pendant le téléchargement ou l'installation.** Relancez `START-HERE.bat` (ou `./setup.sh`). Il
  reprend là où il s'était arrêté.
- **C'est très lent et le voyant du disque clignote sans arrêt, ou il affiche « the engine stopped unexpectedly ».**
  Votre PC n'a pas assez de RAM libre. Fermez les autres programmes (les navigateurs en utilisent beaucoup), ou
  choisissez une taille plus petite (Q2_0 ou IQ2_XS).
- **Il dit que le port 8080 est déjà utilisé.** Strata tourne déjà. Cherchez sa fenêtre.

Autres problèmes et leurs solutions : [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Toujours bloqué ? Ouvrez une
[issue](https://github.com/Niko1221/Strata/issues) et joignez `strata-<model>.log` depuis le dossier Strata. Vous
avez trouvé un problème de sécurité ? Signalez-le en privé : [SECURITY.md](SECURITY.md).

## Comment ça marche ?

Les modèles comme celui-ci tournent d'habitude sur des serveurs avec des centaines de gigaoctets de mémoire
graphique. Votre carte graphique a 12-24 Go. Strata fait tenir le modèle en **répartissant le travail sur tout votre
PC**. Pensez à une cuisine : ce que vous utilisez tout le temps reste sur le plan de travail, et le reste attend dans
le placard.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="Les 24,576 experts du modèle : les plus sollicités sur la carte graphique, tous en RAM, une table de correspondance sur le SSD"></p>

- **Le modèle est une équipe de 24,576 petits spécialistes (les « experts »).** Chaque mot n'en demande que 10.
- **Votre carte graphique** garde les quelques milliers d'experts les plus utilisés. **Votre RAM** les contient tous,
  et **votre processeur** travaille sur les autres en même temps. **Votre SSD** contient une grande table de
  correspondance.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="Un petit assistant devine les mots suivants ; le grand modèle les vérifie tous d'un coup et garde les bons"></p>

- **Deviner, puis vérifier :** un petit assistant devine les quelques mots suivants. Le grand modèle les vérifie
  tous d'un coup. Vous obtenez la même réponse, 1.6-1.8x plus tôt.
- **Les longs textes sont lus par gros morceaux** (jusqu'à 8,192 tokens à la fois), à plus de 1,000 tokens par seconde.

L'explication plus longue : [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Chaque partie et ses chiffres :
[les détails](docs/DETAILS.md#how-it-works) et l'[article](docs/paper/Strata-Paper.pdf).

## Crédits et licence

Le modèle est [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), de l'équipe Qwen. Il a été
compressé par [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5)
et Unsloth. Strata utilise des parties de [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Tous les
crédits : [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata est open source sous [licence MIT](LICENSE).
Quelques parties et chaque modèle ont leur propre licence ([lesquelles](docs/HOW_IT_WORKS.md#license)).

## Soutenir Strata

Strata est gratuit et open source. S'il vous est utile, vous pouvez soutenir son développement :

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>
