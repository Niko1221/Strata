# Hakua / Strata

[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)

IA locale sous Windows pour discuter, utiliser des agents et rechercher avec des sources. Hakua est le fork de Niko1221/Strata maintenu indépendamment par zapabob, sans approbation annoncée du projet d’origine.

## Version et téléchargements

<!-- release-060-candidate -->
Ce code est le candidat d’intégration 0.6.0. Application, Terminal, Control et sources du moteur natif portent 0.6.0. Les vérifications de compilation, d’installation et de publication sont en cours ; le numéro ne prouve pas la disponibilité des fichiers. La [version 0.5.9](https://github.com/zapabob/Strata/releases/tag/v0.5.9) vérifiée est conservée. Le compte rendu de livraison précise le statut réel de 0.6.0.

[0.6.0](https://github.com/zapabob/Strata/releases/tag/v0.6.0) · [Strata-0.6.0-windows-x64-setup.exe](https://github.com/zapabob/Strata/releases/download/v0.6.0/Strata-0.6.0-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.6.0/Strata-0.6.0-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.0/SHA256SUMS.txt)

## Installation et mise à jour

Utilisez l’installeur NSIS Windows x64 de ce fork. Il s’installe par utilisateur avec ses propres Node, Python et Control, sans outils de développement système. Le moteur CUDA signé est un composant distinct. Le tar.7z doit contenir le même installeur, les composants, les instructions et les sommes de contrôle, sans poids de modèles ni données personnelles. Extraire 7z puis tar et vérifier BUNDLE-SHA256SUMS.txt.

Terminez les requêtes, déchargez les modèles, sauvegardez les données et quittez aussi l’icône de notification. Réinstallez dans le dossier existant. Conservez modèles, conversations, mémoire, outils, clés et autorisations. Importez le moteur compressé avec l’option de fichier existant, puis chargez un modèle compatible ou un GGUF local. Le nom du modèle ne suffit pas à établir sa compatibilité. [Installation et retour arrière](docs/HAKUA_INSTALL.md).

## Fonctions conservées

`/btw` lance un véritable sous-agent sans outils pour une question annexe, avec son propre arrêt. `/research` et `/deepresearch` recueillent des sources. La recherche peut utiliser RustXNG Full fourni. Les requêtes sont transmises aux fournisseurs externes activés ; la recherche n’est donc pas entièrement hors ligne. Le nombre de sources ne garantit pas l’exactitude.

L’apprentissage des compétences de travail et leur application automatique sont deux options distinctes, désactivées dans les nouveaux profils. Préférences, compétences privées modifiables, révisions et annulation sont conservées. Les opérations sur fichiers, l’historique et la mémoire délimités, les chemins PrismML/TurboQuant compatibles, la mémoire d’experts bornée, l’intégrité des packs et les diagnostics restent présents. [Compétences](docs/WORKFLOW_SKILLS.md) · [Recherche](docs/BROWSER_SEARCH.md).

## Intégration officielle et compatibilité

La référence officielle est [09b6038869](https://github.com/Niko1221/Strata/commit/09b603886925994d0d4461857b01321ef3aa431c), génération 0.2.14 / moteur 0.1.60 : réseau, choix du brouillonneur, noyaux de prompts, mémoire et correctifs de plateforme. Les versions du fork sont indépendantes. Les chemins DeepSeek supprimés sont conservés comme extension ; sans exécutable dédié, le chargement doit être refusé. Ce paquet Windows ne valide ni entrée native d’images, ni distribution d’un moteur natif DeepSeek, ni inférence Linux/Mac.

## Protection et limites des mesures

Les seuils restent à 3 GiB de RAM disponible, 4 GiB de marge d’engagement mémoire, au moins 768 MiB de réserve GPU et des minimums de disque. La protection RAM Windows surveille chargement et génération. 32 Go de RAM et 16 Go de VRAM ne garantissent pas tout modèle ou contexte. Fenêtre complète, qualité générale et concurrence GPU restent non qualifiées : releaseQualified=false. L’installeur n’a pas de signature Windows Authenticode ; signatures de composants et de flux sont distinctes. Les [mesures historiques](docs/HAKUA_RELEASE.md) gardent leurs versions d’origine.

## Documentation et développement

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

[Niko1221/Strata](https://github.com/Niko1221/Strata) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) · [Sources et mentions tierces](third_party/)
