"""Rewrite all eight entry READMEs from reviewed fork/upstream requirements."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('H:/Strata/evidence/upstream-sync-20261116')
REFS = {'base': 'f33c8c30588ff00e24b6b63bad0bf6a14da19081',
        'fork': '6f13c7555a7d50289f0e3587ab780627c65a8000',
        'upstream': '09b603886925994d0d4461857b01321ef3aa431c'}
LANGUAGES = '[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)'

# Each translation has the same scope and limits. The delivery record, rather
# than this template, owns claims about builds, installation and publication.
COPY = {
    'README.md': (
        'Local AI for Windows, with chat, agent work, research and a choice of compatible models. Hakua is the zapabob fork of Niko1221/Strata; it is independently maintained and is not an upstream-endorsed distribution.',
        'Version and downloads',
        'This source is the 0.6.0 integration candidate. App, Terminal, Control and native source versions are coordinated at 0.6.0. Candidate build, installation and publication checks are still in progress; a version number is not proof that an asset is available. The retained verified release is [0.5.9](https://github.com/zapabob/Strata/releases/tag/v0.5.9). Use the delivery record to check the status of 0.6.0.',
        'Install or update',
        'Use the Windows x64 NSIS installer for this fork. It installs per user and carries private Node, Python and Control; system development tools are unnecessary. The signed native CUDA engine is a separate component. The tar.7z bundle must contain the identical installer, component packages, instructions and checksums. It excludes model weights and personal data. Extract 7z first, then tar, and check BUNDLE-SHA256SUMS.txt.',
        'Finish requests, unload models, back up app data, then quit the app including its tray icon. Run the installer in the existing installation folder. Preserve model folders, chats, memory, tools, keys and approval choices. Import the compressed native component with I already have the file; choose a compatible model or a local GGUF, then Load. A model name alone does not establish engine compatibility. See [installation and rollback](docs/HAKUA_INSTALL.md).',
        'Features kept in this fork',
        '`/btw` starts a real tool-free side-question subagent with its own stop control. `/research` and `/deepresearch` collect sources; configured web search can use bundled RustXNG Full. Search queries go to the enabled external providers, so search is not entirely offline. Source counts do not establish answer accuracy.',
        'Workflow skill learning and automatic application are separate options; new profiles keep both off. Existing choices, editable private skills, revisions and undo are preserved. Workspace and sidebar file operations, scoped chat history and memory, compatible PrismML/TurboQuant paths, bounded expert memory, pack integrity and diagnostic measurements remain part of the fork. See [skills](docs/WORKFLOW_SKILLS.md) and [search](docs/BROWSER_SEARCH.md).',
        'Official integration and compatibility',
        'The pinned official source is [09b6038869](https://github.com/Niko1221/Strata/commit/09b603886925994d0d4461857b01321ef3aa431c), associated with the official 0.2.14 / engine 0.1.60 generation. It brings networking, drafting selection, prompt kernels, memory handling and platform fixes. Fork versions are independent. DeepSeek compatibility is retained as an extension where official paths were removed; a missing dedicated runtime must prevent loading. This Windows package does not imply native image input, a distributed native DeepSeek engine, or qualified Linux/Mac inference.',
        'Protection and measured limits',
        'Retain the 3 GiB available-RAM floor, 4 GiB commit headroom, at least 768 MiB GPU reserve and disk floors. The Windows RAM guard also observes loading and generation. Capacity is checked for the actual model and selected backend; 32 GB RAM and 16 GB VRAM do not guarantee every model or context fits. Full-window inference, general model quality and simultaneous GPU use remain unqualified: releaseQualified=false. The installer has no Windows Authenticode signature; component/feed signatures are a separate mechanism. Historical measurements keep their original versions in [release evidence](docs/HAKUA_RELEASE.md).',
        'Documentation and development',
    ),
    'README.ja.md': (
        'WindowsでローカルAIのチャット、エージェント作業、出典付き調査を使うためのアプリです。HakuaはNiko1221/Strataを基にしたzapabobのフォークで、独立して保守しています。公式からの推奨・認定を示す配布物ではありません。',
        'バージョンと配布物',
        'このソースは0.6.0の統合候補です。アプリ、Terminal、Control、ネイティブエンジンのソースを0.6.0にそろえています。ビルド、上書き導入、公開の検証は進行中です。バージョン表記だけでは公開済みとは判断できません。検証済みの旧版[0.5.9](https://github.com/zapabob/Strata/releases/tag/v0.5.9)を保持しています。0.6.0の実施状況は導入・配布記録で確認してください。',
        'インストールと更新',
        'このフォーク用のWindows x64 NSISインストーラーを使います。ユーザー単位で導入し、専用のNode、Python、Controlを同梱します。開発環境は不要です。署名付きCUDAエンジンは別コンポーネントです。tar.7zには同一のインストーラー、コンポーネント、説明書、チェックサムを収めます。モデル重みと個人データは含めません。7z、tarの順に展開し、内部のBUNDLE-SHA256SUMS.txtを確認してください。',
        '実行中の依頼を終え、モデルをアンロードし、アプリデータをバックアップしてから、トレイを含めて終了します。既存の導入先へ上書きしてください。モデル、会話、メモリー、ツール、キー、承認設定を保持します。圧縮したエンジンはアプリの手元のファイルを使う操作から取り込み、対応モデルまたはローカルGGUFを選んで読み込みます。モデル名だけではエンジン互換性を判断できません。[導入と戻し方](docs/HAKUA_INSTALL.md)に詳しい手順があります。',
        '維持する独自機能',
        '`/btw`は停止操作を独立させた、ツールを使わない実際のサブエージェントです。`/research`と`/deepresearch`は出典を集めて調査します。Web検索には同梱のRustXNG Fullを選べます。検索語は有効な外部検索先へ送信するため、検索まで完全にオフラインになるわけではありません。検索件数や閲覧数は回答の正確さを保証しません。',
        'ワークフロースキルの学習と自動適用は別々に切り替えます。新規プロファイルでは両方とも無効です。既存設定、編集可能な非公開スキル、履歴と取り消しを維持します。ワークスペースとサイドバーのファイル操作、範囲を限定した会話履歴とメモリー、対応するPrismML・TurboQuant経路、上限付きエキスパートメモリー、pack整合性検査、観測機能も維持します。[スキル](docs/WORKFLOW_SKILLS.md)と[検索](docs/BROWSER_SEARCH.md)をご覧ください。',
        '公式の取り込みと互換性',
        '公式ソースの基準は[09b6038869](https://github.com/Niko1221/Strata/commit/09b603886925994d0d4461857b01321ef3aa431c)です。公式0.2.14、エンジン0.1.60世代の通信、ドラフト選択、プロンプト処理、メモリー管理、各OSの修正を取り込みます。フォークのバージョン体系は独立しています。公式が削除したDeepSeek経路は独自拡張として互換性を維持し、専用ランタイムがなければ読み込みを止めます。このWindows配布は、ネイティブ画像入力、専用DeepSeekエンジンの配布、Linux・Macでの推論検証を意味しません。',
        '保護条件と測定上の制限',
        '空きRAM 3 GiB、コミット余力4 GiB、GPU予約領域768 MiB以上、ディスク空き容量の下限を維持します。WindowsのRAM保護は読み込みと生成中も監視します。必要量は実際のモデルと選択したバックエンドで判断します。RAM 32 GB・VRAM 16 GBでも、すべてのモデルやコンテキストが使えるわけではありません。最大長の推論、一般的なモデル品質、GPU同時実行は未認定で、releaseQualified=falseです。インストーラーにWindows Authenticode署名はありません。コンポーネント・フィードの署名とは区別してください。過去の測定は[リリース記録](docs/HAKUA_RELEASE.md)に元のバージョンのまま残します。',
        '資料と開発',
    ),
    'README.de.md': (
        'Lokale KI unter Windows für Chats, Agentenaufgaben und Recherche mit Quellen. Hakua ist der unabhängig gepflegte zapabob-Fork von Niko1221/Strata und keine vom Originalprojekt bestätigte Distribution.',
        'Version und Pakete',
        'Dieser Quellstand ist der Integrationskandidat 0.6.0. App, Terminal, Control und nativer Quellcode tragen 0.6.0. Build, Installation und Veröffentlichung werden noch geprüft; eine Versionsnummer belegt keine verfügbare Datei. Die verifizierte frühere [Version 0.5.9](https://github.com/zapabob/Strata/releases/tag/v0.5.9) bleibt erhalten. Der Auslieferungsnachweis nennt den tatsächlichen Stand von 0.6.0.',
        'Installation und Update',
        'Verwenden Sie das Windows-x64-NSIS-Paket dieses Forks. Es installiert pro Benutzer und enthält eigenes Node, Python und Control; Entwicklungswerkzeuge sind nicht nötig. Die signierte native CUDA-Engine ist eine separate Komponente. Das tar.7z-Paket enthält denselben Installer, Komponenten, Anleitung und Prüfsummen, jedoch keine Modellgewichte oder persönlichen Daten. Zuerst 7z, dann tar entpacken und BUNDLE-SHA256SUMS.txt prüfen.',
        'Aufträge abschließen, Modelle entladen, App-Daten sichern und Strata einschließlich Tray beenden. In den bisherigen Ordner installieren. Modelle, Chats, Speicher, Werkzeuge, Schlüssel und Freigaben erhalten. Die komprimierte Engine über die Option für eine vorhandene Datei importieren, ein kompatibles Modell oder lokales GGUF wählen und laden. Ein Modellname beweist keine Engine-Kompatibilität. [Installation und Rückkehr zur Vorversion](docs/HAKUA_INSTALL.md).',
        'Erhaltene Funktionen',
        '`/btw` startet einen echten werkzeuglosen Nebenfragen-Agenten mit eigener Stoppfunktion. `/research` und `/deepresearch` sammeln Quellen. Die Websuche kann das mitgelieferte RustXNG Full verwenden. Suchanfragen gehen an aktivierte externe Anbieter; die Suche ist daher nicht vollständig offline. Quellenzahlen garantieren keine korrekte Antwort.',
        'Das Lernen von Workflow-Skills und deren automatische Anwendung sind getrennte Optionen, bei neuen Profilen beide aus. Bestehende Einstellungen, private editierbare Skills, Revisionen und Rücknahme bleiben erhalten. Ebenso bleiben Dateioperationen, begrenzter Chat- und Speicherzugriff, kompatible PrismML-/TurboQuant-Pfade, begrenzter Expertenspeicher, Pack-Integrität und Diagnosefunktionen erhalten. [Skills](docs/WORKFLOW_SKILLS.md) · [Suche](docs/BROWSER_SEARCH.md).',
        'Originalprojekt und Kompatibilität',
        'Grundlage ist der offizielle Commit [09b6038869](https://github.com/Niko1221/Strata/commit/09b603886925994d0d4461857b01321ef3aa431c) aus der Generation 0.2.14 / Engine 0.1.60: Netzwerk, Drafting-Auswahl, Prompt-Kernel, Speicherverwaltung und Plattformkorrekturen. Fork-Versionen sind unabhängig. Entfernte DeepSeek-Pfade bleiben als Erweiterung erhalten; ohne passende dedizierte Laufzeit muss das Laden scheitern. Dieses Windows-Paket belegt weder native Bildeingabe noch eine ausgelieferte native DeepSeek-Engine oder geprüfte Linux-/Mac-Inferenz.',
        'Schutz und Messgrenzen',
        'Die Grenzen bleiben bei 3 GiB freiem RAM, 4 GiB Commit-Spielraum, mindestens 768 MiB GPU-Reserve und Mindestplatz auf den Laufwerken. Der Windows-RAM-Schutz überwacht auch Laden und Generierung. 32 GB RAM und 16 GB VRAM garantieren nicht jedes Modell oder Kontextfenster. Volle Kontextlänge, allgemeine Modellqualität und parallele GPU-Nutzung sind nicht qualifiziert: releaseQualified=false. Der Installer hat keine Windows-Authenticode-Signatur; Komponenten- und Feed-Signaturen sind davon getrennt. Frühere Messungen stehen mit ihren Originalversionen im [Auslieferungsnachweis](docs/HAKUA_RELEASE.md).',
        'Dokumentation und Entwicklung',
    ),
    'README.es.md': (
        'IA local en Windows para conversar, trabajar con agentes e investigar con fuentes. Hakua es la bifurcación de Niko1221/Strata mantenida de forma independiente por zapabob; no es una distribución avalada por el proyecto original.',
        'Versión y descargas',
        'Este código es el candidato de integración 0.6.0. La aplicación, Terminal, Control y el motor nativo comparten la versión de código 0.6.0. Siguen pendientes las comprobaciones de compilación, instalación y publicación; el número de versión no demuestra que haya archivos disponibles. Se conserva la [versión 0.5.9](https://github.com/zapabob/Strata/releases/tag/v0.5.9) verificada. Consulte el registro de entrega para conocer el estado real de 0.6.0.',
        'Instalar y actualizar',
        'Utilice el instalador NSIS para Windows x64 de esta bifurcación. La instalación es por usuario e incluye Node, Python y Control privados, sin herramientas de desarrollo del sistema. El motor CUDA firmado es un componente aparte. El archivo tar.7z debe incluir el mismo instalador, componentes, instrucciones y sumas de comprobación, sin pesos de modelos ni datos personales. Extraiga primero 7z y después tar; compruebe BUNDLE-SHA256SUMS.txt.',
        'Termine las solicitudes, descargue los modelos de memoria, guarde una copia de los datos y cierre la aplicación, incluida la bandeja. Instale sobre la carpeta existente. Conserve modelos, conversaciones, memoria, herramientas, claves y autorizaciones. Importe el motor comprimido con la opción de archivo existente y cargue un modelo compatible o GGUF local. El nombre no demuestra compatibilidad con el motor. [Instalación y reversión](docs/HAKUA_INSTALL.md).',
        'Funciones conservadas',
        '`/btw` inicia un subagente real sin herramientas para preguntas paralelas, con su propio control de parada. `/research` y `/deepresearch` reúnen fuentes. La búsqueda web puede usar RustXNG Full incluido. Las consultas llegan a los proveedores externos habilitados; la búsqueda no es completamente local. El número de fuentes no garantiza exactitud.',
        'El aprendizaje de habilidades y su aplicación automática se activan por separado; ambos están desactivados en perfiles nuevos. Se conservan preferencias, habilidades privadas editables, revisiones y deshacer. También se mantienen las operaciones de archivos, el historial y memoria delimitados, las rutas compatibles PrismML/TurboQuant, la memoria de expertos limitada, la integridad de packs y los diagnósticos. [Habilidades](docs/WORKFLOW_SKILLS.md) · [Búsqueda](docs/BROWSER_SEARCH.md).',
        'Integración oficial y compatibilidad',
        'La referencia oficial es [09b6038869](https://github.com/Niko1221/Strata/commit/09b603886925994d0d4461857b01321ef3aa431c), de la generación 0.2.14 / motor 0.1.60: red, selección de borradores, procesamiento de prompts, memoria y correcciones de plataforma. Las versiones de la bifurcación son independientes. Las rutas DeepSeek retiradas se conservan como extensión; la ausencia de un ejecutable dedicado debe impedir la carga. Este paquete Windows no acredita entrada nativa de imágenes, distribución de un motor nativo DeepSeek ni inferencia validada en Linux/Mac.',
        'Protección y límites medidos',
        'Se mantienen 3 GiB de RAM disponible, 4 GiB de margen de memoria comprometida, al menos 768 MiB reservados en GPU y los mínimos de disco. La protección de RAM de Windows observa carga y generación. 32 GB de RAM y 16 GB de VRAM no garantizan cualquier modelo o contexto. La inferencia de ventana completa, la calidad general y el uso simultáneo de GPU siguen sin calificación: releaseQualified=false. El instalador no tiene firma Windows Authenticode; las firmas de componentes y del feed son independientes. Las mediciones históricas mantienen sus versiones en el [registro](docs/HAKUA_RELEASE.md).',
        'Documentación y desarrollo',
    ),
    'README.fr.md': (
        'IA locale sous Windows pour discuter, utiliser des agents et rechercher avec des sources. Hakua est le fork de Niko1221/Strata maintenu indépendamment par zapabob, sans approbation annoncée du projet d’origine.',
        'Version et téléchargements',
        'Ce code est le candidat d’intégration 0.6.0. Application, Terminal, Control et sources du moteur natif portent 0.6.0. Les vérifications de compilation, d’installation et de publication sont en cours ; le numéro ne prouve pas la disponibilité des fichiers. La [version 0.5.9](https://github.com/zapabob/Strata/releases/tag/v0.5.9) vérifiée est conservée. Le compte rendu de livraison précise le statut réel de 0.6.0.',
        'Installation et mise à jour',
        'Utilisez l’installeur NSIS Windows x64 de ce fork. Il s’installe par utilisateur avec ses propres Node, Python et Control, sans outils de développement système. Le moteur CUDA signé est un composant distinct. Le tar.7z doit contenir le même installeur, les composants, les instructions et les sommes de contrôle, sans poids de modèles ni données personnelles. Extraire 7z puis tar et vérifier BUNDLE-SHA256SUMS.txt.',
        'Terminez les requêtes, déchargez les modèles, sauvegardez les données et quittez aussi l’icône de notification. Réinstallez dans le dossier existant. Conservez modèles, conversations, mémoire, outils, clés et autorisations. Importez le moteur compressé avec l’option de fichier existant, puis chargez un modèle compatible ou un GGUF local. Le nom du modèle ne suffit pas à établir sa compatibilité. [Installation et retour arrière](docs/HAKUA_INSTALL.md).',
        'Fonctions conservées',
        '`/btw` lance un véritable sous-agent sans outils pour une question annexe, avec son propre arrêt. `/research` et `/deepresearch` recueillent des sources. La recherche peut utiliser RustXNG Full fourni. Les requêtes sont transmises aux fournisseurs externes activés ; la recherche n’est donc pas entièrement hors ligne. Le nombre de sources ne garantit pas l’exactitude.',
        'L’apprentissage des compétences de travail et leur application automatique sont deux options distinctes, désactivées dans les nouveaux profils. Préférences, compétences privées modifiables, révisions et annulation sont conservées. Les opérations sur fichiers, l’historique et la mémoire délimités, les chemins PrismML/TurboQuant compatibles, la mémoire d’experts bornée, l’intégrité des packs et les diagnostics restent présents. [Compétences](docs/WORKFLOW_SKILLS.md) · [Recherche](docs/BROWSER_SEARCH.md).',
        'Intégration officielle et compatibilité',
        'La référence officielle est [09b6038869](https://github.com/Niko1221/Strata/commit/09b603886925994d0d4461857b01321ef3aa431c), génération 0.2.14 / moteur 0.1.60 : réseau, choix du brouillonneur, noyaux de prompts, mémoire et correctifs de plateforme. Les versions du fork sont indépendantes. Les chemins DeepSeek supprimés sont conservés comme extension ; sans exécutable dédié, le chargement doit être refusé. Ce paquet Windows ne valide ni entrée native d’images, ni distribution d’un moteur natif DeepSeek, ni inférence Linux/Mac.',
        'Protection et limites des mesures',
        'Les seuils restent à 3 GiB de RAM disponible, 4 GiB de marge d’engagement mémoire, au moins 768 MiB de réserve GPU et des minimums de disque. La protection RAM Windows surveille chargement et génération. 32 Go de RAM et 16 Go de VRAM ne garantissent pas tout modèle ou contexte. Fenêtre complète, qualité générale et concurrence GPU restent non qualifiées : releaseQualified=false. L’installeur n’a pas de signature Windows Authenticode ; signatures de composants et de flux sont distinctes. Les [mesures historiques](docs/HAKUA_RELEASE.md) gardent leurs versions d’origine.',
        'Documentation et développement',
    ),
    'README.pl.md': (
        'Lokalna AI w Windows: rozmowy, zadania agentów i badania ze źródłami. Hakua to niezależnie utrzymywany przez zapabob fork Niko1221/Strata, bez deklarowanego poparcia oryginalnego projektu.',
        'Wersja i pliki',
        'Ten kod jest kandydatem integracji 0.6.0. Aplikacja, Terminal, Control i źródła silnika natywnego mają wersję 0.6.0. Weryfikacja kompilacji, instalacji i publikacji trwa; numer wersji nie dowodzi dostępności plików. Zachowano zweryfikowane [wydanie 0.5.9](https://github.com/zapabob/Strata/releases/tag/v0.5.9). Faktyczny stan 0.6.0 podaje zapis dostarczenia.',
        'Instalacja i aktualizacja',
        'Użyj instalatora NSIS Windows x64 tego forka. Instaluje się dla użytkownika i zawiera własne Node, Python i Control; narzędzia programistyczne nie są potrzebne. Podpisany silnik CUDA to osobny komponent. tar.7z ma zawierać ten sam instalator, komponenty, instrukcje i sumy kontrolne, bez wag modeli i danych osobistych. Rozpakuj najpierw 7z, potem tar, i sprawdź BUNDLE-SHA256SUMS.txt.',
        'Zakończ żądania, zwolnij modele z pamięci, zrób kopię danych i zamknij aplikację również w zasobniku. Zainstaluj w istniejącym katalogu. Zachowaj modele, rozmowy, pamięć, narzędzia, klucze i zgody. Zaimportuj skompresowany silnik opcją posiadanego pliku, wybierz zgodny model lub lokalny GGUF i załaduj go. Nazwa modelu nie dowodzi zgodności z silnikiem. [Instalacja i powrót do poprzedniej wersji](docs/HAKUA_INSTALL.md).',
        'Zachowane funkcje',
        '`/btw` uruchamia rzeczywistego podagenta bez narzędzi do pytania pobocznego, z osobnym zatrzymaniem. `/research` i `/deepresearch` gromadzą źródła. Wyszukiwanie może używać dołączonego RustXNG Full. Zapytania trafiają do włączonych dostawców zewnętrznych; wyszukiwanie nie jest całkowicie offline. Liczba źródeł nie gwarantuje poprawności.',
        'Uczenie umiejętności i automatyczne stosowanie to osobne opcje, obie wyłączone w nowych profilach. Zachowane są ustawienia, prywatne edytowalne umiejętności, wersje i cofanie zmian. Pozostają operacje plikowe, ograniczony zakres historii i pamięci, zgodne ścieżki PrismML/TurboQuant, ograniczona pamięć ekspertów, integralność pakietów oraz diagnostyka. [Umiejętności](docs/WORKFLOW_SKILLS.md) · [Wyszukiwanie](docs/BROWSER_SEARCH.md).',
        'Integracja oficjalnego projektu',
        'Punktem odniesienia jest oficjalny commit [09b6038869](https://github.com/Niko1221/Strata/commit/09b603886925994d0d4461857b01321ef3aa431c), generacja 0.2.14 / silnik 0.1.60: sieć, wybór modelu pomocniczego, jądra promptów, pamięć i poprawki platform. Numeracja forka jest niezależna. Usunięte ścieżki DeepSeek pozostają rozszerzeniem; brak dedykowanego środowiska ma blokować ładowanie. Pakiet Windows nie potwierdza natywnego wejścia obrazu, dystrybucji silnika DeepSeek ani sprawdzonej inferencji Linux/Mac.',
        'Ochrona i ograniczenia pomiarów',
        'Zachowane minima to 3 GiB dostępnego RAM, 4 GiB zapasu pamięci zadeklarowanej, co najmniej 768 MiB rezerwy GPU i wolne miejsce na dyskach. Ochrona RAM w Windows obserwuje też ładowanie i generowanie. 32 GB RAM i 16 GB VRAM nie gwarantują dowolnego modelu lub kontekstu. Pełne okno, ogólna jakość modeli i współbieżność GPU pozostają niezakwalifikowane: releaseQualified=false. Instalator nie ma podpisu Windows Authenticode; podpisy komponentów i kanału aktualizacji są odrębne. [Dawne pomiary](docs/HAKUA_RELEASE.md) zachowują pierwotne wersje.',
        'Dokumentacja i rozwój',
    ),
    'README.pt-BR.md': (
        'IA local no Windows para conversar, trabalhar com agentes e pesquisar com fontes. Hakua é o fork de Niko1221/Strata mantido de forma independente por zapabob; não é uma distribuição endossada pelo projeto original.',
        'Versão e downloads',
        'Este código é o candidato de integração 0.6.0. Aplicativo, Terminal, Control e fontes do motor nativo estão em 0.6.0. As verificações de compilação, instalação e publicação continuam; o número de versão não comprova a disponibilidade dos arquivos. A [versão 0.5.9](https://github.com/zapabob/Strata/releases/tag/v0.5.9) verificada foi preservada. Consulte o registro de entrega para o estado real de 0.6.0.',
        'Instalar e atualizar',
        'Use o instalador NSIS Windows x64 deste fork. A instalação por usuário inclui Node, Python e Control próprios, sem exigir ferramentas de desenvolvimento no sistema. O motor CUDA assinado é um componente separado. O tar.7z deve conter o mesmo instalador, componentes, instruções e somas de verificação, sem pesos de modelos ou dados pessoais. Extraia primeiro 7z e depois tar; confira BUNDLE-SHA256SUMS.txt.',
        'Termine as solicitações, descarregue os modelos da memória, faça backup dos dados e feche também o ícone da bandeja. Instale na pasta existente. Preserve modelos, conversas, memória, ferramentas, chaves e aprovações. Importe o motor compactado pela opção de arquivo existente, escolha um modelo compatível ou GGUF local e carregue. O nome do modelo não comprova compatibilidade. [Instalação e reversão](docs/HAKUA_INSTALL.md).',
        'Recursos preservados',
        '`/btw` inicia um subagente real sem ferramentas para perguntas paralelas, com parada própria. `/research` e `/deepresearch` reúnem fontes. A busca pode usar RustXNG Full incluído. As consultas vão aos provedores externos habilitados; a busca não é totalmente offline. A quantidade de fontes não garante precisão.',
        'Aprendizado de habilidades e aplicação automática são opções separadas, ambas desativadas em novos perfis. Preferências, habilidades privadas editáveis, revisões e desfazer são preservados. Também permanecem operações de arquivos, histórico e memória com escopo, caminhos PrismML/TurboQuant compatíveis, memória limitada de especialistas, integridade de packs e diagnósticos. [Habilidades](docs/WORKFLOW_SKILLS.md) · [Busca](docs/BROWSER_SEARCH.md).',
        'Integração oficial e compatibilidade',
        'A referência oficial é [09b6038869](https://github.com/Niko1221/Strata/commit/09b603886925994d0d4461857b01321ef3aa431c), geração 0.2.14 / motor 0.1.60: rede, escolha de rascunhador, kernels de prompts, memória e correções de plataforma. As versões do fork são independentes. Caminhos DeepSeek removidos são mantidos como extensão; sem o executável dedicado, a carga deve ser bloqueada. Este pacote Windows não valida entrada nativa de imagens, distribuição de motor nativo DeepSeek ou inferência Linux/Mac.',
        'Proteção e limites medidos',
        'Permanecem os mínimos de 3 GiB de RAM disponível, 4 GiB de margem de memória comprometida, pelo menos 768 MiB de reserva GPU e espaço livre em disco. A proteção RAM do Windows observa carga e geração. 32 GB RAM e 16 GB VRAM não garantem qualquer modelo ou contexto. Janela completa, qualidade geral e concorrência GPU continuam sem qualificação: releaseQualified=false. O instalador não tem assinatura Windows Authenticode; assinaturas de componentes e do feed são mecanismos separados. As [medições históricas](docs/HAKUA_RELEASE.md) mantêm suas versões originais.',
        'Documentação e desenvolvimento',
    ),
    'README.zh-CN.md': (
        '在 Windows 上运行本地 AI，支持聊天、智能体任务与带来源的研究。Hakua 是 zapabob 独立维护的 Niko1221/Strata 分支，不代表原项目的推荐或认证。',
        '版本与下载',
        '当前源码是 0.6.0 集成候选版。应用、Terminal、Control 与原生引擎源码均为 0.6.0。构建、覆盖安装与发布验证仍在进行，版本号并不证明文件已经公开。已验证的旧版 [0.5.9](https://github.com/zapabob/Strata/releases/tag/v0.5.9) 保留。请通过交付记录确认 0.6.0 的实际状态。',
        '安装与更新',
        '请使用本分支的 Windows x64 NSIS 安装程序。按用户安装，内含专用 Node、Python 和 Control，无需系统开发工具。签名的原生 CUDA 引擎是独立组件。tar.7z 应包含完全相同的安装程序、组件、说明与校验文件，不包含模型权重或个人数据。先解压 7z，再解开 tar，核对 BUNDLE-SHA256SUMS.txt。',
        '先完成请求、卸载已加载模型、备份应用数据，再退出应用及托盘图标。安装到原目录，保留模型、会话、记忆、工具、密钥与审批设置。通过使用已有文件的选项导入压缩引擎，选择兼容模型或本地 GGUF 后加载。模型名称本身不能证明引擎兼容性。[安装与回退](docs/HAKUA_INSTALL.md)。',
        '保留的分支功能',
        '`/btw` 会启动真实的无工具旁支子智能体，并提供独立停止操作。`/research` 与 `/deepresearch` 收集来源。网页搜索可使用内置 RustXNG Full。查询会发送给已启用的外部提供方，因此搜索并非完全离线。来源数量不能保证回答正确。',
        '工作流技能学习与自动应用是独立开关，新建配置默认都关闭。保留已有选择、可编辑的私有技能、修订历史与撤销。也保留文件操作、限定范围的会话历史与记忆、兼容的 PrismML/TurboQuant 路径、有界专家内存、pack 完整性检查及诊断功能。[技能](docs/WORKFLOW_SKILLS.md) · [搜索](docs/BROWSER_SEARCH.md)。',
        '官方集成与兼容性',
        '固定的官方基准为 [09b6038869](https://github.com/Niko1221/Strata/commit/09b603886925994d0d4461857b01321ef3aa431c)，属于官方 0.2.14 / 引擎 0.1.60 代际，包含网络、草稿模型选择、提示词内核、内存与平台修复。分支版本独立编号。官方删除的 DeepSeek 路径作为扩展保留；缺少实际专用运行时必须阻止加载。本 Windows 包不证明原生图像输入、原生 DeepSeek 引擎分发或 Linux/Mac 推理已通过验证。',
        '保护条件与测量限制',
        '保留可用 RAM 3 GiB、提交余量 4 GiB、GPU 预留至少 768 MiB 及磁盘空闲下限。Windows RAM 保护同时监测加载和生成。32 GB RAM 与 16 GB VRAM 并不保证所有模型或上下文都能运行。完整上下文推理、一般模型质量及 GPU 并发仍未获得资格确认：releaseQualified=false。安装程序没有 Windows Authenticode 签名，组件与更新源签名是独立机制。[历史测量](docs/HAKUA_RELEASE.md) 保留原版本信息。',
        '文档与开发',
    ),
}


def main():
    receipt = BASE / 'readme-rewrite.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    assert set(COPY) == {p.name for p in ROOT.glob('README*.md')}
    records = []
    for relative, text in COPY.items():
        intro, version_h, version, install_h, install, update, features_h, features1, features2, integration_h, integration, limits_h, limits, docs_h = text
        inputs = {}
        for label, ref in REFS.items():
            blob = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{ref}:{relative}'])
            inputs[label] = {'ref': ref, 'sha256': hashlib.sha256(blob).hexdigest()}
        content = f'''# Hakua / Strata

{LANGUAGES}

{intro}

## {version_h}

<!-- release-060-candidate -->
{version}

[0.6.0](https://github.com/zapabob/Strata/releases/tag/v0.6.0) · [Strata-0.6.0-windows-x64-setup.exe](https://github.com/zapabob/Strata/releases/download/v0.6.0/Strata-0.6.0-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.6.0/Strata-0.6.0-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.0/SHA256SUMS.txt)

## {install_h}

{install}

{update}

## {features_h}

{features1}

{features2}

## {integration_h}

{integration}

## {limits_h}

{limits}

## {docs_h}

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

[Niko1221/Strata](https://github.com/Niko1221/Strata) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) · [Third-party sources and notices](third_party/)
'''
        path = ROOT / relative
        path.write_text(content, encoding='utf-8', newline='\n')
        records.append({'path': relative, 'inputs': inputs,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump({'scope': 'all eight root README translations', 'records': records,
                   'publicationClaim': 'candidate only; verify and update after delivery',
                   'historicalEvidence': 'retained in Git and linked immutable release records'}, stream, indent=2)


if __name__ == '__main__':
    main()
