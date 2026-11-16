# Hakua / Strata

[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)

WindowsでローカルAIのチャット、エージェント作業、出典付き調査を使うためのアプリです。HakuaはNiko1221/Strataを基にしたzapabobのフォークで、独立して保守しています。公式からの推奨・認定を示す配布物ではありません。

## バージョンと配布物

<!-- release-060-candidate -->
このソースは0.6.0の統合候補です。アプリ、Terminal、Control、ネイティブエンジンのソースを0.6.0にそろえています。ビルド、上書き導入、公開の検証は進行中です。バージョン表記だけでは公開済みとは判断できません。検証済みの旧版[0.5.9](https://github.com/zapabob/Strata/releases/tag/v0.5.9)を保持しています。0.6.0の実施状況は導入・配布記録で確認してください。

[0.6.0](https://github.com/zapabob/Strata/releases/tag/v0.6.0) · [Strata-0.6.0-windows-x64-setup.exe](https://github.com/zapabob/Strata/releases/download/v0.6.0/Strata-0.6.0-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.6.0/Strata-0.6.0-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.0/SHA256SUMS.txt)

## インストールと更新

このフォーク用のWindows x64 NSISインストーラーを使います。ユーザー単位で導入し、専用のNode、Python、Controlを同梱します。開発環境は不要です。署名付きCUDAエンジンは別コンポーネントです。tar.7zには同一のインストーラー、コンポーネント、説明書、チェックサムを収めます。モデル重みと個人データは含めません。7z、tarの順に展開し、内部のBUNDLE-SHA256SUMS.txtを確認してください。

実行中の依頼を終え、モデルをアンロードし、アプリデータをバックアップしてから、トレイを含めて終了します。既存の導入先へ上書きしてください。モデル、会話、メモリー、ツール、キー、承認設定を保持します。圧縮したエンジンはアプリの手元のファイルを使う操作から取り込み、対応モデルまたはローカルGGUFを選んで読み込みます。モデル名だけではエンジン互換性を判断できません。[導入と戻し方](docs/HAKUA_INSTALL.md)に詳しい手順があります。

## 維持する独自機能

`/btw`は停止操作を独立させた、ツールを使わない実際のサブエージェントです。`/research`と`/deepresearch`は出典を集めて調査します。Web検索には同梱のRustXNG Fullを選べます。検索語は有効な外部検索先へ送信するため、検索まで完全にオフラインになるわけではありません。検索件数や閲覧数は回答の正確さを保証しません。

ワークフロースキルの学習と自動適用は別々に切り替えます。新規プロファイルでは両方とも無効です。既存設定、編集可能な非公開スキル、履歴と取り消しを維持します。ワークスペースとサイドバーのファイル操作、範囲を限定した会話履歴とメモリー、対応するPrismML・TurboQuant経路、上限付きエキスパートメモリー、pack整合性検査、観測機能も維持します。[スキル](docs/WORKFLOW_SKILLS.md)と[検索](docs/BROWSER_SEARCH.md)をご覧ください。

## 公式の取り込みと互換性

公式ソースの基準は[baf16363b4](https://github.com/Niko1221/Strata/commit/baf16363b4403ef0ed4f538f5d5ed11c43fb9535)です。公式0.2.14、エンジン0.1.60世代の通信、ドラフト選択、プロンプト処理、メモリー管理、各OSの修正を取り込みます。フォークのバージョン体系は独立しています。公式が削除したDeepSeek経路は独自拡張として互換性を維持し、専用ランタイムがなければ読み込みを止めます。このWindows配布は、ネイティブ画像入力、専用DeepSeekエンジンの配布、Linux・Macでの推論検証を意味しません。

## 保護条件と測定上の制限

空きRAM 3 GiB、コミット余力4 GiB、GPU予約領域768 MiB以上、ディスク空き容量の下限を維持します。WindowsのRAM保護は読み込みと生成中も監視します。必要量は実際のモデルと選択したバックエンドで判断します。RAM 32 GB・VRAM 16 GBでも、すべてのモデルやコンテキストが使えるわけではありません。最大長の推論、一般的なモデル品質、GPU同時実行は未認定で、releaseQualified=falseです。インストーラーにWindows Authenticode署名はありません。コンポーネント・フィードの署名とは区別してください。過去の測定は[リリース記録](docs/HAKUA_RELEASE.md)に元のバージョンのまま残します。

## 資料と開発

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

[Niko1221/Strata](https://github.com/Niko1221/Strata) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) · [第三者のソースと表記](third_party/)
