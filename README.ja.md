# Hakua / Strata

[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)

Windows でローカル AI のチャット、エージェント作業、出典付き調査を使うためのアプリです。Hakua は Niko1221/Strata を基にした zapabob のフォークで、独立して保守しています。公式からの推奨・認定を示す配布物ではありません。

## 0.6.8 の変更点

このリリースは公式 Strata 0.2.16 の意味を fork にマージし、fork 独自機能はすべて保持しました（公式の削除は適用していません）。Qwen3.8 ネイティブ CUDA エンジンは公式エンジン API に接続され、`strata.exe` の `--families` に `qwen35` が表示され、実 GGUF を標準の常驻 `--serve` プロトコルで配信します。CUDA 13.5 / SM120 でビルド・E2E 検証済み。デコードは Prism llama.cpp と同等です（両者とも同じ DP4A カーネル方式）。反転順ウォーム A/B で F16 KV 書き込みのバッチ化により prefill の KV コストを 27% 削減し、バイト一致 logits を維持しました。Sync-Strata 由来の 32GB RAM 起動前スイーズとコミット制限の案内も含まれます。フルウィンドウ生成と一般モデル品質は未検証です（`releaseQualified=false`）。

## バージョンと配布物

<!-- release-062-stable -->
現在のアプリリリース: [0.6.8](https://github.com/zapabob/Strata/releases/tag/v0.6.8)（アプリ `0.6.8`、Terminal・Control・ネイティブエンジン `0.6.3`）。この Windows アプリ専用リリースは意図的に GitHub Latest ではなく、Latest は `0.6.3` のままです。GPU エンジンは別途署名された [`v0.6.4-preview.1`](https://github.com/zapabob/Strata/releases/tag/v0.6.4-preview.1) コンパニオンです。最大長の推論や一般的なモデル品質は未認定のため、`releaseQualified=false` を明示しています。

[0.6.8](https://github.com/zapabob/Strata/releases/tag/v0.6.8) · [setup EXE](https://github.com/zapabob/Strata/releases/download/v0.6.8/Strata-0.6.8-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.6.8/Strata-0.6.8-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.8/SHA256SUMS.txt) · [BUNDLE-SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.8/BUNDLE-SHA256SUMS.txt)

![Hakua / Strata real-model Japanese chat on Qwen3.8-27B](docs/media/native-qwen38-japanese-chat.png)
*`strata-q38.exe` で `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf` を実行している実環境スクリーンショット（CUDA 13 / RTX 5060 Ti 16 GB）。*

## Qwen3.8 試作版の推奨 GGUF

RTX 5060 Ti（VRAM 16 GB）でローカルのテキスト会話を試すなら、実機検証に使った [soyaakinohara さんの混合量子化 GGUF（3.69 bpw）](https://huggingface.co/soyaakinohara/qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf) をおすすめします。量子化と公開を行った **soyaakinohara さん**、元の重みを公開した **AEON-7** に感謝します。モデルは別途ダウンロードし、まず4,096トークンのコンテキストでお試しください。ファイル名の「12GB」は VRAM 12 GB での動作保証ではありません。

[署名済みネイティブ CUDA 13.5 プレビュー](https://github.com/zapabob/Strata/releases/tag/v0.6.4-preview.1) は、インストール済み 0.6.3 アプリで日本語の2ターン生成・応答途中の停止・アンロードに合格しました。初回リクエスト2組で短文 F16 プリフィルを従来ネイティブ経路より35〜43%短縮しました。同条件のデコードは Prism（llama.cpp）が速い結果でした。[導入・公開鍵・制限](docs/Q38_NATIVE_PREFILL_PREVIEW.md)

## インストールと更新

このフォーク用の Windows x64 NSIS インストーラーを使います。ユーザー単位で導入され、専用の Node・Python・Control を同梱しているため、システム側の開発環境は不要です。ネイティブ CUDA エンジンは別コンポーネントです。`tar.7z` にはインストーラー、CUDA エンジンパッケージ、Control パッケージ、説明書、チェックサムを収めており、モデル重みと個人データは含みません。

実行中の依頼を終え、モデルをアンロードし、アプリデータをバックアップしてから、トレイを含めてアプリを終了してください。既存の導入先へそのまま上書きインストールすると、モデル・会話・メモリー・ツール・API キー・承認設定を保持したまま更新できます。圧縮した CUDA コンポーネントはアプリの「手元のファイルを使う」操作から取り込み、対応モデルまたはローカル GGUF を選んで読み込んでください。モデル名だけではエンジン互換性を判断できません。詳しくは [導入と戻し方](docs/HAKUA_INSTALL.md) を参照してください。

## 0.6.2 の主な特徴と独自機能

0.6.2 では Windows CUDA 13 / Blackwell（SM120）向けの [ローカル Qwen3.8-27B 系 Dense GGUF ランナー（`strata-q38.exe`）](docs/QWEN38_DENSE_NATIVE.md) を大幅に強化しました：
- **テンソル単位の混合量子化ポリモーフィズム（`StrataQuantTensor`）**: `F32`, `F16`, `BF16`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ3_S`, `IQ2_S`, `IQ4_XS` に加え、PrismML `PQ2_0`（type 142）を含む全11種をテンソル単位で動的ディスパッチ。
- **Tensor Core バッチ Prefill ＋ マルチトークン GDN ＋ 融合 Spin8 Attention ＋ CUDA Graph Decode（CUDA 13.5 SM120）**:
  - 140トークン Prefill（`--kv spin8_k8v4 --ctx-size 4096`）: `36,760.7 ms`（`3.81 tok/s`, `--conservative-sync`）→ **`1,558.3 ms`（ウォーム `89.8 tok/s`、23.6倍高速化）**。
  - 定常 Decode（`GEN 32`、31 steps、パック単位 GEMV ＋ 8層フェンス CUDA Graph）: `206.4 ms/step` → **`85.13 ms/step`（`11.75 step/s`、2.42倍高速化）**。
  - **長文脈実測（`--kv spin8_k8v4`, RTX 5060 Ti 16 GB）**: **65K (`65,536`)** 通過（1,024tok Prefill `10,813.5〜11,220.7 ms`, `91.3〜94.7 tok/s`、Decode `670.4 ms` / 7 steps）、**96K (`98,304`)** 通過（`required_with_reserve=15.09 GB`, 140tok Prefill `1,968.7 ms`）、**131K (`131,072`)** は 768 MiB GPU 予約保護によりガード拒否（`required=15.97 GB > available=15.91 GB`）。
- **実 MTP `K=3` ドラフト検証ループ（`blk.64.nextn.*`）**: `--spec-draft-n-max 0..3`、`KvCache::rewind`、GDN 状態スナップショット／ロールバックに対応。
- **KV キャッシュと日本語プロファイル**: `F16` / `Spin8 K8V4` / `F16-K・Turbo2/3/4-V`（`auto` 順: `f16` → `f16_turbo4` → `spin8_k8v4`）と、任意の日本語出力プロファイル（`model-default` で制約オフ）を選択可能。

`/btw` は停止操作を独立させた、ツールを使わない実際のサブエージェントです。`/research` と `/deepresearch` は出典を集めて調査し、Web 検索には同梱の RustXNG Full を利用できます（外部検索先へクエリを送信するため、検索は完全オフラインではありません）。出典数は回答の正確さを保証しません。

ワークフロースキルの学習と自動適用は別々に切り替えられ、新規プロファイルでは両方とも無効です。既存設定、編集可能な非公開スキル、履歴と取り消しを維持します。ワークスペース／サイドバーのファイル操作、スコープ限定チャット履歴とメモリー、互換 PrismML・TurboQuant 経路、上限付きエキスパートメモリー、pack 整合性検査も維持しています。[スキル](docs/WORKFLOW_SKILLS.md) と [検索](docs/BROWSER_SEARCH.md) をご覧ください。

## 公式の取り込みと互換性

公式ソースの基準は [c775728069](https://github.com/Niko1221/Strata/commit/c77572806996349b49995e4f98d41b5294d36a21)（公式 0.2.15 / エンジン 0.1.60 相当）です。通信、ドラフト選択、プロンプトカーネル、メモリー管理、各 OS の修正を取り込んでいます。フォークのバージョン体系は独立しています。DeepSeek 互換性は公式が削除した経路を独自拡張として維持し、専用ランタイムがなければ読み込みを止めます。この Windows 配布は、ネイティブ画像入力、専用 DeepSeek エンジンの配布、Linux・Mac での推論検証を意味しません。

## 保護条件と測定上の制限

空き RAM 3 GiB、コミット余力 4 GiB、GPU 予約領域 768 MiB 以上、ディスク空き容量の下限を維持します。Windows の RAM 保護は読み込みと生成中も監視します。RAM 32 GB・VRAM 16 GB でもすべてのモデルやコンテキストが収まるわけではありません。最大長推論・一般的なモデル品質・GPU 同時実行は未認定（`releaseQualified=false`）です。インストーラーに Windows Authenticode 署名はありません。過去の測定記録は [リリース記録](docs/HAKUA_RELEASE.md) に元のバージョンのまま残しています。

## 資料と開発

[導入](docs/HAKUA_INSTALL.md) · [リリース記録](docs/HAKUA_RELEASE.md) · [API・ハードウェア](docs/AI_REFERENCE.md) · [Terminal](docs/TERMINAL.md) · [ソースビルド](docs/SOURCE_INSTALL.md) · [パッケージ化](packaging/README.md) · [貢献ガイド](CONTRIBUTING.md) · [セキュリティ](SECURITY.md) · [ライセンス](LICENSE)

```sh
python -m pytest tests/terminal
```

```sh
cd apps/host
pnpm test
pnpm run build
```

### Niko1221/Strata — 上流 Terminal リファレンス

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

[Niko1221/Strata](https://github.com/Niko1221/Strata) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) · [サードパーティ表記](third_party/)

## 0.6.3 の変更点

- ネイティブ Q38 の prefill は、トークン単位でスケールした FP16 活性値と 20 MiB の L2 常駐重みタイルを用いる cuBLAS Tensor Core GEMM に移行しました。decode は形式別の整列 Q8_1 dp4a GEMV と LayerFence(8) の CUDA Graph を使用します。
- 長文脈の受け入れ判定は実測 VRAM に基づく段階的リストから prefill チャンクを選択します。768 MiB の GPU 予約、3 GiB の RAM 下限、4 GiB のコミット下限は変更していません。
- 数値テストの合成重みを実モデル相当の振幅へ正規化しました。厳密な `_exact_q8` 許容値は緩めておらず、非整列の GEMV はバイトロードのスカラーカーネルへフォールバックします。
- 検証: ネイティブテスト 17/17 が成功し、compute-sanitizer の memcheck/synccheck/racecheck は 0 errors / 0 hazards です。一般モデルの品質・全ウィンドウ推論・GPU 並行実行は未検証のままです (`releaseQualified=false`)。
- 調査根拠: NVIDIA CUDA 13.5 リリースノート (cuBLASLt アルゴリズム 66 の Tensor Memory 並行時訂正) と Qwen3.8-Next アーキテクチャ報告 (GDN と全体注意の 3:1 ハイブリッド、r=4 圧縮と K=2048 トークン予算の QSA、ゲート付き残差、ホスト先読み n-gram 埋め込み)。
