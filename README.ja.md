<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">あなた自身のパソコンで、高性能な AI を。</h3>
<p align="center">チャット、コーディング、画像の処理を、すべてローカルで。</p>

<p align="center">
  <a href="#モデル">モデル</a> · <a href="#インストール"><b>インストール</b></a> ·
  <a href="#ハードウェア">ハードウェア</a> · <a href="#ツール">ツール</a> ·
  <a href="#ヘルプ">ヘルプ</a> · <a href="#strata-を支援する">支援</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Strata をダウンロード"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT ライセンス"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Buy Me a Coffee で Strata を支援する"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <b>日本語</b> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

**無料でオープンソース。自分の環境で動かせます。** デスクトップで、またはお好みのコーディングツールと一緒に Strata を使えます。アカウントは不要です。

> **個人フォーク (c-OOL-io/Strata-WebSearch)。** このフォークは内蔵チャットと serve フロントエンドに**ウェブ検索と完全なページ読み取り**を追加します: 上流に検索がない軽量なターミナルのウェブアプリです (上流の検索は App 側にあります)。Strata のチャットと同じ作り方の思想です: **デフォルトで完全にオフ** (オンにするまで検索関連のものが PC から出ることはなく)、オフのチャットには**トークンを追加せず**、検索テキストはモデルのコンテキストウィンドウの一部に予算制限されます (既定値は `max_context` に応じてスケールし、小さいコンテキストのインスタンスも安全)。
>
> - **検索エンジン:** 既定は DuckDuckGo (キー不要)、SearXNG (自分のセルフホストインスタンス)、Brave、Tavily、Serper (Google)、Exa、Perplexity は自分のキーで利用。キーはブラウザー内に留まり、検索関連のものがサーバーに保存されることはありません。
> - **クエリオプティマイザー:** 口語的な表現はローカルモデルへの 1 回の高速な呼び出しで明確な検索クエリに書き換えられ、書いた言語のまま保たれます。失敗した場合は元のメッセージがそのまま検索されます。実際に検索されたものが入力と異なる場合、Sources 行に表示されます。
> - **結果ページの読み取り:** 上位の結果ページを取得し、抽出した本文として注入します (ナビ/フッター/スクリプトを除去、`main`/`article` を優先)。Direct / Jina フォールバック / Jina すべて の 3 段階のリードモードと、サードパーティが URL を見うる場合の明示的な警告付き。
> - **検証可能な回答:** すべての回答に `Sources` 行 (エンジン、実際のクエリ、クリック可能なリンク、`read` バッジ) が付き、Markdown 書き出しにも含まれます。検索の失敗は静かに失敗するのではなく、原因名付きの警告として残ります (キー未設定、エンジン到達不能、bot チェック)。
> - **強化:** `/search` はアプリの own-page ガードの後にあります (外部 Origin 拒否)、http(s) のみの URL チェックでページ読み取りがローカルファイルの読み取りになることはなく、注入されるウェブテキストは信頼できないものとして扱われ、バックエンド (`serve/websearch.py`) は標準ライブラリのみ、新しい依存なし、Windows/Linux/macOS 対応。
> - **今回の更新の新機能:** コンポーザーの地球アイコンが **Off → On → Auto** を循環するようになりました (Auto はメッセージごとにモデルが判断し、失敗時はキーワードのヒューリスティックにフォールバック)。オプションのブラウザー内のみの**メッセージボックス下書き** (既定ではオフ) はリロードやチャット切り替え後も残り、チャット設定から**14 言語の UI** (de, es, fr, ja, ko, pt, ru, zh, hi, ar は RTL、tr, it, pl) を選べます。言語リストは英語を先頭に、次にブラウザーの言語（対応していれば）、その後はアルファベット順に並び、それぞれ 2 文字のコード付きで表示されます。
>

> ダウンロード前に 2 つの注意点:
>
> 1. **下のダウンロードリンクは Niko1221/Strata の素のアップストリームアプリ**です。プリビルドバイナリにこのフォークの検索は**入りません**。検索付きの版はソースのみ:`git clone`、ブランチ `web-search`、その後 START-HERE.bat / setup.sh。
> 2. **このフォークは軽量な Terminal 経路のためでもある**:ヘッドレスサーバーで簡素な serve/CLI だけを使う人 (アプリなし、チャット UI なし) も同じソースから使えます。検索は serve フロントエンド内蔵。
>
> **CLI でのインストール**（このフォーク、`web-search` ブランチ）：Strata のソースと同じようにクローンしてセットアップします（検索は内蔵）。
> このフォーク独自のインストールガイド（ターミナル / ソース、更新方法込み）は [docs/INSTALL.FORK.md](docs/INSTALL.FORK.md)：
>
> ```sh
> git clone -b web-search https://github.com/c-OOL-io/Strata-WebSearch.git && cd Strata-WebSearch
> ./setup.sh          # first run: .venv, the engine build, the model; then it serves
> ```
>
> RTX 4080 SUPER / Ryzen 9 7950X3D のベンチ結果は `bench/` にあります。

<p align="center">
  <img src="docs/media/web-search-chat.png" width="360" alt="Strata チャット内のウェブ検索">
  <img src="docs/media/web-search-settings.png" width="360" alt="設定のウェブ検索:エンジンとキー">
</p>
<p align="center"><sub>左:回答の下の Sources 行(使用エンジンと実際に検索したクエリ)付きのチャット。入力欄(composer)の地球アイコンで切り替え。<br>右:設定の Web search 区:オン/オフ、クエリの最適化、エンジン選択(DuckDuckGo / SearXNG / Brave / Tavily / Serper / Exa / Perplexity)、キー入力、ページ読み取り。</sub></p>

## モデル

まず **ISTA-DASLab** をおすすめします。次に **Unsloth** などの提供元です。メモリに収まる互換モデルがある場合に選べます。

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| 提供元 | 選択肢 |
| --- | --- |
| **ISTA-DASLab · おすすめ** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S。[Coder: IQ1\_M](docs/MODELS.md#coder)（コーディング向け）。 |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)。メモリ使用量は少ないですが、動作は遅くなります。 |
| **その他の選択肢** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[サイズ、必要なメモリ、互換性を比べる](docs/MODELS.md)。

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0。現在、このモデルで文書化されているパックです。

[モデルガイドと互換性](docs/QWEN36.md)。

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

**デスクトップアプリ**と**ターミナル**で使えます。メモリに収まらない分は SSD から読み込むため、空きディスク容量がたくさん必要です。

| セット（IST-DAS） | ダウンロード | 必要な空き容量 |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | 約245 GB |
| 2.00 bpw | 257 GB | 約265 GB |
| 2.25 bpw | 274 GB | 約280 GB |

NVIDIA RTX 30・40・50 シリーズ（Windows と Linux）と、AMD Radeon AI PRO R9700（Linux）で動きます。セットアップがファイルと DeepSeek エンジンをダウンロードし、モデルを起動します。

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

Strata 0.2.5 で 2.00 bpw のセットを使って測定しました。「出力」は回答の速さ、「プロンプト」は Strata が入力を読み込む速さです。

| ハードウェア | 出力 | プロンプト |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB RAM | 4〜6 トークン/秒 | 約 110 トークン/秒 (4K) |
| RTX 3060 12 GB, 46 GB RAM | 9〜11 トークン/秒 | 約 155 トークン/秒 (4K) |
| 1x Radeon AI PRO R9700, 251 GB RAM | 21〜25 トークン/秒 | 320〜360 トークン/秒 (8K) |
| 4x Radeon AI PRO R9700 | 74〜77 トークン/秒 | 400〜450 トークン/秒 (8K) |

[メモリ、ディスク、詳細](docs/DEEPSEEK41.md) · [リリースノート](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

どれが動くかわからない場合は、[お使いのパソコンに合うモデルを選ぶ](docs/MODELS.md#pick-by-memory)。

## インストール

<a id="option-a-the-strata-app"></a>

### デスクトップアプリ

**[Windows、Linux、Mac 用の Strata をダウンロード →](https://github.com/Niko1221/Strata/releases/latest)**

アプリをインストールし、モデルを選んで、チャットを始めましょう。インストーラーはシステムごとに1つです。初回起動時に、アプリがお使いのグラフィックスカード用のエンジンをダウンロードします。[インストーラーのヘルプ](docs/INSTALL.md#the-app)。

<sub>ファイル: `Strata-0.2.8-windows-x64-setup.exe` · `Strata-0.2.8-linux-x86_64.AppImage` · `Strata-0.2.8-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="チャットとライブのコードプレビューを並べて表示した Strata デスクトップアプリ">
  <br><sub>チャット、ファイル、ライブプレビューを1つのウィンドウで。</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### ターミナルアプリ

ターミナルのほうが好みですか？ Strata をインストールして、**`strata launch`** を実行します。

<details open>
<summary><b>インストールコマンド</b></summary>

**Linux / Mac**

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

Windows では、インストール後に新しいターミナルを開きます。そのあと:

```sh
strata launch
```

画面の案内に従って、モデルをダウンロードして起動します。[ターミナルガイド](docs/TERMINAL_LAUNCH.md)。

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="モデルの動作、速度、メモリ、最近のリクエストを表示する Strata ターミナルアプリ">
  <br><sub>動作中のモデルとハードウェアの状態を確認できます。</sub>
</p>

<a id="build-from-source"></a>

[ソースからビルド](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [インストールガイド（全文）](docs/INSTALL.md)

<a id="what-you-need"></a>

## ハードウェア

| 対応ハードウェア | ガイド |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50 シリーズ** | [NVIDIA のセットアップ](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900、RX 9060 / 9070、AI PRO R9700** | [AMD のセットアップ](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · 試験的** | [RX 6000 のセットアップ](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Intel のセットアップ](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 以降** | [Mac のセットアップ](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Strix Halo のセットアップ](docs/STRIX_HALO.md) |

<a id="using-it"></a>

## ツール

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**。OpenAI 互換と Anthropic 互換の API を通じて、ほかのツールも使えます。

**`strata connect`** を実行すると、お使いのツールの設定方法が表示されます。[接続ガイド](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect)。

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## ヘルプ

[すべてのガイド](docs) · [トラブルシューティング](docs/TROUBLESHOOTING.md) · [不具合を報告](https://github.com/Niko1221/Strata/issues/new/choose) ·
[ベンチマーク](docs/COMMUNITY_BENCHMARKS.md) · [仕組み](docs/HOW_IT_WORKS.md) · [貢献する](CONTRIBUTING.md)

## Strata を支援する

**ローカル AI をみんなに開かれたものに。** コーヒー1杯が、開発と、より多くのハードウェアでのテストの支えになります。

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: Strata を支援する" height="48"></a>

[MIT ライセンス](LICENSE) · [クレジットとモデルのライセンス](docs/HOW_IT_WORKS.md#credits) · [プライバシー](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>AI ツールと自動リーダー向け: <a href="AGENTS.md">技術リファレンス</a> · <a href="docs/AI_SETUP.md">セットアップ手順</a></sub>

**スポンサー、ハードウェア提携、共同開発のご相談:** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).
