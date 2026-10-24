<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="380">
  </picture>
</p>

<p align="center"><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <b>日本語</b> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a></p>

<h3 align="center">最先端のモデルを、手元のゲーミングPCで動かす</h3>

<p align="center">NVIDIA、AMD、Intel に対応 · Windows、Linux、Mac · 無料のオープンソース</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?include_prereleases&label=release" alt="最新リリース"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT ライセンス"></a>
  <a href="https://github.com/Niko1221/Strata/stargazers"><img src="https://img.shields.io/github/stars/Niko1221/Strata?style=flat" alt="GitHub のスター"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?logo=buymeacoffee&logoColor=black" alt="コーヒーをおごる"></a>
</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="Strata のモデルが書いたボクセルの塔の庭園。ブラウザで動いています"></a><br>
<sub>ボクセルの塔の庭園。1回のプロンプトで生成し、RTX 5070 上の Strata で動かしています（IQ3_S、128K コンテキスト） ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">動画全体（49秒）</a></sub></p>

<p align="center">
  <a href="#インストール"><b>インストール</b></a> ·
  <a href="#モデル">モデル</a> ·
  <a href="#どのくらい速い">速度</a> ·
  <a href="#使い方">使い方</a> ·
  <a href="#仕組み">仕組み</a> ·
  <a href="#うまくいかないとき">ヘルプ</a> ·
  <a href="#strata-を応援する">応援する</a>
</p>

Strata は **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** を動かします。1250億パラメータのモデルで、
ふつうはサーバーが必要ですが、Strata なら一般的なPCで使えます。0.2.4 からは、ターミナル版で **DeepSeek V4.1-Flash** も動かせます。
チャット、コード作成、画像の読み取りができ、お使いのアプリやコーディングエージェントとも連携します。チャットとファイルはPCの外に出ません。

- **ゲーミングPCで動きます。** RAMとグラフィックスメモリの合計が32 GBあれば始められます。
- **お使いのツールと連携します。** Claude Code、Codex CLI、OpenCode などのエージェントが、OpenAI と Anthropic の API で接続できます。
- **画像を読めます。** 初期設定でオンです。
- **無料でプライバシーも守ります。** MIT ライセンス。アカウントもクラウドもテレメトリも不要です（[何をいつダウンロードするか](docs/SAFETY.md)）。

## 必要な環境

**RAMとグラフィックスメモリ（VRAM）の合計が、少なくとも32 GB必要です。** MacやRyzen AI Maxでは、ユニファイドメモリがその量にあたります。
RAM 32 GBのPCに12〜16 GBのカードがあれば小さいサイズが動き、64 GBならどのサイズも動きます。

| 動く環境 | ガイド |
| --- | --- |
| **NVIDIA** GeForce RTX 20、30、40、50 シリーズ、12 GB以上 | [インストール](#インストール) |
| **AMD** Radeon RX 6800 以降、Radeon AI PRO R9700 | [AMD のカード](docs/AMD_HIP.md) |
| **Intel** Arc | [Intel Arc](docs/INTEL_ARC.md) |
| **Apple Silicon の Mac**、M1 以降、macOS 15 以降 | [Mac](docs/MAC.md) |
| **AMD Ryzen AI Max（Strix Halo）** などのユニファイドメモリ搭載機 | [Strix Halo](docs/STRIX_HALO.md) |
| 古いカードやプロセッサ | [古いGPU](docs/OLDER_GPUS.md) |

そのほかに、約80 GBの空きディスク容量（SSD）、Windows 10 / 11、Linux、または macOS、最新のグラフィックスドライバーが必要です。
同じメーカーのカードを2〜3枚使って、1つのモデルを動かせます（[マルチGPU](docs/MULTI_GPU.md)）。

## インストール

### 1. Strata アプリ（いちばん簡単）

お使いのPCに合うファイルをダウンロードして開き、すすめられたモデルのインストールを押します。

**Windows 10 / 11**

[![Windows、NVIDIA、ドライバー 580 以降](https://img.shields.io/badge/Windows-NVIDIA%20%C2%B7%20driver%20580%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-cuda13.exe)
[![Windows、NVIDIA、古いドライバー](https://img.shields.io/badge/Windows-NVIDIA%20%C2%B7%20older%20driver-4C7A00?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-cuda12.exe)
[![Windows、AMD RX 9000 または R9700](https://img.shields.io/badge/Windows-AMD%20RX%209000%20%C2%B7%20R9700-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-hip.exe)
[![Windows、AMD RX 7000 または Ryzen AI Max](https://img.shields.io/badge/Windows-AMD%20RX%207000%20%C2%B7%20Ryzen%20AI%20Max-B3131B?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-hip-rdna3.exe)

**Linux**

[![Linux、NVIDIA、ドライバー 580 以降](https://img.shields.io/badge/Linux-NVIDIA%20%C2%B7%20driver%20580%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-nvidia-cuda13.AppImage)
[![Linux、NVIDIA、古いドライバー](https://img.shields.io/badge/Linux-NVIDIA%20%C2%B7%20older%20driver-4C7A00?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-nvidia-cuda12.AppImage)
[![Linux、AMD RX 9000 または R9700](https://img.shields.io/badge/Linux-AMD%20RX%209000%20%C2%B7%20R9700-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-rx9070.AppImage)
[![Linux、AMD RX 7000](https://img.shields.io/badge/Linux-AMD%20RX%207000-B3131B?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-rx7900.AppImage)
[![Linux、AMD Ryzen AI Max](https://img.shields.io/badge/Linux-Ryzen%20AI%20Max-8A0E15?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-strixhalo.AppImage)

**Mac**（Apple Silicon、macOS 15 以降）

[![Mac、Apple Silicon](https://img.shields.io/badge/Mac-Apple%20Silicon-333333?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-mac.dmg)

NVIDIA のドライバーが 580 より古い場合（`nvidia-smi` で確認できます）は、古いドライバー用のファイルを選んでください。AMD の RX 6000 カードは
RX 7000 用のファイルを使います。合うエンジンは自動で取得されます。すべてのファイルは
[リリースページ](https://github.com/Niko1221/Strata/releases/latest)にあります。

| 初回の起動 | すること |
| --- | --- |
| **Windows** | `.exe` を実行します。管理者権限は不要です。「Windows によってPCが保護されました」と表示されたら、**詳細情報**を選び、**実行**を選んでください。 |
| **Linux** | ファイルを実行可能にして（`chmod +x`）起動します。FUSE 2 が必要です。`sudo apt install libfuse2t64`（Ubuntu 24.04 以降）または `libfuse2`（22.04）を実行してください。 |
| **Mac** | `.dmg` を開き、Strata を「アプリケーション」にドラッグします。 |

モデルのダウンロードは大きく（40〜70 GB）、止まっても続きから再開します。

### 2. ターミナル（1行）

`strata` コマンドです。スクリプトやサーバー、AI エージェント向けです。DeepSeek を動かすにもこれを使います。

Linux と Mac:

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

Windows（PowerShell）:

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

そのあと起動します。質問は1つずつ表示されます（モデル、コンテキスト、画像、接続を許可する相手）。必要なものをダウンロードして、
サーバーを開始します。

```sh
strata launch
```

管理者権限は不要です。インストールでは、ダウンロードしたものすべての SHA-256 を検証します。Windows では、インストール後に新しいターミナルを開いてください。
すべての画面とフラグは [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md) にあります。

### 3. AI にセットアップを任せる

Claude Code、Cursor、Codex、GitHub Copilot などのコーディングアシスタントに、次の文を貼り付けます。

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

AI はグラフィックスカード、RAM、ディスクを調べ、合うモデルを選んで、インストールと起動をし、アプリの接続方法を教えてくれます。
AI ツールは [MCP サーバー](docs/MCP_SERVER.md)を通じて、Strata のインストール、起動、停止もできます。

自分でビルドしたい方は[ソースからビルド](#ソースからビルド)を見てください。

## モデル

アプリと `strata launch` は、メモリ量に合うサイズをすすめます。小さいサイズは速く、大きいサイズは少し賢くなります。

### メモリが少ないなら Gyro

[Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)（agentionai 製）は、Qwen3.8-Flash-Next をいちばん小さく動かせる方法です。
NVIDIA、AMD、Intel Arc、Apple Silicon で動きます。

| サイズ | 必要なメモリ（RAM + VRAM） | ダウンロード | 向いているPC |
| --- | ---: | ---: | --- |
| **Gyro-S** | 27.6 GB | 58.5 GB | ほかのサイズが入らないPC向け。32 GB の Mac mini で毎秒 9 トークン。 |
| **Gyro-M** | 35 GB | 92.0 GB | 品質は IQ3_XXS 程度で、GPUメモリの使用量は20%少なくなります。 |

`strata launch --model gyro-s` で起動します。詳細は [docs/GYRO.md](docs/GYRO.md) にあります。

### Qwen3.8-Flash-Next のサイズ

| メモリ（RAM + VRAM） | 選ぶモデル | 理由 |
| --- | --- | --- |
| **32 GB** | **Gyro-S**、またはコード向けの **[Coder](docs/MODELS.md#coder)** | Coder は、コードで使うエキスパートを残します |
| **40 GB** | **[Swift 1.5](docs/MODELS.md#swift-15)** IQ2_XS（39.2 GB）、または **Gyro-M** | Swift は答える前に考える時間が短くなります |
| **48 GB** | **IQ3_XXS**（47.0 GB） | 遅くなりますが、品質は上がります |
| **64 GB** | **IQ3_S**（54.8 GB） | 公開されているテストでは完全版と同等です |
| **96 GB 以上** | Unsloth の **UD-IQ4_XS** から **Q8_0** | 完全版にいちばん近いです |

表の行より RAM が少なくても、遅くなるだけで動くことがあります。Strata は自動で[低RAMモード](docs/MODELS.md#a-big-graphics-card-and-little-ram)に切り替わります。
そのほかに、NVIDIA カード向けの [NVFP4](docs/NVFP4.md)、[Qwen3.6-35B-A3B](docs/QWEN36.md)（より小さく、複数のリクエストを同時に処理）、
[OrcaRouter の Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs) があります。すべてのサイズと、どの構成で何が動くか:
[docs/MODELS.md](docs/MODELS.md)。

### DeepSeek V4.1-Flash（ターミナル版、0.2.4 から）

ずっと大きなモデルで、12 GB のグラフィックスカード 1 枚と 46 から 64 GB の RAM で動きます。Strata はよく使う部分をメモリに置き、
残りは SSD から読みます。セットは [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) が作りました。
NVIDIA RTX 30 から 50 シリーズのカードと Radeon AI PRO R9700 で動きます。`strata launch` で選んでください。デスクトップアプリはまだ対応していません。

| セット | ダウンロード | 品質 | コードの出力速度 |
| --- | ---: | --- | --- |
| **IQ1_M**（1.75 ビット） | 239 GB | いちばん小さく、ふつうの文章は弱めです | PC ではまだ測っていません |
| **IQ2_XXS**（2.00 ビット） | 257 GB | バランスが良いです | 毎秒 10 トークン（RTX 3060）、5（RTX 5070）、48（R9700 4 枚） |
| **IQ2_XS**（2.25 ビット） | 274 GB | 元のモデルにいちばん近いです | 毎秒 15 トークン（R9700 1 枚）、80（R9700 4 枚） |
| **IQ4_NL**（Engram テーブル） | 上のどのダウンロードにも含まれます | 3 つのセットが共有する参照テーブルです | |

`strata launch --model ist-das-2.00bpw`（または `ist-das-1.75bpw`、`ist-das-2.25bpw`）で起動します。品質の数値、PC ごとの速度、手動インストールは
[docs/DEEPSEEK41.md](docs/DEEPSEEK41.md) にあります。

## どのくらい速い？

12 GBのグラフィックスカード（RTX 5070、RAM 64 GB）での数値です。ほかのカードの結果は
[各サイズの速度](docs/MODELS.md#how-fast-is-each-size)と[コミュニティの結果](docs/COMMUNITY_BENCHMARKS.md)を見てください。

| モデル | 回答が出る速さ | プロンプトを読む速さ |
| --- | --- | --- |
| **Q2_0** | `████████████` 94 tokens/s | `████████████` 2,650 tokens/s |
| **IQ2_XS** | `██████████░░` 79 tokens/s | `█████████░░░` 2,090 tokens/s |
| **IQ3_XXS** | `████████░░░░` 62 tokens/s | `████████░░░░` 1,750 tokens/s |
| **IQ3_S** | `███████░░░░░` 53 tokens/s | `███████░░░░░` 1,620 tokens/s |
| **Coder** | `███████░░░░░` 55 tokens/s | `██████████░░` 2,180 tokens/s |

16 GBの AMD カード（RX 9070 XT、RAM 47 GB）の場合:

| モデル | 回答が出る速さ | プロンプトを読む速さ |
| --- | --- | --- |
| **Q2_0** | `████████░░░░` 60 tokens/s | `█████░░░░░░░` 1,160 tokens/s |
| **IQ2_XS** | `███████░░░░░` 52 tokens/s | `█████░░░░░░░` 1,110 tokens/s |
| **Coder** | `██████░░░░░░` 44 tokens/s | `██████░░░░░░` 1,420 tokens/s |

1トークンは英単語の約¾にあたるので、毎秒60トークンは読む速さより速くなります。「プロンプトを読む速さ」は、長い文書やコード、チャット履歴を Strata が取り込む速さです。

## 使い方

### アプリで

チャット、モデルとGPU・CPU・RAMの状態を見る**モニター**、**モデル**ページ、**設定**があります。チャットはPC上のフォルダで作業できます。
**手動**モードでは、ファイルを変更したりコマンドを実行したりする前に確認します。ウェブ検索もできます（DuckDuckGo または Bing、アカウント不要）。
PCの外に出るのは検索語だけです。

<p align="center"><img src="docs/media/app-chat.png" width="900" alt="Strata アプリ。エージェントが小さなボクセルの塔の庭を作ったチャットで、右側のプレビューにそのページが開いています"><br>
<sub>エージェントがボクセルの塔の庭を 1 つの HTML ファイルとして書き、そのページがチャットの横のプレビューで開きます</sub></p>

### ターミナルで

<p align="center"><img src="docs/media/terminal-monitor.png" width="900" alt="モデルが返信を書いている間の Strata ターミナルのモニター概要"><br>
<sub>モデルが返信を書いている間の <code>strata monitor</code>。速度、GPU、メモリ、最近のリクエストが表示されます</sub></p>

```sh
strata launch                              # guided: model, context, pictures, access, then serve
strata launch --model q2_0 --yes --no-tui  # no questions: serve and print the address
strata launch --status                     # what is being served now
strata launch --stop                       # stop it
strata monitor                             # watch speed, GPU, memory and requests
strata update                              # get the newest engine
strata doctor                              # check the installed files
```

スマートフォンや別のPCから Strata に接続するには、`strata launch --access lan` を使います。API キーが自動で作られます。
`strata --help` ですべてのコマンドを確認できます。説明は [docs/TERMINAL.md](docs/TERMINAL.md) にあります。

### AI エージェントやほかのアプリと使う

エージェントは、上の1行コマンドで Strata をインストールし、`strata launch --model NAME --yes --no-tui` を実行します。手順、フラグ、終了コードは
[docs/AI_SETUP.md](docs/AI_SETUP.md) にあります。エージェント向けのリポジトリのルールは [AGENTS.md](AGENTS.md) にあります。

| アプリが使うもの | ベース URL |
| --- | --- |
| OpenAI API（ほとんどのアプリとエージェント） | `http://127.0.0.1:8080/v1`（API キーもモデル名も何でも可） |
| Anthropic API（Claude Code） | `http://127.0.0.1:8080`、`/v1` は付けません。Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` |
| OpenAI Responses API（Codex CLI） | `http://127.0.0.1:8080/v1/responses`（[設定方法](docs/DETAILS.md#the-responses-api-and-codex-cli)） |

`strata connect` は、OpenCode、Codex CLI、Claude Code、oh-my-pi、pi、Hermes Agent、OpenClaw の正確な設定を表示します。
`strata connect opencode --write` なら、設定を書き込みます（バックアップ付き）。

### 知っておくと便利なこと

- **画像**は、画像を読めるモデルすべてでオンです。チャットに画像を添付または貼り付けるか、アプリから送ってください。詳細: [docs/IMAGES.md](docs/IMAGES.md)。
- **思考のレベル**は off、low、medium、high から、チャットのメニューか、お使いのアプリの「reasoning effort」で選びます。off がいちばん速くなります。
- **チャットの最初のメッセージ**は全文読みます。30,000トークンあたり約1分かかります。続きのメッセージは数秒で始まります。
- **一度に1つのリクエスト**が初期設定です。複数にするには `--parallel N` を使います（[BATCHING.md](docs/BATCHING.md)）。12 GBのカードでは、1つずつの回答が遅くなります。

## ソースからビルド

<details>
<summary>リポジトリから Windows、Linux、Mac 用にビルドする</summary>

[Strata をダウンロード](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip)して展開するか、`git clone` します。
**Windows:** `START-HERE.bat` をダブルクリックします。**Linux:** Strata フォルダで `./setup.sh` を実行します。

**Mac（Apple Silicon、macOS 15 以降）:** 最初に一度、`xcode-select --install` で Apple のコマンドラインツールをインストールします
（Xcode 本体は不要です）。Strata フォルダでターミナルを開き、`./setup.sh` を実行します。Mac 用エンジンをお使いの Mac 上でビルドし
（約7分）、同じ質問をして、モデルをダウンロードし、起動します。2回目以降は `./setup.sh` ですぐに起動します。
お使いの Mac に合うサイズは [docs/MAC.md](docs/MAC.md) を見てください。

インストーラーはカードを見つけて、合うエンジンを設定します。どのモデルとサイズにするか、コンテキストをどれだけにするか（モデルが一度に覚えておけるテキストの量）、
画像を読ませるかを聞かれます。おすすめの答えでよければ Enter を押します。そのあとモデル（約70 GB）をダウンロードして起動します。
ダウンロードが止まったら、もう一度実行すると続きから再開します。ブラウザで `http://127.0.0.1:8080` の Strata ページが開きます。

> モデルの起動中は、PCが1〜3分ほど（初回がいちばん長い）遅くなったり、応答しなくなったりすることがあります。Strata は35〜55 GBをRAMに読み込み、
> その一部をグラフィックスカード用に固定するためです。ウィンドウを閉じずに待ってください。

次回からは `START-HERE.bat`（または `./setup.sh`）をもう一度実行するだけです。すぐに起動し、同じものを2回ダウンロードすることはありません。ウィンドウを閉じるとモデルが止まります。
`UPDATE.bat`（`./update.sh`）は、起動せずに Strata を更新します。更新、Docker、複数のカード、ファイルの保存先、すべてのオプションは
[docs/INSTALL.md](docs/INSTALL.md) にあります。

PCの性能を測るには、`BENCH.bat`（`./bench.sh`）を実行します。5〜10分かかり、共有できるレポートが作られます
（[方法](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)）。

**オプション: モデル全体がRAMに入るとき、1枚のカードをさらに速くする。** RubenTorney による、自分でオンにする2つの設定です（#1904、#1905）。
グラフィックスカードが1枚で、RAM がすべてのエキスパートを保持できるときに効果があります。たとえば64 GBのPCでの Q2_0 や IQ3_XXS です。
Strata のデータフォルダにある `strata-<model>.json`（[場所](docs/INSTALL.md#where-things-are-stored)）を開いて、次を追加します。

- `"args"` のリストに `"--adapt-async", "1"` を追加します。エキスパートキャッシュが、回答を止めずに補助スレッドで更新されます。
  RTX 5070（Windows、Q2_0）での測定: デコードが4〜5%、コードや長い文書では7〜10%速くなりました。回答は初期設定と少し違うことがありますが、実行ごとの結果は変わりません。
- `"env"` ブロックに `"STRATA_PIN_BY_NEED": "1"` を追加します（Windows）。Windows ではグラフィックスカード用に固定できる RAM が半分ほどしかありません。
  この設定は、カードが最初に必要とするエキスパートを優先して固定します。RTX 5070 での測定: プロンプトの読み取りが4.4%速くなりました。

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

変更したら Strata を再起動してください。元に戻すには、2つの項目を削除します。

</details>

## 仕組み

このようなモデルは、ふつう数百GBのグラフィックスメモリを持つサーバーで動きます。お使いのグラフィックスカードは12〜24 GBです。Strata は、PC全体で作業を分担して、モデルを収めます。
図書館にたとえてみます。

- モデルは、24,576人の小さな専門家（「エキスパート」）からなるチームです。24,576冊の参考書のようなものです。1つの単語に必要なのは、そのうち10冊だけです。
- **グラフィックスカードは机です。** 小さくて速く、最もよく使う約4,000冊が開いて置いてあります。
- **RAM は閲覧室の書棚です。** 24,576冊すべてがここにあります。単語に必要な本が机になければ、プロセッサがすぐに書棚から読み、その間もカードは作業を続けます。
- **SSD は書庫です。** 29 GBの参照テーブルがしまってあります。モデルは単語ごとにその数行だけを読むので、RAM に置いておく必要はありません。

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="モデルの24,576人のエキスパート。よく使うものはグラフィックスカードに、全員が RAM に、参照テーブルは SSD にあります"></p>

小さな助手が、次の数単語も予想します。大きなモデルがそれをまとめて確認し、正しいものだけを残します。答えは同じで、1.6〜1.8倍早く返ります。
長い文章は、一度に最大8,192トークンの大きな塊で、毎秒1,000トークンを超える速さで読みます。

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="小さな助手が次の単語を予想し、大きなモデルがそれをまとめて確認して、正しいものだけを残します"></p>

詳しい説明は [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) にあります。各部分と数値は
[詳細](docs/DETAILS.md#how-it-works)と[論文](docs/paper/Strata-Paper.pdf)を見てください。

## うまくいかないとき

- **Strata を初めて起動したときに、PCが固まりました。** モデルを RAM に読み込んでいます。ウィンドウを閉じずに待ってください。10分たっても固まったままなら、
  PCを再起動し、ほかのプログラムを閉じてからもう一度試すか、小さいサイズを選んでください。
- **ダウンロードやインストールの途中で止まりました。** アプリか `strata launch` をもう一度起動してください。止まったところから再開します。
- **とても遅い、または「the engine stopped unexpectedly」と表示される。** PCの空き RAM が足りません。
  ほかのプログラムを閉じるか（ブラウザはメモリを多く使います）、Q2_0 や IQ2_XS のような小さいサイズを選んでください。
- **このPCに合うエンジンがないと表示される。** アプリでは、**設定、エンジン**を開いて**エンジンを更新**を押します。ターミナルでは
  `strata update` を実行します。
- **ポート 8080 はすでに使われていると表示される。** Strata がすでに動いています。ウィンドウを探してください。

そのほかの対処法は [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) にあります。それでも解決しない場合は、
[issue](https://github.com/Niko1221/Strata/issues) を作成して、ログを添付してください。アプリでは**設定、エンジン、診断、ログを保存**、
ターミナルでは `strata doctor` の出力です。セキュリティ上の問題は、非公開で報告してください: [SECURITY.md](SECURITY.md)。

## クレジットとライセンス

モデルは Qwen チームの [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) です。圧縮は
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)、UkisAI（Swift 1.5）、agentionai（Gyro）、
Unsloth が行いました。DeepSeek V4.1-Flash のセットも ISTA-DASLab が作りました。Strata は
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) の一部を使っています。すべてのクレジットは
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits) にあります。Strata は [MIT ライセンス](LICENSE)のオープンソースです。一部のコンポーネントと
すべてのモデルには、それぞれのライセンスがあります（[該当するもの](docs/HOW_IT_WORKS.md#license)）。

## コントリビューション

バグを見つけた、変更やベンチマークを送りたい、という方は、先に [CONTRIBUTING.md](CONTRIBUTING.md) を読んでください。
ほかの人があなたの成果を確認できるように、何を書くべきかが説明してあります。

## Strata を応援する

Strata は無料で、MIT ライセンスです。私が入手できるハードウェアで開発とテストをしています。お使いのPCで快適に動いたなら、コーヒー1杯分の支援は、
新しいカードのテストと、報告いただいた不具合の修正を続ける助けになります。無理のない範囲で大丈夫です。使って、気づいたことを報告してもらえるだけでも同じように助かります。

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>

Strata のスポンサーに関心がある方は、[GitHub](https://github.com/Niko1221) から連絡してください。
