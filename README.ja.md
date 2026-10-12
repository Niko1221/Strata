<h1 align="center">Strata</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · **日本語** · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>1,250 億パラメータの AI モデルを、手元のゲーミング PC で動かす</b><br>
NVIDIA または AMD のグラフィックカード（12 GB 以上） · Windows または Linux · 無料・オープンソース</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="Strata で生成した、五重塔のある庭。ボクセルで表現され、ブラウザーで動作している"></a><br>
<sub>五重塔のある庭をボクセルで表現。RTX 5070 で動かした Strata（IQ3_S、コンテキスト長 128K）に、1 回のプロンプトで作成を依頼 ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">動画を見る（全編 49 秒）</a></sub></p>

Strata を使うと、**[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** を一般的な PC で動かせます。
通常はサーバーを必要とする大規模で高性能な AI モデルで、チャットやコードの生成、画像の読み取りができます。
アプリやコーディングエージェントとも連携でき、データは PC の外に送信されません。

## どのくらい速い？

一般的なゲーミング PC 2 台で速度を測定しました。英語の場合、1 トークンはおよそ ¾ 語に相当します。

- **生成速度：** 短いチャットで回答を生成する速度です。毎秒 60 トークンなら、人が読むよりも速く文章が表示されます。
- **プロンプトの読み込み：** 入力した内容を読み込む速度です。ここでは、文書やコード、チャット履歴など 32K トークン分を読み込ませています。

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>
<tr><td>

| モデルサイズ | 生成速度 | プロンプトの読み込み |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| モデルサイズ | 生成速度 | プロンプトの読み込み |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA の測定には、Q2_0 のみエンジン 0.1.36、それ以外は 0.1.26 を使用しています（回答 4K、プロンプト 32K）。測定結果の詳細は
[DETAILS.md](docs/DETAILS.md#speed-measured) をご覧ください。VRAM の容量が大きいカードほど高速で、RTX 3090（24 GB）なら
毎秒およそ 100-140 トークンの生成速度が見込めます。長いチャットや他のカードでの性能については、[モデルごとの速度](docs/MODELS.md#how-fast-is-each-size)や
[コミュニティの測定結果](docs/COMMUNITY_BENCHMARKS.md)を参照してください。

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>Strata は無料で使えます。お使いの PC で快適に動いたら、コーヒー 1 杯分のご支援が開発を続ける励みになります。</sub></p>

## 必要なもの

| | |
| --- | --- |
| **グラフィックカード** | **NVIDIA** GeForce RTX 20、30、40、50 シリーズ、または **AMD** Radeon RX 7900 XT / XTX、RX 7800 XT / 7700 XT、RX 9060 XT、RX 9070 / 9070 XT、Radeon AI PRO R9700、RX 6800 / 6900 シリーズ。**VRAM 12 GB 以上**が必要です。 |
| **RAM** | 32 GB 以上。[利用できるモデル](#どのモデルを選べばいい)は RAM の容量によって異なります。64 GB あれば、どのサイズも利用できます。 |
| **ディスク** | 約 80 GB の空き容量。SSD を使うと初回の起動が大幅に速くなるため、おすすめです。 |
| **OS** | Windows 10 / 11 または Linux。NVIDIA または AMD の最新のグラフィックドライバーも必要です。 |

その他の必要なソフトウェアは、インストーラーが用意します。2 枚または 3 枚のカードにモデルを分散させることもできます（[マルチ GPU](docs/MULTI_GPU.md)）。

以下のハードウェアにも試験的に対応しています。いずれもコミュニティのメンバーが手元のマシンで実装・テストしたものです。

- **旧世代のグラフィックカード**（Tesla P40 / V100、GTX 10、Radeon VII / MI50、RX 6700 XT、RX 5500 XT）：[旧世代 GPU の対応状況](docs/OLDER_GPUS.md)。
- **Intel Arc**（Linux でソースからビルド）：[Intel Arc の対応状況](docs/INTEL_ARC.md)。
- **AMD Ryzen AI Max（Strix Halo）**（Linux でソースからビルド）：[Strix Halo の対応状況](docs/STRIX_HALO.md)。
- **AVX2 非対応の旧世代 CPU**：動作しますが、速度は遅くなります。[旧世代 CPU の対応状況](docs/INSTALL.md#older-cpus-experimental)を参照してください。

必要な環境の一覧は、[docs/INSTALL.md](docs/INSTALL.md#what-you-need)をご覧ください。

## インストール

### AI にセットアップしてもらう

Claude Code、Cursor、Codex、GitHub Copilot などの AI コーディングアシスタントを使っている場合は、次の文を貼り付けてください。

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

AI がグラフィックカード、RAM、ディスクを確認して、PC に合ったモデルを選びます。インストールと起動を行い、
アプリとの接続方法も案内します。また、Strata の [MCP サーバー](docs/MCP_SERVER.md) を使えば、
AI ツールから Strata のインストール・起動・停止を行えます。

### 自分でやる

[Strata をダウンロード](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip)して展開します（`git clone` でも構いません）。
**Windows：** **`START-HERE.bat`** をダブルクリックします。**Linux：** Strata フォルダーで **`./setup.sh`** を実行します。

NVIDIA でも AMD でも手順は同じです。インストーラーがカードを検出し、対応するエンジンを用意します。
続いて、次の項目を設定します。

- 使用するモデルとサイズ
- コンテキストの長さ（モデルが覚えておけるテキストの量）
- 画像を読み取る機能を使うかどうか

各項目で Enter を押すと、推奨設定が選ばれます。設定が終わるとモデル（約 70 GB）をダウンロードし、起動します。
ダウンロードが途中で止まった場合は、もう一度実行すると続きから再開できます。起動が完了すると、ブラウザーで
`http://127.0.0.1:8080` が開き、Strata の画面が表示されます。

> **モデルの起動中は、1-3 分ほど PC の動作が遅くなったり、応答しなくなったりすることがあります**（初回は特に時間がかかります）。
> Strata が 35-55 GB のデータを RAM に読み込み、その一部をグラフィックカード用に固定（ピン留め）しているためです。
> 正常な動作なので、ウィンドウに表示される処理状況を確認しながら、閉じずにお待ちください。

**次回からは** `START-HERE.bat`（または `./setup.sh`）を実行するだけですぐに起動できます。2 回目以降は、ダウンロード済みのファイルを再度取得する必要はありません。
モデルを停止するにはウィンドウを閉じてください。`UPDATE.bat`（`./update.sh`）を使うと、Strata を起動せずに更新できます。
更新方法や Docker、複数のカードの使い方、ファイルの保存先、各種オプションについては、[docs/INSTALL.md](docs/INSTALL.md)をご覧ください。

## どのモデルを選べばいい

インストーラーが RAM の容量に合ったモデルをおすすめします。同じモデルでも、圧縮率によって複数のサイズが用意されています。
サイズが小さいほど高速に動作し、大きいほど回答の品質が少し高くなります。

| RAM | おすすめのモデル | 選ぶ理由 |
| --- | --- | --- |
| **32 GB** | **Coder** | 32 GB で動作するコーディング向けモデル（24 GB のカードがあれば Q2_0 と IQ2_XS も動作） |
| **48 GB** | **IQ2_XS**（速度優先なら Q2_0） | これより大きいサイズはメモリに収まらない |
| **64 GB** | **IQ2_XS**（おすすめ）、または IQ3_XXS / IQ3_S | どのサイズも利用可能。IQ3_S は回答の品質が最も高い一方、速度は最も遅い |
| **96 GB 以上** | **IQ3_S**、または Unsloth の UD-IQ4_XS（約 4-bit） | 他のアプリを開いたままでも、最大サイズのモデルを動かせるだけの余裕がある |

- **[Coder](docs/MODELS.md#coder)：** エキスパートを半分に減らしたコーディング向けモデルです。SWE-bench Verified ではフルモデルのスコアの
  91% を達成しており（開発者による測定）、32 GB の RAM で動作します。ただし、コード以外の用途では性能が劣り、中国語などの CJK 言語の処理も苦手です（#438）。
  こうした用途には、すべてのエキスパートを残した Q2_0、IQ2_XS、IQ3_S を選んでください。
- **[Swift 1.5](docs/MODELS.md#swift-15)：** 回答前の思考時間が大幅に短くなるようファインチューニングしたモデルです。
  回答の品質をほぼ保ったまま、より短い時間で答えが返ってきます。
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs)：** Unsloth による約 4-bit のモデルです。回答の品質は IQ3_S と UD-Q4_K_XL の中間です。
  ダウンロード容量は 94 GB です。RAM が約 80 GB 未満の場合、回答の生成中にモデルの一部を SSD から読み込むため、速度が落ちます（NVMe SSD を使うと改善します）。
- **[Unsloth UD-Q4_K_XL](docs/MODELS.md#unsloth-ud-q4_k_xl-experimental)**（試験的）：回答の品質がフルモデルに最も近いモデルです。
  ただし、回答の生成中にモデルの大部分を SSD から読み込むため、RAM が 64 GB の PC では毎秒 7-8.5 トークン程度の速度になります。
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs)：** 手動でのセットアップが必要です。
  インストーラーのメニューからは選べません。

各サイズのダウンロード情報や必要なメモリ容量は、[docs/MODELS.md](docs/MODELS.md)をご覧ください。後から別のモデルを追加する場合は、
`SETUP.bat`（Linux：`./setup.sh --setup`）を実行してください。

## 使い方

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="Strata の Monitor タブ（左）と、コードを生成するコーディングエージェント（右）"><br>
<sub>動画に登場する五重塔の庭を作成中の画面。左が Strata の <b>Monitor</b>、右がコードを生成しているコーディングエージェント</sub></p>

- **ブラウザーで使う：** `http://127.0.0.1:8080` を開きます。チャット用の **Chat**、モデルや GPU/CPU/RAM の状態をリアルタイムで確認できる **Monitor**、
  設定や接続先のアドレスを確認できる **About** の各タブがあります。
- **アプリやコーディングエージェントと連携する：** 「OpenAI-compatible」プロバイダーを追加し、ベース URL に
  **`http://127.0.0.1:8080/v1`** を指定します。API キーとモデル名には任意の値を指定できます。
  - Anthropic API を使うアプリ：`http://127.0.0.1:8080/v1/messages`（Claude Code では
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` を設定）。
  - Codex CLI など、OpenAI Responses API を使うアプリ：`/v1/responses`
    （[設定方法](docs/DETAILS.md#the-responses-api-and-codex-cli)）。
- **思考（Thinking）の設定：** チャットのメニュー、または連携先アプリの「reasoning effort」で **off、low、medium、high** から選びます。
  off が最も速く、難しい質問には high が向いています。
- **画像を読み取る：** セットアップ時に「Images?」で yes を選びます。その後、チャットの **Picture** ボタンをクリックするか、連携先のアプリで画像を添付してください。
  AMD のカードを使う場合、Linux では CPU で画像を読み取ります。Windows ではまだ画像の読み取りに対応していません。
- **スマートフォンや別の PC から使う：** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>` で起動します。API キーは必ず設定してください。
- **リクエストは 1 つずつ：** 標準設定では 1 つずつ処理し、それ以外のリクエストは順番を待ちます。同時に処理するには、
  `"parallel": 2` を設定してください（[BATCHING.md](docs/BATCHING.md)）。12 GB のカードでは、リクエストごとの回答速度が落ちます。
- **長いプロンプトを送る：** チャットの最初のメッセージは全文を読み込むため、30,000 トークンあたり約 1 分かかります。
  2 通目以降は数秒で応答が始まります。

[チャットの保存場所](docs/INSTALL.md#where-things-are-stored)や[API の詳細](docs/DETAILS.md#using-it)も参照してください。

## うまくいかないとき

- **初回起動時に PC が応答しなくなる。** モデルの読み込み中は、一時的に応答しなくなることがあります。ウィンドウを閉じずにお待ちください。
  10 分たっても応答しない場合は、PC を再起動して他のアプリを閉じてからやり直すか、モデルのサイズを小さくしてください。
- **ダウンロードやインストールが途中で止まる。** `START-HERE.bat`（または `./setup.sh`）をもう一度実行してください。
  中断したところから再開できます。
- **動作が極端に遅く、ディスクのアクセスランプが点滅し続ける。または「the engine stopped unexpectedly」と表示される。** RAM の空き容量が不足しています。
  ブラウザーなど、RAM を多く使う他のアプリを閉じるか、小さいモデル（Q2_0 または IQ2_XS）を選んでください。
- **ポート 8080 が使用中と表示される。** Strata がすでに起動しています。起動済みのウィンドウを探してください。

その他の問題と対処方法は、[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)をご覧ください。解決しない場合は
[Issue](https://github.com/Niko1221/Strata/issues) を作成し、Strata フォルダー内の `strata-<model>.log` を添付してください。
セキュリティ上の問題は、[SECURITY.md](SECURITY.md)の手順に従って非公開で報告してください。

## しくみ

この規模のモデルは通常、数百ギガバイトの GPU メモリを搭載したサーバーで動かします。一方、手元のグラフィックカードのメモリは
12-24 GB 程度です。Strata は **PC 全体で処理を分担する** ことで、こうした環境でもモデルを動かせるようにしています。
台所で、よく使うものは調理台に置き、それ以外は食品庫にしまっておくような仕組みです。

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="24,576 個のエキスパートのうち、よく使うものをグラフィックカードに配置。全エキスパートは RAM に、ルックアップテーブルは SSD に保持"></p>

- **モデルは「エキスパート」と呼ばれる 24,576 個の小さな処理単位から構成されています。** 1 語の生成に必要なのは、このうち 10 個だけです。
- **グラフィックカード** に、使用頻度の高い数千個のエキスパートを置きます。**RAM** にはすべてのエキスパートを保持し、
  **CPU** が残りの処理を並行して担当します。**SSD** には大きなルックアップテーブルを保存します。

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="小さな補助モデルが次の数語を予測し、大きなモデルがまとめて検証。正しい予測だけを採用する"></p>

- **予測してから検証：** 小さな補助モデルが次の数語を予測し、大きなモデルがまとめて検証します。
  回答を変えずに、生成速度を 1.6-1.8 倍に高められます。
- **長いテキストはまとめて読み込みます**（一度に最大 8,192 トークン）。読み込み速度は毎秒 1,000 トークンを超えます。

仕組みについては、[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)で詳しく説明しています。各処理の詳細や測定値は、
[技術資料](docs/DETAILS.md#how-it-works)と[論文](docs/paper/Strata-Paper.pdf)をご覧ください。

## クレジットとライセンス

使用しているモデルは、Qwen チームが開発した [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) です。
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)、UkisAI（Swift 1.5）、Unsloth が提供する圧縮版を利用しています。
Strata は [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) の一部も使用しています。クレジットの一覧は
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits)をご覧ください。Strata は [MIT License](LICENSE) で公開されていますが、一部の構成要素と
各モデルには、それぞれ別のライセンスが適用されます（[ライセンスの詳細](docs/HOW_IT_WORKS.md#license)）。

## Strata を支援する

Strata は無料で使えるオープンソースソフトウェアです。役に立ったら、開発へのご支援をお願いします。

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>
