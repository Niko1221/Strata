# Install Hakua / Strata on Windows

## 0.6.0 candidate and local dense GGUF

The reviewed source candidate adds the Windows CUDA 13 / Blackwell Qwen3.8-27B-family native trial. Installation and public delivery of 0.6.0 remain pending their receipts; the installed and published stable release remains 0.5.9. After a component covering `strata-q38.exe` is delivered, choose Use a local GGUF, select the original checkpoint, check it, then Prepare and load. A previous component without this helper reports a missing-engine error. Do not copy a development executable into the installed application.

The combined native path supports PrismML 0.4.1 PQ2_0 weights and Spin8 or F16-K/Turbo-V cache choices. It preserves the original model file. The installer, native component and tar.7z do not contain complete model weights. Auto KV preserves the requested context and fails if no supported placement fits the reserves. Start with the measured short-context scope; explicit 98,304/131,072 capacity is not evidence that full input is practical. Read [native choices and measured limits](QWEN38_DENSE_NATIVE.md). Existing Hermes memory, workflow learning, tool/key/approval choices and separately installed runtime selections remain preserved.

## 0.5.9 workflow safety update

The published coordinated Windows release is 0.5.9 for app, Terminal, Control and
native engine. Use its [standard setup EXE](https://github.com/zapabob/Strata/releases/download/v0.5.9/Strata-0.5.9-windows-x64-setup.exe)
or [tar.7z bundle](https://github.com/zapabob/Strata/releases/download/v0.5.9/Strata-0.5.9-windows-x64.tar.7z)
after confirming the [delivery record](../_docs/2026-11-13_stable-059-workflow-safety_Codex.md).
Formal v0.5.9 is GitHub Latest and the mutable latest alias. All seventeen public files matched anonymous download checks. The delivery record distinguishes real-model checks from fixture-model checks.
The standard per-user overwrite, backup and rollback instructions below apply.
RustXNG stays at 0.31.1 Full; existing provider and approval choices are retained.

Windows local GGUF loading reserves 5 GiB of available RAM and, for fitting fully GPU-resident models, additional loading and mapped-page headroom. The hard floor stays 3 GiB. The host
monitors its own engine throughout loading and generation and stops it early
under physical memory pressure. A stopped model may need a smaller batch or
memory mapping disabled before reloading. Context is never silently reduced.
Research permits only search and page reads, then answers without tools.
Inventory CSV audits use calculated comparisons; passed means successful.

## Historical stable 0.5.8

Use the [0.5.8 setup EXE](https://github.com/zapabob/Strata/releases/download/v0.5.8/Strata-0.5.8-windows-x64-setup.exe)
or [complete tar.7z](https://github.com/zapabob/Strata/releases/download/v0.5.8/Strata-0.5.8-windows-x64.tar.7z),
and compare [SHA256SUMS.txt](https://github.com/zapabob/Strata/releases/download/v0.5.8/SHA256SUMS.txt).
At its publication, this formal fork release was GitHub Latest and the mutable latest alias. It has been superseded by 0.5.9. App,
Terminal, Control and freshly built native engine are all 0.5.8. RustXNG stays
0.31.1 Full; third-party packages keep their independent versions.

Finish requests, unload models, back up app data and quit Strata including its
tray icon. Run the EXE in the existing per-user installation folder, usually
`%LOCALAPPDATA%\Programs\Strata`. Models, chats and memory stay in separate
data locations. Do not delete model folders to update or roll back.

Extract 7z first, then tar, and verify the internal BUNDLE-SHA256SUMS.txt. The
bundle contains the identical Tauri NSIS installer, signed native 0.5.8 CUDA
component and Control 0.5.8. Import the engine tar.gz with **I already have the
file**, keeping it compressed. Choose a compatible model or local GGUF and Load.
No system Python, Node or compiler is needed. Models start unloaded.

In Skills, learning and automatic application are independent options. Existing
local choices were preserved; both remain off for new profiles. `/btw` uses a
real tool-free subagent; `/deepresearch` gathers sources. RustXNG Full remains in
Settings → Tools → Web search. [Skills and undo](WORKFLOW_SKILLS.md).

The app installer has no Windows Authenticode signature; engine/Control feed
signatures are independent. All eight exact-source CI workflows passed. Native
measurement covers twenty-one numerical cases and sixteen short model requests
at 4096 capacity on RTX 5060 Ti 16 GB / 32 GB RAM. Full-window/general model
quality and GPU concurrency remain unqualified; `releaseQualified=false`.
See the [delivery record](../_docs/2026-11-11_stable-058-context-ci_Codex.md) and [measured limits](HAKUA_RELEASE.md).
Older releases and backups remain available for rollback; their original
instructions and measurements follow below.

## Historical stable 0.5.7

Use the [0.5.7 setup EXE](https://github.com/zapabob/Strata/releases/download/v0.5.7/Strata-0.5.7-windows-x64-setup.exe)
or [complete tar.7z](https://github.com/zapabob/Strata/releases/download/v0.5.7/Strata-0.5.7-windows-x64.tar.7z),
and compare [SHA256SUMS.txt](https://github.com/zapabob/Strata/releases/download/v0.5.7/SHA256SUMS.txt).
This is the stable zapabob fork release. It is selected by GitHub Latest and the
mutable latest alias; older releases and immutable tags remain available.

Finish requests, unload models, back up app data and quit Strata, including its
tray icon. Run the new EXE using the existing per-user installation folder.
The usual location is `%LOCALAPPDATA%\Programs\Strata`. Models, chats and memory
have separate data locations. Do not delete model folders to update or roll back.

For the bundle, extract 7z first and then tar; check BUNDLE-SHA256SUMS.txt inside.
It contains the identical EXE, native 0.5.5 signed CUDA component and Control
0.5.6. Import the engine tar.gz using **I already have the file**, keeping it
compressed. Select a compatible model or local GGUF and click **Load**. Models
start unloaded. No system Python, Node or compiler installation is needed.

In **Skills**, enable workflow learning and automatic application separately.
Learned documents are private and editable, with retained revisions and undo.
`/btw` uses a real tool-free subagent, with its own Stop; `/deepresearch` gathers
sources. RustXNG Full remains selectable in **Settings → Tools → Web search**.
[Settings, data and undo](WORKFLOW_SKILLS.md) · [Search](BROWSER_SEARCH.md).

The EXE has no Windows code signature; engine/feed component signatures are
separate. The native component retains its earlier hardware measurements and
`releaseQualified=false`. Full-window generation, general answer quality and
GPU concurrency are not qualified by this app release. Read the
[stable delivery record](../_docs/2026-11-11_stable-057-release_Codex.md) for actual
checks and the [native measured limits](HAKUA_RELEASE.md). Rollback uses a retained
installer/component plus the pre-update profile backup.

## Historical preview instructions and measurements

0.5.7 adds real /btw subagents, private workflow skill learning and automatic skill application. [Settings, data and undo](WORKFLOW_SKILLS.md).


Stable 0.5.7 adds `/btw`, `/deepresearch` and bundled RustXNG Full.
Use the [new installer](https://github.com/zapabob/Strata/releases/download/v0.5.7/Strata-0.5.7-windows-x64-setup.exe)
or [tar.7z bundle](https://github.com/zapabob/Strata/releases/download/v0.5.7/Strata-0.5.7-windows-x64.tar.7z).
The [new delivery record](../_docs/2026-11-11_stable-057-release_Codex.md)
separates actual publication/install checks from the previous preview's evidence.
The native 0.5.5 engine and Control 0.5.6 are retained; the instructions below
still apply to engine import, data preservation and rollback.

[Quick installation entry point / かんたん導入](../INSTALL.md) ·
[Web search and browser use / Web検索・ブラウザー操作](BROWSER_SEARCH.md)

The [0.5.5-local.1 preview](https://github.com/zapabob/Strata/releases/tag/v0.5.5-local.1)
integrates official 0.2.13. It includes Control 0.5.6 and uses native 0.5.5,
PrismML 0.4.1, TurboQuant 0.4.0 and Hakua-memory 0.3.5. Checks used Windows x64,
CUDA sm_120, RTX 5060 Ti 16 GB,32GB RAM and driver621.14. **The unchanged CUDA
resource guard passes with zero violations; releaseQualified=false retains the
broader quality limits.** Installed Q36/Prism/agent/restart/bulk-delete checks passed.
The historical054 Prism timeout cause remains unconfirmed. Full-window/general/
low-precision quality, public search sites and positive Qwen3.8 projection remain
unqualified. See [the exact measured scope](HAKUA_RELEASE.md).

## Install or update

Download the [Windows setup EXE](https://github.com/zapabob/Strata/releases/download/v0.5.5-local.1/Strata-0.5.5-local.1-windows-x64-setup.exe)
and compare it with [SHA256SUMS.txt](https://github.com/zapabob/Strata/releases/download/v0.5.5-local.1/SHA256SUMS.txt).
The desktop installer has no Windows code signature. Component signatures and checksums
are separate checks; they are not Windows executable signing.

Unload models, finish active requests and quit Strata, including its tray icon, before
updating. Back up the app data first. Run one installer at a time. Do not remove the
installation directory while setup is running. The installer replaces its managed
application files and preserves unrelated files in the installation root.

The usual per-user location is `%LOCALAPPDATA%\Programs\Strata` on C:. A previous
installation can retain another drive. Choose the intended folder in the installer.
Models, chats and memory have their own data locations; changing the application
folder does not move them. Leave existing model files where they are and select
that model folder in Strata. System Node, Python, pip and compilers are not required.

GPU runtimes are separate from the NSIS app, following upstream's Windows layout.
Download this preview's [signed engine tar.gz](https://github.com/zapabob/Strata/releases/download/v0.5.5-local.1/strata-engine-0.5.5-windows-x86_64-cuda-cuda13.tar.gz).
On the preparation screen choose **I already have the file** and select the tar.gz
without extracting it. The app checks the fork signature and file hashes before
activation. Use this preview's file to obtain measured native 0.5.5: the default
stable feed and GitHub Latest still select 0.5.1. Model loading is manual.

For one download, use [Strata-0.5.5-local.1-windows-x64.tar.7z](https://github.com/zapabob/Strata/releases/download/v0.5.5-local.1/Strata-0.5.5-local.1-windows-x64.tar.7z).
Extract the 7z, then the tar. The folder contains the same setup EXE, signed CUDA
component, Control ZIP, feed, install scripts and INSTALL.md. Check the outer
SHA256SUMS.txt before extraction, and BUNDLE-SHA256SUMS.txt inside. Models and
personal data are not included. The app and component can be installed from these
files; WebView2 first-time setup may still require internet access if it is absent.

The 7z uses Copy mode: the contained EXE/gzip/ZIP files are already compressed.
It is a real tar inside a 7z container. The small licensed, default-off control
vector in Control is included; full model weights and personal profiles are not.

## Stable version, preview feeds and rollback

[Stable 0.5.1](https://github.com/zapabob/Strata/releases/tag/v0.5.1) remains available
with its own installer/component/checksums. Keep a pre-update profile backup.
Installed component rollback selects a retained component; it does not require
fetching an older feed. A desktop downgrade uses the older installer and its
documented compatibility limits; do not delete model folders to downgrade.

The optional Terminal scripts in the bundle install Control 0.5.6. To opt into
the preview feed for this PowerShell session:

```powershell
$env:STRATA_FEED='https://github.com/zapabob/Strata/releases/download/v0.5.5-local.1'
irm "$env:STRATA_FEED/install.ps1" | iex
```

`STRATA_FEED` here is session-scoped; set it explicitly in later sessions when
using preview updates. The signed feed uses the existing stable *transport*
channel for client compatibility at an immutable prerelease URL. It is not
GitHub Latest and does not mean releaseQualified=true. Once its newer sequence
is accepted in a component folder, an older stable feed is refused by the
anti-replay guard. Retain that guard; use installed-component rollback or a
separate data/component folder to keep stable and preview update state separate.

## Existing models and caches

Use **Use a local GGUF** for a file already on the PC. One shard is enough to select
a split set; all required shards are checked. Catalog setup also searches the chosen
model directory's subfolders for one matching valid set. If more than one matches,
choose the intended file. It does not silently select a different fine-tune.

A native model can need a one-time expert-pack conversion. The app displays completed
layers when the packer reports them. Resume reuses the matching checkpoint and counts
space already allocated to its temporary file. Existing GGUFs and expert packs stay
on disk; normal unload and restart do not delete them.

Qwen3.6 private conversation reuse is negotiated with the running engine. Within
one authenticated conversation, a completed prefix can be reused. Another conversation,
missing identity, cancellation, invalid requests and five minutes of idle time clear
that private state. This does not delete the model or its expert pack. The first
implementation accepts one active request; parallel private Qwen3.6 requests are refused.
The Host no longer rejects Qwen3.6 before startup because its private-cache digest
is absent from a list. The server checks the engine's actual capability before it
accepts requests. An engine without that capability needs the matching update.

The Qwen3.6 mapped expert pack is opt-in and requires an allowlisted executable.
It avoids the legacy full host expert arena, but the operating system may still page
mapped data. The experimental `STRATA_Q36_PIN_PIPELINE=1` and
`STRATA_Q36_STREAM_RESIDENT=1` environment switches retain the model's numeric formats.
They are off by default. The release record describes their bounded comparison; it is
not a guarantee of speed on another model, context length or PC.

## Chat, tools and memory

For a supported PQ2_0 GGUF, select **PrismML**. For TurboQuant, choose **TurboQuant**,
K=`FP16` and V=`TURBO2`, `TURBO3` or `TURBO4`, then reload. The release keeps the
quality-related refusals for unsupported K-cache choices. See [backend limits](LOCAL_LLAMA_BACKENDS.md).

Choose the loaded local model in the chat. Web search needs network access and a model
that emits a supported tool call. A tool request, its returned result and the assistant's
final answer are separate parts of the same turn. Stop cancels the active turn.
The app's request adapter keeps Qwen3.6 cache ownership separate for each chat.

Hakua memory is optional and remains off until enabled. Open it from the sidebar after
loading the model. Reference/save consent is selected separately for each message.
Additional local runtimes use the app's local profile key automatically. First enabling
memory can restart the owned server with authentication and shared caches disabled;
finish active requests first. See the [memory guide](../serve/MEMORY_HARNESS.md).

Native image input and the separate DeepSeek native engine are not included in this
Windows release. Other GPUs, HIP, SYCL, Linux and Metal need their own builds and tests.

## 日本語の導入手順

プレビュー0.5.5-local.1は短いQwen3.6・Prism要求、エージェント応答、会話一括削除を実installedで確認しています。
CUDA資源違反は0件です。旧054のPrism timeoutは原因未確定の記録として保持し、releaseQualified=falseです。
全文入力・一般品質・低精度品質を保証する結果ではありません。
起動時にはモデルを自動読み込みせず、利用者が読み込みを開始します。

`0.5.5-local.1`のsetup EXEとチェックサムを取得します。更新前にモデルをアンロードし、
トレイのStrataも終了してから、一度だけインストーラーを実行してください。
Cドライブに入れる場合は、導入先を`%LOCALAPPDATA%\Programs\Strata`に指定します。
以前Hドライブを選んだ環境では、その場所が引き継がれることがあります。
モデル・会話・記憶の保存先は別なので、モデルを移したり消したりする必要はありません。

専用Node・Pythonは同梱されています。GPUエンジンは別配布です。同じプレビューから
`strata-engine-0.5.5-windows-x86_64-cuda-cuda13.tar.gz`を取得し、圧縮したまま
準備画面で読み込みます。署名と全ファイルのハッシュを
確認してから有効化します。旧コンポーネントはロールバック用に残します。

一括で取得する場合は`Strata-0.5.5-local.1-windows-x64.tar.7z`を使います。7z、tarの順で展開すると、
同じNSISインストーラー、署名付きエンジン、Control、配信情報、導入スクリプト、説明書が
入っています。モデルや個人データは含まれません。検証済みの範囲と既知の制限は
[リリース記録](HAKUA_RELEASE.md)で確認してください。

GitHub Latestと既定のstable配信は0.5.1です。プレビュー配信は明示URLで選びます。
新しいsequenceを受理した更新状態では古いstable配信を拒否します。保護を無効にせず、
保持したコンポーネントのロールバックか、別のデータ・コンポーネント保存先を利用します。

取得済みGGUFは「ローカルGGUFを使用」で選択します。分割GGUFは一つを選ぶと一式を確認します。
モデルと変換済みパックは保持します。Qwen3.6の会話キャッシュは、認証された同じ会話で再利用し、
別の会話、キャンセル、エラー、識別情報のない要求、5分間の待機では消去します。
この消去でモデルファイルやパックが消えることはありません。

インストーラーと一時展開には数GBの空き容量が必要です。モデルの容量は別に確保してください。
公開状況、実測条件、未確認事項は[リリース記録](HAKUA_RELEASE.md)を参照してください。
