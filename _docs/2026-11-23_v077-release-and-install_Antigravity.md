# 2026-11-23 Hakua v0.7.7 リリース公開・SHAリードバック・実機上書き完了ログ (Antigravity)

## 1. 概要
- `H:/Strata/worktrees/engine-067`（ブランチ `engine/0.6.7`）にて、引き継ぎ書の残作業 1〜7 をすべて完了。
- 新規 Q38 バイナリ（`strata-q38.exe` CUDA 13.5 SM120）を含む 0.6.7 エンジンコンポーネント `strata-engine-0.6.7-windows-x86_64-cuda-cuda13-r2.tar.gz` を含めた **GitHub Latest リリース `v0.7.7`** を公開し、全 4 アセットの SHA-256 リードバック完全一致および実機上書きインストールを完了。

## 2. バージョン更新・タグ・リリース公開
- コミット: `f520eca52` (`[Release] 0.7.7 - publish 0.6.7 CUDA 13.5 engine component with fresh Q38`, `engine/0.6.7`)
  - 変更対象: `VERSION`, `docs/data/release.json`, `app/desktop/package*.json`, `app/desktop/src-tauri/Cargo.toml`（`libloading` は変更せず維持）, `app/host/package*.json`, README 8 言語。
  - `py -3 tools/release_manifest.py --check`: **`release manifest: ok (0.7.7, engine 0.6.7)`**
- タグ・ブランチプッシュ: `origin/engine/0.6.7`, `v0.7.7`
- GitHub Latest URL: https://github.com/zapabob/Strata/releases/tag/v0.7.7

## 3. 公開アセット SHA-256 リードバック検証（4/4 `state=uploaded` 一致）
| アセット名 | サイズ (B) | GitHub Digest (SHA-256) | 状態 |
| :--- | ---: | :--- | :--- |
| `strata-engine-0.6.7-windows-x86_64-cuda-cuda13-r2.tar.gz` | `1,523,107,958` | `eb3bf3f005f0d1058ee107c07844d74351870514a3d7d806cb94e73020b3c6ce` | `uploaded` / 一致 |
| `strata-hy4-0.1.64-windows-x86_64-cuda-cuda13.tar.gz` | `426,088,482` | `fad3a1952f6fe6304eeddd57db76cb74f87b34a04b78f5fb83e895898f93ff91` | `uploaded` / 一致 |
| `Strata-0.7.5-windows-x64-setup.exe` | `127,142,902` | `734558cc38a0982921b133b0f8e3a79dee8b574879710685074bcc6265231eeb` | `uploaded` / 一致 |
| `Strata-0.7.5-windows-x64.tar.7z` | `127,145,634` | `b761f99860890af4f4e4f11ae4e2cf28e464b4e2257c26d9994254e61029d21c` | `uploaded` / 一致 |

## 4. 実機上書きインストール検証（ログ: `H:/Strata/evidence/v077-install.log`）
- アプリ配備先: `C:/Users/downl/AppData/Local/Programs/Strata`（`COPIED_APP_FILES=29702`）
- 0.6.7 エンジンコンポーネント配備先: `C:/Users/downl/AppData/Local/Strata/components/engine/0.6.7-windows-x86_64-cuda-cuda13`（`COPIED_COMP_FILES=155`、fresh `strata.exe` / `strata-inkling2.exe` / `strata-q38.exe` 含む）
- レジストリ更新: `HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Strata` (`DisplayVersion=0.7.7`, `REG_IMPORT=0`)
- ユーザー設定保持: `settings.json` SHA-256 = `6B7D7B113663611A207F9CC9C004385241766C7238D01B65CD86743548EA6CC0`（`SETTINGS_PRESERVED=True`）
- GUI 起動確認: `GUI_TITLE=ai.strata.desktop-siw EXITED=False`（6 秒時点の WebView2 初回初期化中は `RESPONDING=False`、確認後正常終了）

## 5. 注意点・残余リスク
- Windows Authenticode 署名なし、ed25519 署名検証なし、SHA-256 digest 検証のみ。`releaseQualified=false`。
- `strata-q38.exe` の自己申告バージョンは `strata-q38 own-native 0.6.3 CUDA13 SM120`（fork Q38 独自採番であり product `0.7.7` / upstream `0.1.65` とは独立）。
- `engine/q38/src/CMakeLists.txt` と `engine/data/` の worktree 固有修正は引き継ぎ方針どおり未コミットで保持し、`main` は一切変更していない。
- `H:/Strata/evidence` および `H:/Strata/build` の証拠ファイルはすべて保持済み。
