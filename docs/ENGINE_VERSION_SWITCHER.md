# Engine 0.1.52 ⇄ 最新 の切り替え（コミュニティ発ガイド）

Strata を使った環境の中には、最新リリースのエンジンがビデオドライバ要件で動かないものがあります。
その場合、**公式リリースの署名済み tarball からエンジンを選んで 0.1.52 ⇄ 最新 を安全に切り替えるツール**をコミュニティ側で公開しています。

- ガイド・スクリプト一式: https://github.com/KONOAION/strata-engine-switcher
- セットアップは **AGENT-SETUP-ja.md をエージェントに渡すだけ**で完了します（英語: AGENT-SETUP-en.md / 中国語: AGENT-SETUP-zh.md）。

要点は3つ:

1. GitHub Release の公式 tarball（v0.2.7 / v0.2.9）からエンジンを選ぶ。tarball 自体はこのリポジトリには置きません。
2. スクリプトが `state.json` に `engine.pinned` / `engine.skipped` を書き込み、Strata は sha256 検証済みのエンジンだけを起動します。
3. `engine-status.ps1` / `engine-latest.ps1` でいつでも最新に戻せます（Windows のみ）。

動作確認済み環境: **Windows 11 / NVIDIA RTX 3080 / driver 621.01 / CUDA 13.x**。
実行スクリプトは作者の環境でのみ動作確認しており、公開ドキュメントは readme のプロンプトをコピーして実行した結果に基づいています。バグは上のリポジトリの issue で報告します。

ライセンスは本家と同じ MIT です。
