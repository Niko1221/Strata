# 2026-11-23 Sakura-Inkling-2-Mini 65K–98K コンテキスト + Hermes Agent 統合ログ (Antigravity)

## 1. 概要
- `strata-inkling2.exe`（ELT `--loop-exit-tol` 実装込み）を VS18 Insiders + Ninja (`-j 5`) で差分リビルド。
- `Sakura-Inkling-2-Mini-Uncensored-Q4_K_M.gguf` を RTX 5060 Ti (16 GB VRAM) 上で **98,304 (96K) コンテキスト**にて実測検証し、Windows WDDM VRAM 過負荷（`1.77 t/s`）を `--expert-vram 7.8` で完全解消（**`22.44 t/s`**, **12.7x 高速化**, 32/32 トークン完全一致）。
- `terminal/serve/server.py` および `terminal/serve/frontend.py` を拡張し、GGUF 埋め込みトークナイザ／チャットテンプレート（`{% generation %}` タグ対応）と `InklingOutputParser`（ツール呼び出し・思考チャンネル対応）を接続。
- Hermes Agent (`C:\Users\downl\.hermes\config.yaml`) に `inkling2` プロバイダ（`http://127.0.0.1:8092/v1`, `context_length: 98304`）を追加し、65K / 98K 用設定ファイルと起動スクリプトを整備。

## 2. 変更ファイル
- `terminal/serve/server.py`
  - `engine_args()`: `inkling_kind.is_inkling(cfg)` 時に `inkling_kind.engine_args(...)` へディスパッチ。
  - `main()`: `arch == "inkling2_loop"` かつ `--model <*.gguf>` 指定時、`vocab.json` ディレクトリ不要で `dense_tokenizer(model_file)` から GGUF 埋め込みトークナイザ・チャットテンプレートを読み込み。
- `terminal/serve/frontend.py`
  - `ChatTemplate.__init__()`: HuggingFace `{% generation %}` / `{% endgeneration %}` タグの自動除去と `_is_inkling` 判定を追加。
  - `ChatTemplate.render()` / `_detect_caps()`: `inkling_dialect.prepare_messages()` / `template_kwargs()` の適用および `supports_tools=True` / `supports_tool_calls=True` / `supports_parallel_tool_calls=True` 返却を追加。
  - `InklingOutputParser`: `InklingDialect.feed_channels()` を `JsonOutputParser` へ橋渡しし、`parser_for_template()` から返却。
- `configs/strata-sakura-inkling2-98k.json` (新規)
- `configs/strata-sakura-inkling2-65k.json` (新規)
- `scripts/start-sakura-inkling2-hermes.ps1` (新規)
- `C:\Users\downl\.hermes\config.yaml` (`providers.inkling2` 追加、既存 `amd` / `strata` / `sarashina` は維持)

## 3. 実測・検証結果
### ビルド
- `cmake --build H:/Strata/build/engine-cuda135 --target strata-inkling2 -j 5` (VS18 Insiders `vcvars64.bat`): **成功 (exit 0)**

### RTX 5060 Ti (16 GB / CUDA 13.5.37) 実機ベンチマーク (`Sakura-Inkling-2-Mini-Uncensored-Q4_K_M.gguf`)
| 条件 | コンテキスト | VRAM 配置 | Prefill (16 tok) | Decode (31 tok) | 備考 |
| :--- | ---: | :--- | ---: | ---: | :--- |
| 自動リザーブ (2048 MiB) | `98,304` (96K) | `1872/3072` (`9.07 GiB`, `tier1=128`, `tier2=28`) | `5084.9 ms` (`3.15 t/s`) | `17518.4 ms` (`1.77 t/s`) | WDDM 共有メモリへのスワップ発生 |
| **`--expert-vram 7.8`** | **`98,304` (96K)** | **`1598/3072` (`7.67 GiB`, `tier1=128`, `tier2=5..6`)** | **`1339.3 ms` (`11.95 t/s`)** | **`1381.5 ms` (`22.44 t/s`)** | **12.7x 高速化** / 32/32 完全一致 / GPU部 `516.9 ms` |

### 単体・統合テスト
- `terminal/serve/test_inkling_dialect.py`: **8/8 PASS** (`0.049s`)
- `ChatTemplate` + `parser_for_template` on `Sakura-Inkling-2-Mini-Uncensored-Q4_K_M.gguf`: **`InklingOutputParser` 検出 & テンプレート描画成功**
- `terminal/serve/server.py` (`--lazy-load`, `configs/strata-sakura-inkling2-98k.json`): `/health` および `/v1/models` 応答確認

## 4. 残余リスク・推奨アクション
- `H:` ドライブからの 17.5 GB 初回読み込みに約 3〜4 分かかるため、`start-sakura-inkling2-hermes.ps1` で常駐起動して使う運用を推奨（常駐後はプレフィックス KV キャッシュ再利用が効く）。
- 65K–98K 実長プロンプトでの長時間会話・ツール呼び出し E2E および ELT (`--loop-exit-tol`) の実機スイープは未実施。必要に応じて `-LoopExitTol 0.15` 等で段階検証を推奨。
