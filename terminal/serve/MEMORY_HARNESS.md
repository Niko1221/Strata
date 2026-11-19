# Hakua local memory

The Hakua Windows fork bundles the reviewed Hakua-memory 0.3.5 base wheel. Loading a native Strata model prepares one local memory profile. A request reads or saves memory only when the person explicitly consents to that action. This is a single-user desktop feature.

Real native-engine verification on 2026-10-30 saved a fictional preference, stopped the server and engine, started a fresh process and recalled that preference. Read-only consent left the memory records unchanged, and both servers exited normally. This verifies that scenario; multiuser isolation, general summary quality and the final installer still require their own checks. See [the release qualification record](../docs/HAKUA_RELEASE.md).

The 0.4.0 source candidate adds an explicit memory dialog for owned PrismML/PQ2_0 and TurboQuant runtimes. Finish active requests, load the verified backend, then open **Hakua memory** in the sidebar. This restarts that server with authentication and shared prompt/slot caches disabled. The dialog uses the local principal key automatically; select read and save independently before each message. Both choices reset after sending. These backends share the native route's local memory root, without writing the dialog conversation to ordinary chat history. Plain text is supported; tools and attachments are not. Only a normally completed answer can produce a consented factual summary; empty summaries and bare acknowledgments are rejected. Closing or stopping cancels an unfinished turn, but cannot undo a summary already committed. Packaging and final release qualification are still pending. See [local backends](../docs/LOCAL_LLAMA_BACKENDS.md).

## Use the desktop profile

Load a supported native Strata model, then open **Hakua memory** in the sidebar. The dialog supplies the local memory page and its profile key. The key belongs to this profile; the Host's ordinary chat key is different.

On the local page, select a memory checkbox to enter memory mode. Open **About → Settings** and paste the profile key. Before each message, select **read memory**, **save summary**, or both. The page clears those choices after sending. Closing the Host dialog clears its displayed key.

Reading and saving are independent. Ordinary chat does not implicitly grant either permission. No SOUL.md, USER.md or MEMORY.md file is imported automatically. The profile lives under the app's data directory, outside its program files; back up that directory before migrating or rolling back the app.

The app preserves explicit operator settings, including a disabled harness. Its automatic local profile applies to the native Strata route, not DeepSeek's separate engine. It disables the native prompt and conversation caches for memory requests and uses the same policy when tuning. No performance improvement is claimed for that policy.

## Configure a source installation

Source servers keep memory off unless their configuration enables it. Supply a dedicated absolute root and a unique ASCII key through an environment variable. The same key must already be accepted by the server's existing `api_key` configuration.

```json
{
  "hakua_harness": {
    "enabled": true,
    "users": [
      {
        "principal": "local",
        "key_env": "MY_LOCAL_MEMORY_KEY",
        "root": "H:/private-memory/local"
      }
    ]
  }
}
```

Managed launches through `strata runtime start` also need an accepted API key that is not assigned to a memory principal. Readiness uses that operator key for telemetry and launch attestation; it does not grant management access to a memory key. The desktop app creates its separate Host key automatically. A source configuration with only memory-principal keys cannot complete managed readiness.

The example creates no persona approvals. Protect the profile directory with the operating system's access controls. Do not point it at another application's database. Principal names, keys and roots come from this operator configuration; HTTP requests cannot select a principal or arbitrary path.

The native engine's effective arguments must contain `--prompt-cache 0` and `--conversation-cache-mib 0`. Slot save/restore is rejected. Unset `STRATA_BATCH_TRACE`; a nonempty value enables private protocol tracing in the Python server. The desktop launcher removes explicit off values before starting it, but direct source launches must use the correct environment.

Install the reviewed base wheel in the Python environment that runs `serve.server`. The bundled wheel is pinned by its SHA-256 in all four runtime locks. Embedding/RAG extras are not part of the basic package. Replacing the wheel requires source and writer-contract review, lock regeneration, fresh Control staging and import tests under each target interpreter.

## Request contract

Each request independently supplies:

```json
{
  "strata_memory": {
    "read": true,
    "store_summary": true,
    "turn_id": "a-unique-turn-id"
  }
}
```

The direct profile key authenticates the request. The distinct Host key is not mapped to a memory principal, and shared gateway traffic cannot carry `strata_memory`. Authentication, profile-root and cache-policy checks run before native model/GPU startup.

Read consent opens existing profile files without constructing a Hakua store. SQLite reads use immutable read-only connections and omit databases with outstanding WAL/SHM sidecars. A read-only operation must not create, migrate or repair a database.

Only successful normal text turns may produce a remembered summary. Tool continuations, cancelled/failed turns and requests without save consent do not invoke the writer. The per-profile `strata-turns.db` ledger records an attempt before calling Hakua. A retry with the same ID and content does not write again; the same ID with different content is refused. A crash after the attempt can leave persistence uncertain, and that attempt is not replayed. This is at-most-once writer invocation.

## Optional persona context

An operator may approve SOUL.md, USER.md and MEMORY.md with a `persona_sha256` mapping of filenames to reviewed lowercase SHA-256 digests. Unapproved, changed, linked or oversized files are omitted even when reading is allowed. The default limit is 8192 bytes per file; `persona_max_bytes` may be set from 1 through 32768.

Approval is checked against the bytes inserted into the prompt. Content changes require operator reapproval and configuration reload. MEMORY.md supplies approved prompt context; it is not a raw-turn storage destination. Hermes synchronisation, semantic graphs, Obsidian export and sleep remain separate operator tools.

## Verification

The Python harness and Host/browser contract tests use temporary profiles and scripted models. They test consent, identity, privacy gates, writer failures and cancellation. The additional native-engine test verifies one real save/restart/recall scenario. Neither establishes production multiuser isolation or general model quality.

```powershell
python -m unittest serve.test_memory_harness -v
node --test apps/host/tests/hakua-desktop.test.mjs apps/host/tests/native-gateway.test.mjs
node --test serve/web/test_memory_chat.cjs
node --test serve/web/test_memory_python_server.cjs
```

Browser tests need their documented local browser environment. A skipped test is not a passed runtime scenario. Keep multiuser production memory disabled until native cache switching, restored sessions and identity boundaries have been qualified with real traffic.

## 日本語

Hakua版では、対応するネイティブStrataモデルを読み込むと、アプリ専用のローカル記憶プロファイルを用意します。サイドバーのHakua memoryを開き、表示された専用キーを記憶ページのAbout → Settingsへ入力してください。通常チャットのキーとは異なります。

記憶の参照と要約の保存は、送信するたびに個別に選択します。同意は次の送信へ引き継ぎません。個人のSOUL.mdやMEMORY.mdは自動で取り込みません。PrismMLとTurboQuantの推論経路には、現在この記憶機能を接続していません。

2026年10月30日の実モデル試験では、架空の好みの保存、完全停止、新しいプロセスでの再起動、記憶の読み出し、参照だけでは記憶が増えないこと、正常終了を確認しました。複数ユーザーでの運用、要約品質全般、最終インストーラーの検証は別の確認事項です。
