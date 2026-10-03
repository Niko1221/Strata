# R4 — Responses acceptance and freeze

Result: pass for the documented stateless HTTP/SSE profile and pinned clients.
This is a CPU/MockEngine protocol checkpoint, not a native-model quality or grammar
checkpoint. All synthetic model output is labeled in the probes and receipts.

- Branch: `work/responses-api`.
- Worktree: `C:/Users/dflanag3/Documents/fleet/strata-responses-stateless`.
- Original-author base: `99f3dbd0b21d1401b3769e0c0d963913607f380b`.
- R0 commit: `0fc5140d03830036c26fee028127bdb8ce81bd53`.
- R1 commit: `88a63bcb95bd2d4d08b1bbb170c395c00620f5cf`.
- R2 commit: `53c4f9f6d0e1c121a387a2c5f2b47ba91b058c2f`.
- R3 commit: `5cb57467d79fb30d3b53ff249fbf2baef1e3b5c5`.
- The R4 implementation/evidence commit is recorded after committing in the
  companion `R4-receipt.json`, without rewriting the tested commit.

## Changed responsibility and revisions

R4 freezes the adapter implemented in R1–R3 and its expanded reasoning/tool
profile. The deployment replay key now comes exclusively from
`STRATA_RESPONSES_REPLAY_KEY`, following the handoff's environment-only key rule.
This supersedes R3's automatic key file. Missing/malformed keys fail startup;
the flag-off path imports no crypto module and creates no Responses runtime.
Two restarted processes reuse the environment secret, with no persisted files.

The final regression run exposed Windows connection resets on the existing
`/load` and `/unload` endpoints, which closed with unread request bodies. The
same full security suite reproduced the error with server source loaded directly
from the selected upstream Git commit. See [baseline reproduction](upstream-security-full.txt),
[initial run](tests/serve-test_security.txt), and [recheck](security-recheck/serve-test_security.txt).
The fix consumes their unused bodies before replying, bounded to 64 KiB and two
seconds. Existing auth/Origin/content-type decisions stay in place. A new test
verifies that incomplete bodies cannot trigger the operation. No test was weakened.

## Commands, results and evidence

Commands use this worktree's isolated Python 3.13 environment. SDK and crypto
versions are pinned in [R3/python-lock.txt](../R3/python-lock.txt); the server's
optional crypto requirements are in the repository root.

| Check | Command / result | Evidence |
|---|---|---|
| Full contracts/regressions | `python tools/responses_verify.py --out docs/responses-evidence/R4/final-tests serve.test_security serve.test_responses serve.test_server serve.test_lifecycle serve.test_structured serve.test_monitor serve.test_mcp serve.test_detok` — 235 passed, five existing detokenizer skips | [results](final-tests/test-results.json) |
| Final environment-key implementation | `python tools/responses_verify.py --out docs/responses-evidence/R4/replay-key-tests serve.test_responses` — all 44 passed | [log](replay-key-tests/serve-test_responses.txt) |
| Real local Codex task | `python tools/responses_codex_probe.py --out docs/responses-evidence/R4/codex` — four requests; client read, edited and verified the disposable file; three successful tool results | [receipt](codex/codex-task.json), [raw client log](codex/codex-task.txt), [actual exchanges](codex/codex-exchanges.json), [profile](codex/codex-profile.toml) |
| Official SDK | `python tools/responses_sdk_probe.py --out docs/responses-evidence/R4/sdk` — JSON, typed SSE, SDK assembly, namespaced tool/result loop, visible reasoning, real summary pass, authenticated replay, strict response/event validation | [receipt](sdk/sdk-probes.json), [exchanges](sdk/sdk-loop.json), [events](sdk/sdk-events.jsonl) |
| Real process restart | `python tools/responses_restart_probe.py --out docs/responses-evidence/R4/restart` — new process, encrypted-only history restored, zero persisted files | [receipt](restart/restart-probe.json), [server log](restart/restart-server-log.txt) |
| Source and scope audit | Native source/build files unchanged from the original-author base | [audit](source-audit.json) |
| Acceptance inventory | All R0–R4 cases mapped to concrete evidence | [case map](acceptance.tsv) |

The SDK's high-level generic ParsedResponse serializer warning is retained in
[sdk-warnings.txt](sdk/sdk-warnings.txt). Strict validation of actual wire objects
and events passes. It was not replaced with permissive validation.

Codex is `codex-cli 0.160.0`, SHA-256
`fdda5fa3cf3fb3d000b876720742857676293e4315e4b045fae6f8bd7e866d1d`.
The Python SDK is `openai==3.23.0`. The Windows test profile uses the documented
native unelevated sandbox, an isolated credential-free client home and a normal
disposable workspace. External HTTP attempts are blocked by its loopback proxy.
The local model name uses Codex's reported fallback metadata, not hosted-model
impersonation. The server does not execute the client's functions.

## Limits and next gate

The tests use existing MockEngine and ByteTokenizer. No native engine, GPU,
native speculative mode, grammar implementation or throughput was certified.
The native source audit proves that R did not alter MTP/serving code; CPU legacy
tests cover the existing Chat Completions, Anthropic and service paths. Five
optional detokenizer tests remain skipped, not counted as passes.

Strict/omitted-strict tools, JSON output constraints, Lark/custom tools, hosted
tools, storage/background, compaction, media and WebSockets remain excluded.
Larger or differently configured Codex sessions may request these and receive
explicit errors. The user-facing enablement and exact subset are documented in
[RESPONSES.md](../../RESPONSES.md); all request/type examples are tracked files.

After the full R4 commit and companion receipt exist, create `work/gbnf` at
`C:/Users/dflanag3/Documents/fleet/strata-native-gbnf` from exactly that checkpoint.
G0 must establish persistent non-speculative serving before grammar enforcement.
G6 remains deferred. No fetch, push, reset, stash, rebase or PR publication occurred.
