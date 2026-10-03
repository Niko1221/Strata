# R3 — client-owned tools and reasoning

Status: passed for the documented adapter profile. Native model quality is not
certified by these CPU/MockEngine tests.

- Branch: `work/responses-api`.
- Worktree: `C:/Users/dflanag3/Documents/fleet/strata-responses-stateless`.
- Original-author base: `99f3dbd0b21d1401b3769e0c0d963913607f380b`.
- R2 parent: `53c4f9f6d0e1c121a387a2c5f2b47ba91b058c2f`.
- The full R3 implementation commit is recorded in the subsequent R4 phase report.

## Changed responsibilities

`serve/responses.py` validates non-strict function descriptions, flattens namespaces
only for the native template/parser, and restores separate namespace/name fields
on the wire. `call_id` binds client results; item IDs identify output items.
Strata does not execute the client's functions. Streamed argument strings remain
the canonical strings, without schema enforcement or post-stream reformatting.

The request-local assembler now emits actual reasoning text, generates requested
summaries through another bounded use of `Service.run()`, and returns authenticated
encrypted replay. These additions follow the user's explicit expansion of the
reasoning scope. Raw thinking is never relabeled as a summary; foreign encrypted
tokens are rejected. `serve/response_replay.py` handles the deployment key and the
maintained Fernet implementation. Only that key persists, not response records.

`serve/server.py` loads the key only when the feature is enabled and adds optional
reasoning-token accounting to the existing semantic service result. It does not
change native execution, MTP, the scheduler, or default endpoint behavior.
`prompt_cache_key` maps to the sole existing engine as a routing hint, without a
cache-isolation or retention claim. `client_metadata` is bounded diagnostic data.

Summary requests buffer later semantic output until the reasoning item finishes.
This was needed for the actual Codex client's sequential item handling. Summary
generation adds latency and prompt/generation work; both passes share the request's
output allowance and use existing FIFO/cancellation/draining.

## Commands and results

All commands ran in this worktree using its isolated `.venv/Scripts/python.exe`.

| Command | Result | Evidence |
|---|---|---|
| `python tools/responses_verify.py --out docs/responses-evidence/R3/tests serve.test_responses` | 44 passed | [contract log](tests/serve-test_responses.txt) |
| `python tools/responses_sdk_probe.py --out docs/responses-evidence/R3/sdk` | SDK 3.23.0 text, SSE, namespaced read/write/result loop, reasoning, summary, encrypted-only replay and strict wire validation passed | [receipt](sdk/sdk-probes.json), [actual exchanges](sdk/sdk-loop.json), [typed events](sdk/sdk-events.jsonl) |
| `python tools/responses_restart_probe.py --out docs/responses-evidence/R3/restart` | Two distinct server processes; supplied encrypted history restored using only the deployment key | [receipt](restart/restart-probe.json) |
| `python tools/responses_codex_probe.py --out docs/responses-evidence/R3/codex-workspace` | Real Codex 0.160.0, four requests, three successful client commands, disposable file edited and verified | [receipt](codex-workspace/codex-task.json), [client log](codex-workspace/codex-task.txt), [actual exchanges](codex-workspace/codex-exchanges.json) |
| `python tools/responses_verify.py --out docs/responses-evidence/R3/regressions` | 190 passed, five existing detokenizer skips | [results](regressions/test-results.json) |
| `python tools/responses_examples.py` | 21 request examples, 170 nested SDK type definitions, examples of every type/field, real encryption of synthetic data | [index](../../responses-examples/INDEX.txt) |
| `git diff --check` | Passed | Repeated before the implementation commit |

The SDK's high-level ParsedResponse serializer emits a generic-type warning with
the pinned Pydantic combination. It is retained in [sdk-warnings.txt](sdk/sdk-warnings.txt).
Independent validation of the actual HTTP response objects and typed events passes.
No SDK monkeypatch or custom client is used.

Earlier Codex attempts remain as explicitly unsuccessful evidence. `codex/` found
an item-order issue, fixed by finishing reasoning before following items.
`codex-ordered/` completed the protocol but had no configured Windows sandbox.
`codex-sandboxed/` and `codex-final/` exposed working-directory and private-temp-ACL
problems in the harness. The passing `codex-workspace/` uses the documented
`windows.sandbox="unelevated"`, explicit paths/workdir, and a normally created
disposable workspace. No system policy, account ACL or personal client config was
changed. External HTTP attempts by the CLI were rejected by the loopback proxy.

## Provenance and limitations

All model output in these tests is synthetic MockEngine script output. The SDK and
Codex HTTP requests/tool-result exchanges are real. The client, not Strata, edited
the test file. No GPU, llm-49, llm-60 or r730 was contacted during R3. The original
checkout and hardware branches remain untouched. No fetch, push, reset, stash or
rebase was performed.

Strict or omitted-strict tools, JSON output constraints, Lark/custom tools, hosted
tools, storage/previous-response IDs, background, compaction, media and WebSockets
remain excluded. There is no universal Codex compatibility claim. The actual client
reports fallback metadata for the local model name; the model is not impersonated.

The next gate is R4 acceptance and a full-commit pass receipt. GBNF must start from
that exact passing checkpoint, with independent build/runtime artifacts.
