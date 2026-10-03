# R1: stateless request boundary and assembler

Worktree `<WORKSPACE>/strata-responses-stateless`, branch `work/responses-api`.
Dependency: R0 commit `0fc5140d03830036c26fee028127bdb8ce81bd53` on original-author
main `99f3dbd0b21d1401b3769e0c0d963913607f380b`.

The request-local assembler owns facts, output buffers and terminal transitions.
The HTTP handler validates JSON and reuses existing auth, CORS, disconnect watcher
and monitor. `Service.run()` has opt-in semantic lifecycle notifications;
existing callers retain the original iterator contract. No native files, MTP
configuration, response store or tool executor changed.

Commands and evidence:

```sh
python tools/responses_verify.py --out docs/responses-evidence/R1/tests serve.test_responses
python tools/responses_verify.py --out docs/responses-evidence/R1/regressions serve.test_security serve.test_lifecycle serve.test_server
```

[15 contract tests](tests/serve-test_responses.txt) pass: defaults/CLI/config
activation, unauthorized and foreign-origin requests, manual history, preserved
instruction priority, no retained context, malformed JSON, unsupported capabilities,
truthful completion/length statuses, stable final snapshots and immutable terminals.
[Regression results](regressions/test-results.json) cover the existing service,
Chat/Anthropic adapters, security and lifecycle. Environment: R0's pinned Python
environment, CPU MockEngine with synthetic scripted output, no native/GPU run.

The feature is default off. See [enablement and profile](../../RESPONSES.md).
Streaming is explicitly rejected at this checkpoint; function tools await R3.
R0's Codex blocker remains. Next gate: typed SSE from the same execution, exact
reassembly, failures and disconnect/drain behavior in R2.
