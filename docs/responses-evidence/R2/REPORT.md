# R2: typed SSE and execution cleanup

Worktree `<WORKSPACE>/strata-responses-stateless`, branch `work/responses-api`.
Dependency: R1 commit `88a63bcb95bd2d4d08b1bbb170c395c00620f5cf`.

The handler now serializes the same assembler events already consumed by final
JSON. Stable item/content references, consecutive sequence numbers and exact text
concatenation are tested over real loopback HTTP. No Chat Completions chunks or
secondary endpoint are involved. The monitor observes lifecycle snapshots;
terminal capture happens once, not once per token. Cancellation is confirmed
after iterator cleanup and emits no invented `response.cancelled` event.

```sh
python tools/responses_verify.py --out docs/responses-evidence/R2/tests serve.test_responses
python <HANDOFF>/tools/audit_events.py docs/responses-evidence/R2/synthetic-text.jsonl --final docs/responses-evidence/R2/synthetic-text.final.json
```

[23 contract tests](tests/serve-test_responses.txt) pass. Coverage includes arbitrary
UTF-8 boundaries, empty output, output exhaustion, pre-header validation/loading
failure, post-header failure before/after text, missing service completion,
non-stream failure, exactly one terminal event, cancellation versus confirmed stop,
and real disconnects during queue/prefill/decode followed by a clean next request.

The [raw SSE](synthetic-text.sse), [events](synthetic-text.jsonl) and
[final snapshot](synthetic-text.final.json) are **synthetic MockEngine output**,
not captured model/Codex completions. The handoff's [partial core audit](core-event-audit.txt)
passes; it is additional evidence, not full API conformance. The main contract
tests independently compare final JSON with the HTTP stream.

Test isolation was tightened: the Responses HTTP fixture does not start hardware
telemetry, and the startup test replaces the server's clock reference instead of
patching the shared Python time module (which could interrupt telemetry threads).
This changes no production telemetry or native serving behavior.

Environment: R0 Python lock, CPU-only mock/scripted execution. No native binary or
MTP changes. Function calls and real supported-client loops are the R3 gate.
The pinned Codex profile remains blocked by its excluded representations.
