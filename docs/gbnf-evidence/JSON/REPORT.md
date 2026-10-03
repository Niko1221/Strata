# Native JSON and local Codex checkpoint, 2026-10-03

Real Windows Codex CLI 0.160.0 completed a read/edit/test task against Strata's
native Qwen3.8-Flash-Next Coder IQ1_M model on llm-49. Its automatic strict-schema
title request also completed with `{"title":"Fix add function in calc.py"}`.
Seven additional native HTTP cases passed, including typed SSE and a tool loop
whose external result was deliberately arbitrary text.

This historical checkpoint is followed by the
[30-schema/12-tool qualification](../schema-inventory/REPORT.md), which adds
strict function validation, native parser fixes and the complete captured inventory.

## Source and changed responsibilities

Branch `work/gbnf`, worktree `strata-native-gbnf`, tested source commit
`9576ab924b91123caa96d14c441010dcd3d5f38f`. The [receipt](receipt.json) pins the
client, catalog, native executable, settings and cleanup; the
[native source manifest](native-source-manifest.json) matches the binary's inputs.
The earlier [steering checkpoint](../../responses-evidence/steering/REPORT.md)
records real interruption, a permission update and successful subsequent output.

At the user's explicit request this extends the original handoff's JSON exclusion.
`serve/responses_json.py` validates the Responses `text.format` boundary and final
JSON. The existing native compiler accepts JSON Schema through the same XGrammar
matcher, token masks, service queue, speculative state and cancellation/draining.
The existing assembler serves both JSON and typed SSE. No localhost endpoint
translation, second inference engine, workflow framework or schema repair exists.
The native schema frame also carries a thinking-token ceiling; reasoning and
client-owned tools remain outside the final-answer schema.

The original raw-grammar work budgets were too small for bounded JSON strings
over this vocabulary. The 1–36-character Codex title needs more than 100 million
compile-work units, about 1.2 seconds and 5.2 MiB on this host. JSON compilation
now allows 150 million units while retaining the 2.5-second deadline and 16 MiB
compiled-size limit. Matching allows 64 million units per JSON sequence, including
discarded speculative work. Raw GBNF budgets are unchanged.
[Compile measurements](compile-budget-measurements.txt),
[matcher measurements](matcher-budget-measurements.txt) and the
[actual-vocabulary regression](actual-vocabulary-tests.txt) retain the evidence.
That regression accepts a 36-character title and rejects empty/37-character titles.

## Commands and results

```text
.venv/Scripts/python -m unittest -v serve.test_server serve.test_responses serve.test_security serve.test_lifecycle serve.test_grammar serve.test_grammar_scope serve.test_responses_json
ctest --test-dir <native-build> --output-on-failure -R 'grammar|serve_input'
<native-build>/grammar_native_test data/grammars/cases.json <model-pack>/tokenizer <vocab-audit-output>
python tools/responses_json_probe.py --base-url <authenticated-server>/v1 --model qwen3.8-flash-next --out <new-evidence-directory>
```

| Check | Result |
|---|---|
| Python service/API/lifecycle/schema tests | [231 passed](python-tests.txt). |
| Native grammar, input framing, CUDA selectors and speculation | [10 passed](native-tests.txt). |
| Actual Qwen vocabulary | 248320 tokens; existing 41-case grammar corpus and new title constraints passed. |
| Real Codex task | Four local shell calls read, edit and verify; three unchanged tests passed independently. |
| Real Codex title | Actual `text.format` with `strict:true`, required title, no extra keys and length 1–36 completed. |
| Native HTTP JSON | Five completed cases, one expected incomplete result and one expected HTTP 400 before SSE; all assertions passed. |
| Shutdown | Codex exited zero; owned server stopped; GPU process list was empty. |

[Actual Codex title request](codex-title-request.json) and
[response](codex-title-response.json), [tool loop](codex-tool-loop.json),
[turn result](turn-summary.json), [independent tests](independent-tests.txt), and
[unchanged-file receipt](independent-verifier.json) are included. Machine paths in
the title capture were sanitized. No API keys or decrypted replay tokens are included.

The [HTTP receipt](native-http/result.json) links by numeric prefix to complete
requests and JSON/SSE responses in its directory. It covers JSON object mode,
nested arrays/local references/enums/Unicode, concatenated stream-delta equality,
output exhaustion, remote-reference rejection, generated reasoning/function calls,
and a JSON answer after a mocked external tool returned ordinary non-JSON text.
The model output was real; only the external lookup result was mocked.

## Independent 4B comparison

Qwen3.5-4B with BF16 weights and FP16 K/V also produced valid strict JSON and
completed the same small read/edit/test task through an existing llama.cpp Chat
endpoint. [Pins/configuration/results](reference-4b/receipt.json),
[requests and results](reference-4b/requests-and-results.json) and
[independent tests](reference-4b/independent-tests.txt) are retained.

Its direct Codex Responses attempt failed before generation: the reference adapter
skipped namespaced tools and its template rejected system-message ordering. That
endpoint also ignored `text.format`, while its Chat schema interface worked.
[Exact errors](reference-4b/codex-server-errors.txt) establish an adapter problem,
not a quantization symptom. This cross-model/cross-engine comparison does not
isolate the quality effect of quantization and is not native Strata qualification.

The native 4B [loader result](reference-4b/native-strata-guard.txt) is separate.
Strata's `NativeDense` loads dense projections within its supported architecture;
skipping MoE alone would not supply qwen35 residual/norm paths, ordinary attention,
SiLU gating, a dense FFN, tied output head or BF16 tensor dispatch. No dense-model
port was added to this Responses/JSON change.

## Limits and next gate

See [JSON output](../../JSON_OUTPUT.md) and [local Codex setup](../../CODEX_LOCAL.md).
JSON Schema draft 2020-12 validation is mandatory; missing dependencies or unknown
keywords fail admission. Native compilation may reject unrepresentable or expensive
schemas. A final schema-validation failure is `response.failed`, not weaker success.
At this checkpoint, strict function-parameter schemas, Lark tools, hosted tools, images, retained
responses and long-session compaction remain separate unsupported capabilities.
This is one pinned client/model/profile qualification, not universal Codex support.

The next gate is review and publication of these local changes. Deployment still
opts in with `--experimental-responses`; install `requirements-json.txt` and use
the gbnf-v4 native build. Start a fresh local chat with the updated profile.
