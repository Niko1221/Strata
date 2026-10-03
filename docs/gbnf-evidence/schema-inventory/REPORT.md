# JSON and Codex tool schema qualification, 2026-10-03

The 30 documented JSON schema examples and all 12 captured Codex 0.160.0 function
declarations passed native generation on llm-49. Each tool call was followed by a
clearly mocked external result and a schema-constrained JSON answer. The probes
never executed shell, image, agent or goal actions. They supplied known sample
arguments and answers, so these results qualify the protocol and serialization,
not autonomous task selection or every possible schema value.

The tested server source is `ab36ceed2453d9ff7cfff430467a5a2d8f307365` on
`work/gbnf`, worktree `strata-native-gbnf`. The [receipt](receipt.json) records the
native binary, client, profile, settings and cleanup. Hardware/model settings were
RTX 4090, Qwen3.8-Flash-Next Coder IQ1_M, 32768 context, int8 KV, ordinary MTP
`spec=4`, and suffix drafting disabled. Tool probes used medium reasoning and
temperature zero; the interactive Codex profile uses temperature 1.0.

## Changed responsibilities

- `serve/responses_json.py` validates complete strict function arguments and
  normalizes omitted strict settings where the object can be closed without
  discarding explicitly open/dynamic behavior. Validation is mandatory; a strict
  mismatch fails before the completed call event. Native argument token masks
  are not claimed. External tool results remain arbitrary text.
- `serve/frontend.py` interprets explicitly boolean native XML values `True` and
  `False` before producing canonical argument JSON. String and undeclared values
  keep their existing interpretation. No emitted JSON is rewritten.
- At this historical checkpoint, the Responses native tool view added an empty-call example for closed functions
  with no parameters. This reaches the installed model's own template while the
  client-visible declaration stays unchanged. It does not remove invented fields.
  Subsequent [paired prompt work](../../PROMPT_EXAMPLES.md) removes that implicit
  hint and records zero through three explicit examples separately.
- The pinned native dependency now includes the dynamic-object correction in
  `tools/prepare_xgrammar.py`. Without it, `patternProperties` could exclude valid
  `additionalProperties` when there were no named properties. Resource guards and
  final validation of the original schema remain active.

The backend pin is `xgrammar-0.2.8-strata-budget1-json1`; prepare a new dependency
directory and rebuild. [Exact dependency hashes](PIN.json),
[native source hashes and commit verification](native-source-manifest.json), and
[build output](native-build.txt) are retained. No inference scheduler, serving
framework, external store or tool executor was added.

## Commands and results

```text
.venv/Scripts/python -m unittest -v serve.test_server serve.test_responses serve.test_security serve.test_lifecycle serve.test_grammar serve.test_grammar_scope serve.test_responses_json serve.test_responses_tool_schemas
ctest --test-dir <native-build> --output-on-failure -R 'grammar|serve_input'
<native-build>/grammar_native_test data/grammars/cases.json <model-pack>/tokenizer <vocab-audit-output>
python tools/responses_schema_inventory_probe.py --base-url <server>/v1 --model qwen3.8-flash-next --out <new-directory>
python tools/responses_codex_tools_probe.py --base-url <server>/v1 --model qwen3.8-flash-next --catalog-dir docs/codex --out <new-directory>
python tools/responses_json_probe.py --base-url <server>/v1 --model qwen3.8-flash-next --out <new-directory>
```

| Check | Evidence |
|---|---|
| 239 service/API/lifecycle/grammar/schema tests passed | [Python output](python-tests.txt), including all 12 exact declarations, all 12 strict variants, negative strict validation, every byte split of boolean spellings, JSON/SSE equality and replay. Synthetic model events are labeled. |
| 10 native/CUDA groups passed | [Native output](native-tests.txt), including the dynamic-property regression, raw grammar corpus, thinking budgets, speculation and framing. |
| Actual 248320-token vocabulary passed | Same native output: 41 grammar corpus cases and bounded/nonempty title constraints. |
| 30 real generated JSON examples passed | [Schema receipt](native-schemas/result.json); each request and response is stored by its catalog name in that directory. |
| 12 real generated tool loops passed, 24 HTTP requests | [Tool receipt](native-tools/result.json); requests and typed SSE are stored by qualified tool name. Call IDs, names, namespaces, argument types/values and exact stream reconstruction are checked. |
| 7 strict JSON/tool/reasoning cases passed | [Strict-loop receipt](strict-tool-loop/result.json): five completed cases, expected incomplete output, and expected HTTP 400 before SSE for a remote reference. |

The definitions are [the captured Codex declarations](../../codex/tool-declarations-0.160.0.json),
with [explicitly synthetic sample arguments](../../codex/tool-argument-examples-0.160.0.json).
Accepting `view_image` as a client-owned call does not enable image input in the
text-only model profile.

A separate real Codex CLI verification at the immediately preceding source
`6468186d2fe25a006dc17c696fce651716bcdca3` used the complete tool inventory and
ran three unchanged Python tests successfully. Its
[actual tool calls and results](codex-verification-tool-loop.json) retain two
misspelled PowerShell commands and the recovery; the
[final client result](codex-verification-turn.json) records completion. This run
shared the service queue with probes, so its duration is not a speed benchmark.
The earlier [read/edit/test and JSON-title checkpoint](../JSON/REPORT.md) and
[interruption/steering checkpoint](../../responses-evidence/steering/REPORT.md)
remain separate evidence. These are pinned-client qualifications, not universal
Codex compatibility or evidence that every generated command is correct.

## Failures retained and limits

The [original dynamic-property failure](earlier-attempts/dynamic-properties.response.json)
was correctly reported as failed when the blocked valid key led to duplicate keys.
The [no-thinking tool run](earlier-attempts/tools-no-thinking/result.json) passed
9 of 12 fixtures; [medium reasoning before the boolean fix](earlier-attempts/tools-before-boolean-fix/result.json)
passed 10 of 12. The [boolean-fixed run](earlier-attempts/tools-before-empty-call-hint/result.json)
passed 11 of 12 and exposed the invented empty-call wrapper. These failures were
not reclassified as successful calls or silently retried by the server.

Native masks enforce the compiler's representable constraints. Mandatory final
Draft 2020-12 validation checks the original full schema, including constraints
the native grammar cannot enforce. A completed invalid answer/strict call is
never reported as success. Partial output at a token limit remains incomplete.
Known-sample generation does not prove that every schema value can be generated.

Lark custom tools, OpenAI-hosted tools, nested namespaces, multimodal results,
retained-response CRUD, background execution, compaction and WebSockets remain
separate unsupported capabilities. Schema size, native work/memory and context
limits still apply. G6 recovery remains deferred. The next gate is review and
publication of these local changes, preserving the Responses-first landing order.

Enablement and request examples: [JSON output](../../JSON_OUTPUT.md),
[Codex tool schemas](../../CODEX_TOOL_SCHEMAS.md), [local Codex setup](../../CODEX_LOCAL.md).
