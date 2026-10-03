# Codex function schema compatibility

The exact [12 function declarations captured from Codex 0.160.0](codex/tool-declarations-0.160.0.json)
and [sample arguments](codex/tool-argument-examples-0.160.0.json) form the local
compatibility test set. They include exec_command, write_stdin, request_user_input,
view_image, five multi_agent_v1 namespaced functions and three goal functions.
Strata generates calls and accepts results; the local Codex client owns execution.
Accepting a view_image call does not add image input to this text-only model profile.

`serve/test_responses_tool_schemas.py` exercises every definition through final
JSON, typed SSE, exact argument reconstruction, stable namespace/name/call IDs and
full-history result replay. It also tests all twelve schemas after omitted-strict
normalization. These are explicit synthetic-engine transport tests; no shell,
image, agent or goal action is performed. The native Codex task separately proves
real shell execution on the client.

Responses supports `strict:true` function parameters with mandatory JSON Schema
draft 2020-12 validation. Each object must declare `additionalProperties:false`
and mark all properties required; nullable properties represent optional values.
The existing assembler checks the finished argument object **before** emitting
function_call_arguments.done or a completed output item. Invalid arguments fail
the response; they are not rewritten, silently repaired or executed by Strata.
Partial deltas remain partial. Tool results are arbitrary client-supplied text,
not inputs to the argument validator.

At the native XML parsing boundary, explicitly boolean parameters accept Qwen's
`True`/`False` spellings as well as JSON `true`/`false`. This produces one canonical
JSON value before argument deltas are emitted. String and undeclared parameters
retain their existing interpretation; no Python evaluation or post-stream repair
is involved. All byte-split positions and strict JSON/SSE equality are tested.
The native prompt also shows the exact empty call form for closed functions with
no parameters, such as `get_goal`. It does not add an `arguments` parameter or
remove invented parameters from model output.

This is validation at the function-call completion boundary. Native token masks
constrain the final answer when `text.format` requests JSON; they do not constrain
the Qwen XML tool envelope or its arguments. Invalid generated arguments therefore
produce a failed response instead of an invalid completed tool call. There is no
automatic tool retry or claim that every model attempt will succeed.

Explicit `strict:false` retains best-effort descriptions. Omitted/null strict
normalizes compatible objects into strict schemas, and the response reports the
normalized declaration. An explicitly open object or patternProperties cannot be
closed without changing its intended behavior, so omitted-strict uses the
documented non-strict fallback and reports `strict:false`. Explicit strict never
uses that fallback. Install `requirements-json.txt` for schema validation.

The [official function-calling contract](https://developers.openai.com/api/docs/guides/function-calling)
describes strict parameters and omitted-strict normalization. Custom Lark tools,
OpenAI-hosted tools, nested namespace groups and multimodal tool results remain
separate capabilities; they are not implemented by routing a function name.

The schema ZIP's SDK wire inventory includes such excluded types for inspection.
Its compatibility claim concerns the twelve declarations above and individually
labeled JSON answer examples, not every type in the SDK inventory.

The [native schema/tool checkpoint](gbnf-evidence/schema-inventory/REPORT.md)
records all twelve real generated calls and mock-result continuations, 30 JSON
examples, 239 Python checks and the earlier failed attempts. The probe supplies
known sample arguments; it does not execute these client tools.
