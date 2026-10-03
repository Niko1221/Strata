# Native JSON output for Responses

The GBNF branch supports Responses `text.format` with `json_object` and
`json_schema`, including `strict:true`. This extends the earlier handoff scope
at the user's explicit request on 2026-10-03. It uses the existing native
XGrammar compiler, matcher, token masks, single-engine queue and cancellation.
There is no HTTP translation, second inference implementation, or object-only
fallback for schemas.

Build the branch with `STRATA_ENABLE_GBNF=ON` using [NATIVE_GBNF.md](NATIVE_GBNF.md),
install `python -m pip install -r requirements-json.txt`, and start Strata with
`--experimental-responses` plus your existing model config and authentication.
The native engine must advertise `grammar=gbnf-v4`. Older grammar builds are
rejected before generation for JSON requests. Plain GBNF retains v2/v3 compatibility.

For an arbitrary JSON object:

```json
{"model":"qwen3.8-flash-next","store":false,"input":"Return an object with a greeting.","max_output_tokens":256,"text":{"format":{"type":"json_object"}}}
```

For the exact schema used by Codex's automatic chat-title task:

```json
{
  "model": "qwen3.8-flash-next",
  "store": false,
  "input": "Create a short title for fixing an addition function.",
  "text": {"format": {
    "type": "json_schema",
    "name": "codex_output_schema",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {"title": {"type": "string", "minLength": 1, "maxLength": 36}},
      "required": ["title"],
      "additionalProperties": false
    }
  }}
}
```

Set `stream:true` for typed SSE. The concatenated text deltas are exactly the
JSON text in the final response; validation does not reserialize it or repair it.
A completed assistant answer is parsed with duplicate-key, finite-number and
Unicode checks, then validated against JSON Schema draft 2020-12. Schema validity,
size, references and required validator availability are checked before generation.
External schema references are not fetched. The native compiler enforces its schema
constraints during decoding; final schema validation also checks constraints that
cannot be represented by that grammar. Invalid output produces `response.failed`,
never a successful response with weakened guarantees. Output-budget exhaustion
produces `response.incomplete`; its partial text need not yet be complete JSON.

Reasoning and client-owned function calls can precede the JSON answer. The schema
constrains the answer, not tool results, function arguments or reasoning summaries.
A tool-only response can complete without an answer, allowing the normal client
loop to continue. [Strict function parameter validation](CODEX_TOOL_SCHEMAS.md) is supported
before a completed call is emitted; external tool results stay unconstrained.
Lark custom tools remain a separate unsupported capability. Raw `grammar` and `text.format` JSON requirements
cannot be combined on one request.

The server's configured thinking budget is enforced by native token masks for
JSON requests: after that many reasoning tokens only the native end-of-thinking
token is legal. No prompt text is injected to force a wrap-up. Speculative forks
and checkpoints preserve the same counter. JSON generation has the existing
8192-token maximum and 8192-byte schema bound; an omitted output limit uses the
smaller of available context and that token maximum. Total request and native
work/memory bounds remain active. Schema compilation has a 150-million-unit work
ceiling, a 2.5-second operation deadline and a 16 MiB compiled-size limit. The
Codex title schema takes about 1.2 seconds and 5.2 MiB with this Qwen 248K-token
vocabulary on llm-49; the earlier 5-million-unit raw-grammar limit rejected it.
JSON matching has a 64-million-unit sequence budget, including discarded speculative
work, with the existing per-operation deadline. Raw GBNF's budgets are unchanged.
Unsatisfiable or unrepresentable schemas can
fail preflight, and validation failures are not silently retried.

The dependency pin `xgrammar-0.2.8-strata-budget1-json1` also corrects the native
compiler's handling of `patternProperties` together with `additionalProperties`
when no named properties are present. Earlier builds could forbid valid extra
keys. Prepare a new dependency directory with `tools/prepare_xgrammar.py` and
rebuild; the existing dependency directory is preserved. The original schema
still governs final validation, including overlapping patterns and duplicate keys.

The legacy Chat Completions `response_format` implementation is unchanged. The
Responses-only branch still has no native JSON decoder; use the stacked GBNF branch
for this feature. The [local Codex profile](CODEX_LOCAL.md) uses it automatically
when Codex sends a JSON title request.

The [native JSON checkpoint](gbnf-evidence/JSON/REPORT.md) records the real Codex
task and title, seven GPU-backed HTTP cases, schema/tool/reasoning streaming,
231 Python tests, ten native tests and the independent BF16 4B comparison.
The later [schema and tool checkpoint](gbnf-evidence/schema-inventory/REPORT.md)
adds 30 native JSON examples, all twelve captured Codex tool loops and strict
function validation, with 239 Python checks and the retained failing cases.

References: [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[XGrammar compiler](https://xgrammar.mlc.ai/docs/latest/api/python/grammar_compiler.html).
