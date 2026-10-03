# Experimental stateless Responses adapter

This branch adds a bounded `POST /v1/responses` profile over Strata's existing
service. It is disabled by default. The Responses adapter itself does not change
native inference, MTP, model loading, or the legacy Chat Completions structured-output
path. This stacked branch's separate [native grammar contribution](NATIVE_GBNF.md)
starts from the frozen R4 Responses checkpoint.

Enable it when starting an existing configured server:

```sh
python -m pip install -r requirements-responses.txt
python serve/server.py --engine strata --config strata-model.json --experimental-responses
```

Use the path of your actual model config in place of `strata-model.json`.
Set `STRATA_RESPONSES_REPLAY_KEY` in the server environment first. For a disposable
PowerShell test, generate a key without printing it:

```powershell
$env:STRATA_RESPONSES_REPLAY_KEY = python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

For a persistent deployment, supply the same secret from your environment/secret
manager on each start. Regenerating it invalidates earlier encrypted replay items.
Alternatively, add `"experimental_responses": true` at the top level of that
config. The CLI flag enables it even if the config says false; the effective
value is resolved once at startup. No per-request experimental field is needed.
When disabled, the route returns 404 and starts no Responses store or workers.

A CPU-only example, without a model download:

```sh
python serve/server.py --engine mock --script "Hello." --experimental-responses
```

The existing API key, Host/Origin checks and CORS configuration protect the route.
For SDK clients use the same base URL and API key as the other Strata endpoints:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8095/v1", api_key="your-strata-api-key")
response = client.responses.create(
    model="qwen3.8-flash-next",  # use a model ID advertised by /v1/models
    input="Say hello.",
    store=False,
    max_output_tokens=128,
)
print(response.output_text)
```

For a pinned interactive Codex profile, model metadata and coding sampling defaults,
see [Local Codex CLI with Strata](CODEX_LOCAL.md).

## Capability profile

The adapter supports text messages, per-request instructions, explicit `store:false`,
plain text and [native JSON output](JSON_OUTPUT.md), bounded metadata, temperature/top-p, and output limits.
Reasoning defaults to off unless a summary is requested, which defaults effort
to medium. Unknown behavior-changing parameters are errors.
Request bodies are bounded to 4 MiB; input context must fit the existing service.
Configured sampling defaults remain in effect when a request omits them.

| Capability | Policy |
|---|---|
| Full text/message history | Supply in `input`, including content-bearing returned messages and IDs. |
| Server storage | Explicit `store:false` required; omitted or true is rejected. |
| `previous_response_id`, conversation storage, ID-only references | Unsupported; no history is inferred from GPU cache. |
| Retrieve/delete/cancel/background | Not implemented. Disconnects use existing cancellation/draining. |
| `text.format` | Text, `json_object`, or `json_schema`; JSON requires the gbnf-v4 native build and requirements-json.txt. |
| JSON Schema / JSON-object generation | Native schema decoding plus mandatory final validation; see [JSON output](JSON_OUTPUT.md). |
| Streaming | Typed SSE from the same assembler as final JSON; set `stream:true`. |
| Function calls | Client-owned functions, flat or inside `type:namespace`; strict argument validation is supported. |
| Tool choice | `auto` or `none`; forced choices and `parallel_tool_calls:false` with tools are rejected. |
| Reasoning effort | `none`, `low`, `medium`; `high`, `xhigh`, `max` map to native `xhigh`. `minimal` is rejected. |
| Reasoning representations | Actual `reasoning_text`, generated summaries, authenticated Strata-issued replay tokens. |
| `reasoning.summary` | `auto`, `concise`, `detailed`, or null; non-null requires thinking. |
| `include` | Empty or `reasoning.encrypted_content`. Other representations are rejected. |
| `client_metadata` | Diagnostic strings in the existing optional monitor; never instructions or inference settings. |
| `prompt_cache_key` | Routing hint to the sole engine, echoed in the response. No separate cache partitions, billing, or retention guarantee. |
| Image/audio, hosted tools, compaction, WebSockets | Unsupported; rejected. |
| Raw GBNF | This stacked branch adds the [native GBNF extension](NATIVE_GBNF.md), requiring a qualified native build. The guide covers plain answers, reasoning/tool scopes and target-only/G5 speculative modes with their capability limits. The frozen R4 Responses-only checkpoint rejects it. |

To continue, append the previous response's complete `output` items and the new
user message to your supplied history. Resend any instructions you still want.
Responses and runtime handles are request-local; no parent record exists to mutate.
The optional existing API monitor can retain its usual bounded diagnostics in
memory. That is not Responses storage or a conversation continuation mechanism.

For streaming, pass `stream=True` to the same SDK call and iterate the returned
events. Text arrives as `response.output_text.delta`; terminal events are
`response.completed`, `response.incomplete` (output budget), or `response.failed`.
The stream includes lifecycle, output-item and content-part events with stable
indexes/IDs and increasing sequence numbers. Keep-alives are SSE comments.
Validation errors are JSON HTTP errors before SSE headers. A disconnected client
cannot receive a terminal event; existing cancellation drains/stops work before
the next request owns the engine. There is no custom cancellation SSE event.

## Functions and reasoning

Return function calls to your application, execute authorized operations there,
and append a `function_call_output` carrying the matching `call_id`. The output
item's `id` is separate. Multiple results are bound by `call_id`, then rendered
in the native template's call order. Every outstanding call needs one result.
Strata does not execute these functions through its MCP integration.

Namespaces keep `namespace` and `name` separate on the wire. At the existing
template/parser boundary only, `files.read_file` identifies `read_file` inside
`files`. Dots are excluded from each individual wire name, preventing collisions.
Nested namespaces and custom tools are rejected. With explicit `strict:false`,
parameter descriptions guide the model. With `strict:true`, complete generated
arguments are validated before the call is marked complete. Invalid arguments fail
the response and never produce a completed call. Tool envelopes/arguments are not
natively grammar-masked; final answer JSON uses the separate native schema path.
Strict functions require closed objects with every property required. Omitted/null
strict normalizes compatible schemas; explicitly open/dynamic objects retain the
documented best-effort mode and are reported as `strict:false`. See the
[tool schema guide](CODEX_TOOL_SCHEMAS.md).

Raw thinking is emitted as a reasoning item's `content`, with typed
`response.reasoning_text.delta/done` events. A requested summary is produced by
an additional answer-only generation through the same service after the primary
generation releases the FIFO. It is not copied raw thinking. This costs another
prefill and generation pass. Tool-like markup in this summary pass is literal
text, not a function call. The initial pass reserves one quarter of the output
allowance, capped at 128/256/512 tokens for concise/auto/detailed; the summary can
use the remaining allowance. Both passes count toward `max_output_tokens`.
Output following reasoning is buffered until its summary finishes so Codex sees
each reasoning item complete before the following message or function item.
The default, with no requested summary, streams directly.

Usage includes both passes' prompt and generated tokens. `reasoning_tokens`
counts generated tokens while the existing parser is in its reasoning region,
including the closing delimiter and any existing configured thinking-budget wrap.
`cached_tokens` counts native reused prompt tokens; `cache_write_tokens` is the
newly prefilled portion. These are local engine counts, not OpenAI billing data.
Known limitation: reasoning-budget continuation can misattribute reused tokens
to the original prompt. The native coding qualification does not establish
correct cache accounting for that path.
Usage is null when failure or cancellation prevents complete accounting.

Completed reasoning includes an authenticated encrypted replay token. Strata
uses the maintained `cryptography` Fernet implementation, not custom encryption.
The token contains the actual reasoning content, summary, item ID, model ID and
completion status. Only a server with the same deployment key can restore it.
Foreign tokens, tampering, mismatched IDs/models and conflicting visible fields
produce an error before generation. There is no attempt to decode another
provider's state. To replay visible raw `content` without `encrypted_content`,
also omit a nonempty `summary`; the current validator rejects that combination.
A summary alone cannot substitute for the underlying reasoning. Replaying the
full returned item preserves all fields and is the qualified Codex path.

Enabling the route reads `STRATA_RESPONSES_REPLAY_KEY` from the server environment.
No secret file is created. This follows the handoff's environment-only key rule.
Keep this deployment secret private and stable across restarts. Servers sharing
it can restore each other's tokens for the same advertised model. Missing or
malformed keys fail startup before a model is loaded.
Disabled servers neither import the optional crypto dependency nor load a key.

The [text example index](responses-examples/INDEX.txt) covers the supported and
excluded requests. The [complete nested-type examples](responses-examples/ALL_CREATE_PIECE_EXAMPLES.txt)
and [type inventory](responses-examples/ALL_CREATE_FIELDS_AND_PIECES.txt) are
pinned to openai-python 3.23.0. The [encryption round trip](responses-examples/ENCRYPTION_ROUNDTRIP.txt)
shows synthetic plaintext, a public throwaway example key, actual ciphertext and
the restored payload. It contains no deployment secret.

## Ownership and tests

`serve/responses.py` validates capabilities and owns one request-local assembler.
It consumes `Service.run()` semantic events below HTTP serialization. Final JSON
and typed lifecycle/output events come from that same assembler. Token fragments
are appended to incremental buffers; serialization does not rewrite their text.
The handler issues commands and sends snapshots. Existing service admission,
loading, cancellation, draining and metrics remain authoritative for execution.

```sh
python -m pip install -r requirements-responses.txt
python -m unittest serve.test_responses -v
```

Codex CLI 0.160.0 completed a real local read/edit/verify/result loop against this
adapter using explicitly scripted MockEngine output. The
[receipt](responses-evidence/R3/codex-workspace/codex-task.json),
[actual exchanges](responses-evidence/R3/codex-workspace/codex-exchanges.json), and
[tested profile](responses-evidence/R3/codex-workspace/codex-profile.toml) record
the exact client and configuration. The server never edited the file; the client
did. The profile uses the documented Windows sandbox and a normal disposable
workspace; private Python temporary directories failed under the restricted token.

To use that profile locally, save its contents as `strata.config.toml` in your
[Codex configuration directory](https://learn.chatgpt.com/docs/config-file/config-basic#configuration-precedence),
set `base_url` to your Strata server, and run:

```sh
codex --profile strata
```

The tested loopback profile has no API key. For a key-protected server, configure
the provider's documented `env_key` to read your existing Strata key environment
variable. Use a model ID advertised by your server. The pinned client reports
fallback metadata for this local model name; no hosted model is impersonated.

These tests establish this bounded protocol profile, not universal Codex support
or native-model quality. Larger sessions can ask for excluded compaction or
other capabilities and will get explicit errors. The
[R0 report](responses-evidence/R0/REPORT.md) describes the original capture-time
blockers; R3 implements the namespaces, summaries and replay it observed. The
[R4 receipt](responses-evidence/R4/R4-receipt.json) pins that checkpoint; the
[native grammar guide](NATIVE_GBNF.md) records the subsequent G phases.

A subsequent [native hello-world checkpoint](responses-evidence/native-codex/REPORT.md)
ran the real pinned Codex client against Strata's Coder IQ1_M model on llm-49 over
an authenticated LAN tunnel. It returned `Hello world` and completed its turn
with external HTTP requests blocked locally. The report includes the exact
profile, native logs, captured request/response, enablement commands and a
reusable probe. This establishes the greeting session.

The subsequent [native coding checkpoint](responses-evidence/native-codex/coding-task/REPORT.md)
qualifies a real read/edit/verify task with Codex 0.160.0 and an explicit
[local-tool profile](responses-evidence/native-codex/coding-task/codex-profile.example.toml).
Eight Responses requests and nine real shell calls repaired a parser; 12 task
tests, 45 independent checks and 19 companion native HTTP probes passed. The
[plain-text transcript](responses-evidence/native-codex/coding-task/TOOL_LOOP.txt)
shows actual calls, failures, recovery and reasoning without decrypting replay
blobs. The stock fallback profile failed an earlier attempt; the report records
that limitation and the two literal-markup parser fixes it exposed. This is a
specific client/model/profile qualification, not universal Codex compatibility.

Protocol references: [Responses](https://developers.openai.com/api/reference/resources/responses),
[typed streaming](https://developers.openai.com/api/docs/guides/streaming-responses),
[function calling](https://developers.openai.com/api/docs/guides/function-calling).
