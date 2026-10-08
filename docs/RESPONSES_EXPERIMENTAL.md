# Experimental Responses persistence and summaries

These options extend main's existing Responses assembler and shared generation queue. Both default to off.
They do not change Chat Completions or Anthropic requests and do not require a separate model process.

## Instruction and interruption fixes

System/developer input messages retain their instruction role, including permission updates arriving after user
turns. Templates requiring one leading system message receive these messages in their original order in that
block. A late update therefore changes the prompt prefix and can reduce native cache reuse.

A completed reasoning item remains in replay history when an answer was interrupted before its assistant text
arrived. The chat renderer preserves these reasoning-only assistant turns. Raw reasoning remains distinct from
generated summaries; a summary is never substituted for raw reasoning during replay.

## Persistence

Add these options to the normal server command:

```text
--experimental-responses-persistence --responses-store-path ./responses-history
```

Equivalent config entries (paths in config are relative to the config's `cwd`, if set):

```json
{
  "experimental_responses_persistence": true,
  "responses_store_path": "./responses-history",
  "responses_retention_s": 2592000,
  "responses_store_max_mib": 256
}
```

With persistence enabled, `store` defaults to `true`. Each request receives a new immutable response ID. Send
only new input with `previous_response_id`; the server reconstructs that parent's input and output item history.
Two children of one parent are separate branches. Current top-level `instructions`, tools and sampling settings
are not inherited from the parent. Instructions supplied as historical input items remain part of that history.
Completed and token-limited (`incomplete`) parents can continue; active, failed and cancelled parents cannot.
Parents must belong to the current model name. A name check cannot detect different weights deployed under the
same alias: use a different store directory or alias when replacing the model.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="your-server-key")
first = client.responses.create(model="strata", input="Remember the variable name: count.")
second = client.responses.create(
    model="strata", previous_response_id=first.id, input="What was the variable name?"
)
print(second.output_text)
print(client.responses.retrieve(second.id).status)
print(client.responses.input_items.list(second.id, order="asc", limit=20))
client.responses.delete(second.id)
```

The endpoints are `GET /v1/responses/{id}`, `DELETE /v1/responses/{id}`, and
`GET /v1/responses/{id}/input_items`. Input items support `order` (`desc` by default), `limit` (1–100, default 20),
`after` and `before`. IDs within a conversation must be unique. `store: false` can continue a stored parent but
does not retain the new response. With persistence disabled, main's stateless behavior remains: responses return
`store: false`, chaining is rejected and persistence routes return 404.

Storage uses SQLite transactions with one server owning the directory. Turns share ancestor records instead of
duplicating the full conversation. A successful terminal response is stored before it is acknowledged. An active
GET returns its initial `in_progress` snapshot, not a token-by-token snapshot; streaming supplies live output.
Deleting an active response hides it immediately and completion cannot recreate it. Deleting a parent does not
break existing descendants: necessary ancestor data stays internal until the descendants are also gone.

Default retention is 30 days from creation, checked on store operations. Set `responses_retention_s` to zero to
retain until explicit deletion. Active records are not expired while generation is running. On restart,
unfinished records become failed with `server_restarted`; completed histories remain available. Expired or
deleted IDs return 404. Ancestors needed by unexpired descendants remain internally available.

The capacity setting bounds admission and successful writes by serialized input/response bytes. SQLite files,
journals and failure records can exceed that logical limit; it is not a filesystem quota. A full store refuses
new admissions or terminal success with `response_store_full`, rather than silently evicting active histories.
The API key, Host checks and deletion Origin checks use the existing server protections. This is one shared
store for the server's clients, not per-user isolation.

This stores readable API items, not native KV state or hidden profiles. It does not promise fast prefill after
restart: native prompt caching and `/slots/0?action=save|restore` remain separate mechanisms. `conversation`,
`background`, `item_reference`, cancellation endpoints and compact endpoints remain unsupported. When changing
the experimental schema in a future version, migrate the store or select a new directory.

## Switchable second generation

```text
--experimental-responses-summaries
--no-experimental-responses-summaries
```

The config key is `experimental_responses_summaries` (boolean, default `false`). An explicit CLI flag overrides
the config in either direction. The persistence flag also accepts `--no-experimental-responses-persistence`.

When enabled, a request with `reasoning.summary` set to `concise`, `auto` or `detailed` may run a second generation
to summarize its raw reasoning. No summary is generated if the primary response contains no reasoning. Merely
enabling the flag does not summarize requests that omit `reasoning.summary`.

When disabled, the same request runs its primary generation only and returns raw reasoning, answer text and tool
calls, with an empty `summary`. This is the low-latency setting. Disabling summaries does not disable persistence.

The primary reasoning streams first. Answer/tool events wait for the summary when a second pass is needed. The
summary uses the same queue, so another request may run between its two passes. It emits
`response.reasoning_summary_part.added`, summary text deltas, text/part done events, and then the reasoning
item's done event. Both streaming and non-streaming results contain the same summary.

For thinking-enabled requests the server reserves one quarter of the output budget, capped at 128 tokens for
`concise`, 256 for `auto`, and 512 for `detailed`. The primary pass gets the rest; the summary can use the remaining
budget after the primary finishes. At least four output tokens are needed. Usage includes both generations'
input, cached input and output tokens, while reasoning-token usage counts the primary reasoning. Exhausting
either pass's budget produces `incomplete`; a failed summary produces `failed`, not a successful answer with a
fabricated summary. The primary answer is retained when only the summary runs out of tokens.

The second pass adds generation and queue time and changes the engine's current cache. Applications managing
their own native checkpoints must account for that. No P4 latency improvement is claimed by these contract tests.

## Validation

```text
python -m unittest serve.test_responses serve.test_responses_experimental -v
```

Tests use the mock engine, real local HTTP and temporary disk stores. They cover raw-reasoning replay, late
instructions, both summary settings, combined budgets, event order, tools, branching, restart, deletion,
retention, capacity and authentication. An optional test uses the official `openai` Python SDK when installed.
The existing frontend, server, security, lifecycle and structured-output suites cover unchanged serving paths.
