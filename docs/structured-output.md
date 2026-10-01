# Structured Chat Completions

Strata supports `response_format` on `/v1/chat/completions`:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "dialogue",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {"voiceover": {"type": "boolean"}},
      "required": ["voiceover"],
      "additionalProperties": false
    }
  }
}
```

`{"type":"json_object"}` enforces a JSON object without imposing a field schema.
`{"type":"text"}` retains normal decoding. Strict schemas require every object
property and `additionalProperties: false`; optional values can use a nullable
type. Local references, nested arrays, enums, numeric bounds and `anyOf` branches
are supported. Unsupported or unenforceable schemas fail with HTTP 400 before
model loading or generation. Remote schema references are not fetched.

The Python server uses [llguidance](https://github.com/guidance-ai/llguidance) to
compute the allowed tokens. The native CUDA sampler masks forbidden logits before
top-k, nucleus filtering, temperature and selection. Every selected token updates
the request's grammar state. There are no retries, output repairs or inserted
default values. Objects use schema property order. Reasoning, when enabled, must
close before the constrained JSON body; literal tool tags inside JSON strings are
preserved.

Structured requests use single-token verify windows and skip speculative drafting.
This avoids accepting unconstrained speculative tokens. Ordinary requests retain
their existing speculative path. Grammar state and masks never enter the prompt
prefill or a following request.

Streams deliver incremental JSON prefixes through ordinary Chat Completions SSE.
When the token budget runs out, the completion has `finish_reason: "length"` and
may contain partial JSON. The OpenAI SDK's parsing and stream helpers then raise
their normal `LengthFinishReasonError`. A normally completed body is independently
validated before its terminal success chunk; a backend invariant violation returns
HTTP 502 or an SSE error. Partial streams cannot be retracted after delivery.

```python
from openai import OpenAI
from pydantic import BaseModel

class Dialogue(BaseModel):
    voiceover: bool

client = OpenAI(base_url="http://127.0.0.1:5001/v1", api_key="local")
result = client.chat.completions.parse(
    model="your-model",
    messages=[{"role": "user", "content": "The narrator speaks off-screen."}],
    response_format=Dialogue,
)
print(result.choices[0].message.parsed.voiceover)
```

Build the native engine from this source and install the setup dependencies. Its
`READY` protocol advertises `grammar_mask_v1`; an older binary is explicitly
rejected for structured requests instead of falling back to prompting. Only the
Chat Completions response-format interface is covered; this does not add the
Responses API, structured tool calling or a provider-specific refusal policy.

Checks: `python -m unittest serve.test_structured serve.test_grammar`,
`grammar_mask_test` and `sampler_parity --selftest`. OpenAI SDK checks run when
the optional `openai` package is installed; no cloud API key or cloud request is
used. `grammar_mask_test` proves both greedy and sampled decoding cannot select
forbidden high-probability tokens.
