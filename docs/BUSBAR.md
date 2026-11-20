# The busbar gateway

`tools/busbar_gateway.py` puts one OpenAI-compatible address in front of several model providers. You give it a name
like `fast`, and `fast` can mean 80% Claude and 20% GPT, with Bedrock answering when both are down. Your app changes one
line:

```python
client = OpenAI(api_key=BUSBAR_TOKEN, base_url="http://localhost:8080/v1")
client.chat.completions.create(model="fast", messages=[{"role": "user", "content": "Hi"}])
```

It needs no GPU, no model and no engine. It is a separate tool, not part of the Strata server.

## Run it

```sh
python tools/busbar_gateway.py --config busbar.json --port 8080
```

Before running anything, print the routing and send nothing:

```sh
python tools/busbar_gateway.py --config busbar.json --plan --requests 20
```

| Flag | What it does |
|---|---|
| `--config` | The JSON file with the pools (see `tools/busbar.example.json`) |
| `--host` | Address to listen on, default `127.0.0.1` |
| `--port` | Port, default `8080` |
| `--token` | Require this key on every request. Empty means no key, which is only safe on `127.0.0.1` |
| `--plan` | Show how the next requests are routed, then exit |
| `--pool` | Which pool `--plan` shows, default the first one. A name that is not a pool prints an error and exits with 2 |
| `--requests` | How many requests `--plan` shows, default 100 |

## The config

```json
{
  "default_pool": "fast",
  "pools": {
    "fast": {
      "members": [
        {"name": "claude", "weight": 80, "wire": "anthropic",
         "base_url": "https://api.anthropic.com/v1", "model": "claude-sonnet-4-5",
         "key_env": "ANTHROPIC_API_KEY"},
        {"name": "gpt", "weight": 20, "wire": "openai",
         "base_url": "https://api.openai.com/v1", "model": "gpt-5.2", "key_env": "OPENAI_KEY"}
      ],
      "failover": ["bedrock"],
      "timeout_s": 120, "cooldown_s": 30, "max_fails": 2
    },
    "bedrock": {"members": [{"name": "bedrock", "weight": 1, "wire": "openai",
                             "base_url": "https://bedrock-runtime.us-east-1.amazonaws.com/model",
                             "model": "anthropic.claude-sonnet-4-5",
                             "key_env": "AWS_BEARER_TOKEN_BEDROCK"}]}
  }
}
```

`wire` says which API the member speaks: `openai` or `anthropic`. An Anthropic answer is turned into the OpenAI shape
before it reaches your client, so the client sees one format. Keys come from the environment through `key_env`, or from
`key` in the file; the file is read once at start.

## How a request is routed

The split is exact, not random. The gateway uses the smooth weighted round robin, the same method nginx uses, so 80 of
100 requests go to Claude and 20 to GPT, spread evenly instead of in clumps:

```
claude claude gpt claude claude claude claude gpt claude claude
```

If a member fails, the request is tried on the next member of the pool, then on the pools named in `failover`, heaviest
member first. After `max_fails` failures in a row (2 by default) a member is put on cooldown for `cooldown_s` seconds
(30 by default) and skipped until the cooldown ends. One success clears the count. If every member of every pool fails,
the gateway answers 502 with the last error.

A pool name your client sends must exist. A typo is an error, not a silent fallback to the default pool, so traffic does
not quietly go somewhere else.

A member whose connection dies before an answer - a socket closed in the middle of the request, which Windows reports as
an abort - is a failure like any other: the request moves to the next member. A stream the provider cuts short is counted
as a failure too; a client that walks away mid-stream is not, because the provider did answer.

## Endpoints

| Route | What it returns |
|---|---|
| `POST /v1/chat/completions` | The completion. `stream: true` passes the upstream SSE through |
| `GET /v1/models` | The pool names, and the members behind each |
| `GET /v1/stats` | Per member: requests answered, failures, tokens, and how long a cooldown is left |
| `GET /v1/health` | `{"ok": true}` and the pool names |

`max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`, `stop`,
`response_format`, `seed`, `tools`, `tool_choice` and `reasoning_effort` are passed to the upstream as they are.

## Safety

Set `--token` (or `BUSBAR_TOKEN`) if anything besides your own PC may reach the port. Without a key the gateway prints a
warning and stays on `127.0.0.1`. The gateway holds no provider keys of its own beyond what the config and environment
give it, and it sends no usage data anywhere.

## Tests

The routing rules are checked without any network or provider key, using stand-in members that answer or fail:

```sh
python -m unittest tools.test_busbar_gateway
```

34 tests cover the exact split, the spread, failover to another pool, cooldown, cooldown of a failover member, the
payload for both wires, the Anthropic-to-OpenAI conversion, streaming in both directions, a stream the provider cuts
short, a client that stops reading, the `--plan` flag, and the HTTP routes including the key check and the streaming
route end to end.
