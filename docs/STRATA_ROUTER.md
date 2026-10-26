# Running several Strata PCs behind strata-router

[strata-router](https://github.com/mrq1911/strata-router) puts several Strata servers behind one OpenAI- and
Anthropic-compatible address. It keeps each conversation on the server that already read it, so follow-up messages
reuse that server's prompt cache, and moves a conversation only when its server is down or stays busy longer than
reading the prompt again elsewhere would take. It is a separate project (one Node.js file, no dependencies), not
part of Strata.

## Quick start

On each PC, start Strata with batch slots, reachable from the network, with a key:

```bash
export STRATA_KEY='<a long random key>'
strata launch --family qwen --model q2_0 --parallel 2 --access lan --port 8080 --api-key-env STRATA_KEY
```

On the PC that runs the router (Node.js 20 or newer):

```bash
git clone https://github.com/mrq1911/strata-router
(umask 077 && printf '%s' "$STRATA_KEY" > strata-key)
STRATA_BACKENDS="a=10.0.0.1:8080,b=10.0.0.2:8080" MAX_INFLIGHT=2 STRATA_API_KEY_FILE=strata-key \
  node strata-router/router.mjs
```

Point your apps at `http://<router PC>:8000/v1` with the same key. Each answer carries `x-strata-backend: a` or `b`,
and `http://<router PC>:8000/stats` shows which server holds which conversation.

## What to set

| Setting | Why |
| --- | --- |
| `STRATA_BACKENDS` by IP address | Strata refuses host names it does not know (`403 Host ... is not allowed`). Names need `STRATA_ALLOWED_HOSTS` on each server. |
| `STRATA_API_KEY_FILE` | A server with a key answers `/metrics` only with the key. Without it the router counts the server as down. |
| `MAX_INFLIGHT` = `--parallel` | The router sends each server at most this many requests at once and queues the rest itself. |
| `PREFILL_TOK_S` | Prompt reading speed until the router has measured it per server. It decides how long a conversation waits for its busy server before moving. |

Clients send the key on their own requests. The router passes it through and never adds it, so it does not open the
servers to whoever reaches the router.

## Measurements (October 2026)

Two PCs with an RTX 3090 each, Strata 0.1.40.3, Qwen3.8-Flash-Next Q2_0, 262,144 context, `--kv int8`, `--batch 2`.
Short prompts, 200 output tokens, repeated.

One server:

| Requests at once | Each | Together |
| --- | --- | --- |
| 1 | 114 tok/s | 114 tok/s |
| 2 | 63 tok/s | 122 tok/s |
| 4 | ~31 tok/s | 124 tok/s |

Two servers through the router:

| Conversations | Each | Together |
| --- | --- | --- |
| 2 (1 per PC) | ~100 tok/s | ~200 tok/s |
| 4 (2 per PC) | ~60 tok/s | ~245 tok/s |
| 8 (4 per PC) | ~30 tok/s | ~245 tok/s |

Reading a prompt again on a server that does not hold the conversation, at ~1,385 tok/s:

| Prompt | Read again |
| --- | --- |
| 32k tokens | ~23 s |
| 131k tokens | ~95 s |
| 262k tokens | ~190 s |

With the conversation cache on (`--conversation-cache-mib 32768 --conversation-cache-slots 8`), the server that holds
the conversation restored it at 5-6 ms per 1,000 tokens: ~1.4 s for 262k tokens.

The router was not run against 0.2.5. The `/metrics` fields it reads (`live.state`, `live.queued`, `live.running`,
`live.slots`, `engine.batch_slots`, `conversation_cache`) and the key check on `/metrics` are the same in 0.2.5's
`serve/server.py`.
