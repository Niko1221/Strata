# Native Codex hello-world checkpoint

Result: **pass**. A real local Codex CLI session received `Hello world` from
Strata's native model through `POST /v1/responses` and completed normally.
No model output, reasoning or summary was scripted or grammar-forced.

Branch: `work/gbnf`. Worktree:
`C:/Users/dflanag3/Documents/fleet/strata-native-gbnf`.
The running production source is the G5 receipt checkpoint
`26ed30afbd3718e658a3c8d42df900c277f6bcda`, whose implementation is
`2daa12ea803bff43706cb872527e4df5268f940f`.
The following `native-codex-receipt.json` pins this probe and its evidence commit.
No production code changes were needed for this checkpoint.

## The actual session

```text
User: Reply with exactly "Hello world". Do not use any tools.
Codex: Hello world
Process exit: 0
Turn: completed
```

The [Codex output](hello-4abca3e0f5/codex-output.txt),
[request and response](hello-4abca3e0f5/exchanges.json),
[probe result](hello-4abca3e0f5/result.json),
[native engine log](hello-4abca3e0f5/engine.txt), and
[server log](hello-4abca3e0f5/server.txt) agree on the completed turn.
The client made one streaming Responses request with its eight normal tool
entries, including namespaces, `reasoning.summary: auto`,
`include: [reasoning.encrypted_content]`, and `store: false`.
No request-rewriting proxy was used. The server used the existing semantic service.

The response contains actual reasoning text, a separately generated summary,
an authenticated Strata replay token and a completed message saying `Hello world`.
The replay key stayed in the server environment and was not saved; the encrypted
token is retained as received. The [text example](../../responses-examples/NATIVE_HELLO_WORLD.txt)
shows the visible items and explains the encrypted field without decrypting it.

The turn took 50.005 seconds after server readiness. Native logs record a
9,343-token initial prompt (46.846 seconds of prefill, 24 generated tokens) and
a 95-token summary prompt (28 generated tokens). Responses and Codex both report
9,438 input tokens and 52 output tokens, including 20 reasoning tokens.
This is one measured cold-prompt greeting, not an agent-throughput benchmark.

## Native and network provenance

Only the authorized llm-49 host (r4090, RTX 4090) ran inference. Its existing
Coder IQ1_M model, tokenizer, MTP weights and installed Python environment were
used. Context was 32768, KV was INT8, MTP maximum was four, suffix lookup was
off, temperature was zero and the server's existing thinking budget was 256.
See the exact [model configuration](hello-4abca3e0f5/config.json) and
[startup identities](hello-4abca3e0f5/native-server.json).
The native binary SHA-256 was
`b335d98cf16d08795d2e089e12484c3c7212c7b5b85755676b76aab113ba7bbe`.
Every top-level `serve/*.py` source hash was matched to this worktree before startup.

The Windows client was the already installed R0 pin, Codex CLI 0.160.0:
`fdda5fa3cf3fb3d000b876720742857676293e4315e4b045fae6f8bd7e866d1d`.
It used a fresh client home and the [recorded profile](hello-4abca3e0f5/codex-profile.toml),
with an environment-only Strata API key and no OpenAI login. Its real command
is in the [invocation](hello-4abca3e0f5/invocation.json) and probe result.

```text
Windows Codex -> Windows loopback -> authenticated direct SSH over LAN
             -> llm-49 loopback -> Strata HTTP service -> native CUDA model

Codex background HTTP(S) -> separate loopback reject proxy -> HTTP 502
```

Web search was disabled. The reject proxy denied five attempted background
CONNECT requests: two to chatgpt.com and one each to github.com, api.github.com
and ab.chatgpt.com. It opened no outbound connection. These attempts were not
silently omitted from the result. The native server's HTTP proxy environment
also pointed at loopback; offline model-library flags were set. There were no
downloads, package installs, Git network operations or external model calls.
This proves this session completed with external HTTP unavailable to the client;
the proxy is not represented as an operating-system firewall for arbitrary programs.

The [cleanup record](hello-4abca3e0f5/cleanup.json) confirms the test server stopped
and the GPU had no compute processes afterward. The SSH tunnel also exited.
No other host or existing server was changed.

## Reproduce using installed local assets

Start the ordinary Strata server with your existing native model config, the
Responses flag and API monitor. Supply `STRATA_API_KEY` and
`STRATA_RESPONSES_REPLAY_KEY` through the environment as described in the
[Responses guide](../../RESPONSES.md); do not put keys in a fixture or command line.
For this client profile, allow enough context for its roughly 9,343-token prompt.
The tested model config is linked above; its local model paths describe installed
assets, not downloadable artifacts.

```sh
python serve/server.py --engine strata --config YOUR_EXISTING_MODEL_CONFIG.json \
  --host 127.0.0.1 --port 8095 --experimental-responses --api-monitor
```

For a LAN server, forward that authenticated loopback endpoint from Windows:

```powershell
ssh -N -T -o ProxyJump=none -o ProxyCommand=none -o ExitOnForwardFailure=yes -L 127.0.0.1:8095:127.0.0.1:8095 llm-49
```

In another terminal, set `STRATA_API_KEY` to the same server key, then run the
checked-in probe. It creates a fresh Codex home, starts its own reject proxy,
runs the actual client and saves evidence. Use a new output directory each time.
It requires the recorded Windows Codex binary; it never downloads a substitute.

```powershell
python tools/responses_native_codex_probe.py --codex PATH_TO_PINNED_CODEX.exe --base-url http://127.0.0.1:8095/v1 --out .responses-runtime/hello-evidence
```

For a normal interactive session against an already running server, install the
recorded profile as `strata.config.toml` in your Codex configuration directory,
set its base URL and key environment variable, and use `codex --profile strata`.
The probe additionally supplies the offline reject proxy; a normal client session
should use your own equivalent egress policy when internet access must be blocked.

## Limits and next gate

This closes the requested native hello-world goal. It does not qualify native
tool execution, all Codex versions, long histories, compaction, strict schemas or
Lark custom tools. The pinned client emits its fallback-model-metadata warning;
the warning is retained and no hosted model is impersonated. Existing unsupported
capabilities still fail explicitly. G6 recovery remains deferred.

A later [native coding checkpoint](coding-task/REPORT.md) qualifies real client
shell tools and a complete repair/test task with a pinned local-tool profile.
Its [plain-text transcript](coding-task/TOOL_LOOP.txt) and raw captures include
failed attempts, parser fixes and successful verification. The greeting's source,
binary and measurements above remain this earlier checkpoint's own evidence.

The private LAN orchestration needed two local setup corrections before it ran:
an f-string delimiter fix and removal of an unnecessary local crypto-package
dependency. Neither attempt launched a server or exercised Strata. The actual
native Codex attempt above passed without changes to production code or client
request fields.
