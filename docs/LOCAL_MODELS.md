# Local model selection

The Chat page separates native Strata configurations from additional models running in other servers on this PC.
The same saved chats and compaction controls work with both groups. Preparing a model and selecting it are separate
steps: these controls do not download weights or install another inference engine.

## Starting the page

Install Strata and at least one native model through [the normal setup](INSTALL.md). On Windows, `START-Models.bat`
starts the page with native weights loaded only when needed. It uses the installed default configuration unless you
pass `--model` with an installed configuration tag. An existing Strata server is reused without changing its model.

```text
START-Models.bat
START-Models.bat --model iq2_xs
START-Models.bat --no-browser
```

The same launcher can be used from Python on Linux:

```text
python tools/start_models.py --no-browser
```

Ollama is optional. Start it separately, or pass `--start-ollama` to launch an installed Ollama server on
`127.0.0.1:11434`. Other compatible servers must be started separately. This launcher does not install Ollama,
Bonsai, their models or their server configuration.

## Native configurations

Native switching is opt-in. For each configuration you want to select in the page, set these fields in its local
`strata-<tag>.json`, keeping its existing engine, model, tokenizer, memory and authentication settings:

```json
{
  "model_switch": true,
  "host": "127.0.0.1",
  "port": 8080
}
```

This is a partial example to merge into an installed configuration, not a complete model configuration. Do not
commit the resulting local configuration or its API key.

The selector recognizes these configurations:

| Display name | Configuration tag |
| --- | --- |
| Original | `iq2_xs` |
| Coder | `coder-iq1_m` |
| Quality (IQ3_S) | `iq3_s` |
| Swift 1.5 | `swift-iq2_xs` |
| Uncensored (experimental) | `uncensored-iq2_xs` |

Only installed, ready configurations with a tokenizer and the settings above are available. Uncensored is an
experimental configuration; its weights and control-vector setup are not provided by the selector. See
[model information](MODELS.md) for the native models and their requirements.

Start through `START-Models.bat` or Strata's process manager before switching native configurations. A manually
started or unrelated server is not owned by the switcher. Switching is available after generation finishes; it
releases the previous native model and starts the selected configuration. If startup fails, the worker attempts
to restore the previous configuration and reports the result. Saved chats remain on disk, but model processes
and their in-memory conversation caches do not survive a restart.

## Additional local providers

Prepare a model in an OpenAI-compatible server and open the additional-model form in About. Enter a display name,
base URL, model ID and context limit. The discovery control reads the server's model list. Saving a profile checks
that the requested model is listed before selecting it.

Base URLs must use HTTP and refer to `localhost`, `127.0.0.1` or `::1`. The server normalizes localhost to loopback,
ignores HTTP proxies and refuses redirects, embedded credentials and nonlocal addresses. For example, an already
running Ollama server uses `http://127.0.0.1:11434/v1`. Profiles currently connect without an upstream API key.

Set the context value at or below the model server's actual configured limit. Counts shown for additional models
are byte-based estimates, not their tokenizer's exact counts. Enable images only if that model and server support
image requests. Changing this checkbox does not add vision support to the model.

Profiles are saved in `.local/model-providers.json`. The global server selection starts with the native provider
after a restart. The page remembers its last additional choice when you return to that group. Selecting a model
affects requests handled by this Strata server, including other connected clients; it is not a per-browser model
process.

Ollama profiles request model release when switching away. The llama.cpp/Bonsai preset uses a router's model-unload
endpoint. A generic compatible server manages its own model memory; selecting another connection does not prove
that its RAM or VRAM has been released. Check the server's own status and logs. Bonsai requires its compatible
engine and router; importing its weights into an arbitrary server does not establish compatibility.

## API clients and feature limits

The OpenAI-compatible address remains `http://127.0.0.1:8080/v1`. When an additional provider is selected, request
model `active` or `strata` to follow it; the provider also accepts its registered ID or upstream model ID. Unknown
additional-provider model IDs remain errors. Native model names and configured aliases retain Strata's existing
behavior. Supply the Strata server's API key if it requires authentication.

Additional providers support OpenAI Chat Completions, streaming and caller-supplied tools. Strata's built-in MCP
tool execution and Anthropic adapter remain native-engine features. Additional provider selection does not extend
those features to the external server.

To stop the managed Strata server, wait for generation to finish and run `STOP-Models.bat`, or:

```text
python tools/start_models.py --stop
```

This deselects and requests release of the active additional model, then stops the managed Strata server. It does
not shut down every separately running model server on the PC.

## Developer checks

The selection, provider validation and launcher tests use local fixtures, without model downloads or a GPU:

```text
python -m unittest serve.test_model_switch serve.test_providers
python tools/test_start_models.py
node --test serve/web/test_model_selectors.cjs
```
