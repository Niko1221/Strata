# Running from a config

After installing with `./setup.sh` (Windows: `START-HERE.bat`), you can keep everyday launch settings in YAML
and start with `make run`. The launcher uses the engine, model paths and GPU choices saved by setup. It starts
the server directly; it does not check for updates, compile, install packages or download models.
Starting restores the model's saved draft-vocabulary subset from the shipped files; a custom subset is kept.

From the Strata folder:

```bash
make models                 # names of models already installed on this PC
make init                   # create strata.yaml; refuses to overwrite an existing file
# Edit strata.yaml, then:
make check                  # validate settings and engine/tokenizer files without loading the GPU
make run                    # Ctrl+C stops the server and engine
```

Make is optional. The same commands work with `.venv/bin/python run.py models`, `init`, `check` or `run`.
On Windows use `.venv\Scripts\python.exe run.py run`. If your environment lives elsewhere, set
`make run PYTHON=/path/to/python`. Python 3.10+ and the packages installed by setup are needed.

For a new installation without starting the server yet:

```bash
./setup.sh --yes --model IQ2_XS --context 32768 --no-start
make init
make run
```

## Pick a model and context

`strata.example.yaml` lists the settings. A small launch file looks like this:

```yaml
model: iq2_xs
context: 32768
host: 127.0.0.1
port: 8080
open_browser: false
```

`model` is a name from `make models`, such as `q2_0`, `coder-q2_0` or `swift-iq3_s`. It selects that model's
`strata-<model>.json` in the Strata folder. To use a config elsewhere, replace `model` with
`config: /path/to/strata-coder-q2_0.json`. Relative config paths are relative to the YAML file.
Choosing a model that is not installed gives an error listing the installed names. Use setup to install another
model or enable its image encoder.

`context` is the full token limit: prompt plus answer. Omit it to keep setup's size. `kv` picks `fp16`, `int8`,
`q4_0` or `k8v4`; `vram_reserve_mib` sets how much VRAM the engine leaves free. Omitted engine settings keep the
installed config's values, including its KV streaming and RAM budget choices. A larger context needs more
memory; `make check` checks the settings and files, not whether that context fits your GPU or RAM. Run setup again
to have it choose memory settings for a different context. Above 262,144 tokens, the launch requires an existing
RoPE configuration whose factor covers the requested context; configure it with
`./setup.sh --setup --context 393216 --no-start` first. See the context extension notes in [DETAILS.md](DETAILS.md).

Use separate files for different runs:

```bash
make init CONFIG=coder.yaml
make run CONFIG=coder.yaml
```

Both commands also accept `--config coder.yaml` when using `run.py` directly.

## Access from other devices

Local launches default to `127.0.0.1`, even if the installed JSON config previously used a network address.
The server listens on IPv4; `make check` rejects IPv6 addresses.
To allow other devices, set these keys in `strata.yaml`:

```yaml
host: 0.0.0.0
api_key_env: STRATA_API_KEY
```

Then start with a key in your environment:

```bash
export STRATA_API_KEY='your-secret'
make run
```

The launch refuses a non-local address without a non-empty API key. `api_key_env` can name another environment
variable; when specified, that variable must be set. Without `api_key_env`, `STRATA_API_KEY` is used if set,
otherwise `api_key` in YAML or the installed JSON config supplies the key. An empty key environment variable gives
an error; unset it for a local launch without a key. The launcher prints whether a key is
set and keeps its value out of its output. `strata.yaml` and its backup are ignored by Git and Docker builds;
add any custom launch filename to your own ignore rules if it holds a key.

Other devices connect to `http://<your PC's LAN address>:8080/v1` and send the same key as their API key.
The web app at `http://<your PC's LAN address>:8080/` asks for it too. The PC's firewall may need to allow the port.
More on [network access and API keys](DETAILS.md#using-it).

## Server settings and updates

The launch also accepts `sampling` (`temperature`, `top_p`, `top_k`, `min_p`), `lazy_load`, `idle_unload_s`,
`fit_max_tokens`, `reasoning_budget_tokens`, `aliases`, `anthropic_thinking`, `effort_position`,
`engine_silence_s` and `api_monitor`, using the same names as the server's JSON config.
For example:

```yaml
lazy_load: true
idle_unload_s: 300
fit_max_tokens: true
sampling:
  temperature: 0.7
  top_p: 0.95
```

The web app's Settings view reads the combined settings. Saving there writes the changed settings to the launch
YAML, keeps the earlier YAML as `.bak` and leaves the setup JSON alone. That save rewrites YAML formatting and
comments. For the web-editable settings, a `null` value removes an inherited setting so the server default applies.
Shared Chat settings are saved beside the YAML file. Restart to use changes to launch settings.

Continue using `./update.sh` (Windows: `UPDATE.bat`) for updates, and setup for installation and hardware changes.
Setup reuses complete local GGUF files by default. To replace a model's files, use
`./setup.sh --model IQ2_XS --force-download --yes --no-start`; see [existing model files](INSTALL.md).
Advanced engine options, GPU selection, MCP servers and network policies stay in the installed JSON config;
the launch keeps them. Existing run scripts and `./setup.sh` still work.
