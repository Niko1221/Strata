# Models in the browser (opt-in)

Run `MODELS.bat` on Windows or `sh models.sh` on Linux. Python 3.10 or newer is
required; the launcher uses `.venv` when it exists, otherwise `py -3` or `python3`.
It opens a Models table at `http://127.0.0.1:8090`. The launcher stays available
while a model downloads, prepares or loads. Chat and API clients use the model
server on `http://127.0.0.1:8080`, as with the normal start scripts.

This is an alternative launcher. `START-HERE.bat`, `setup.sh`, the normal web app,
and every engine backend keep their existing behavior.

## Download and prepare

The table reads `MODELS` and `FAMILIES` from this checkout's `setup.py`. Only
regular installer choices are listed. Experimental sizes and OrcaRouter's
[manual compatibility workflow](ORCA.md) are linked separately. Importing setup
must succeed; the browser does not authorize an install using a fallback catalog.

Choose **Review download**. The plan shows the full model size, free disk space,
RAM estimate, license where supplied by setup, file location and exact command.
Choose **Download and set up** to run:

```
setup.py --setup --yes --no-start --family FAMILY --model SIZE --port 8080 --data-dir DATA
```

`DATA` is the existing data folder remembered by setup, pinned to the reviewed
plan. The GPU, backend, context, KV format, low-RAM mode and cache sizes are
selected by setup itself. No speed or quality tuning is added. New installs use
unattended setup's image default (off). **Review setup** on an installed model
runs official setup again; its generated settings can change just as when setup
is run in a terminal.

The page shows setup's current step, current file's download percentage and the
last output lines. **Stop download** stops the install; reviewing and downloading
the same model again resumes its partial files. Finished steps are handled by
setup, including its own file checks. If the launcher closes, setup continues;
reopening the launcher recovers its progress and finishes the lookup-path step.
The plan's disk estimate is conservative; setup performs its own remaining-byte
check before downloading. This interface does not change the downloader.

## Lookup table on an SSD

The original Qwen GSQ-RCO sizes and Coder share the identical second GGUF shard,
as used by the official installer. If a compatible installed config already
points to a table moved elsewhere (for example, an SSD), new installs keep that
location after setup succeeds. The launcher checks the shards' file sizes,
architecture and tensor directories first, writes the earlier config to `.bak`,
and changes only the lookup and explicit dense-shard paths. It does not move,
delete or rename a GGUF, and it does not change quantization or inference tuning.

This is metadata checking, not an additional full-file checksum: the table must
come from a trusted original Qwen installation. Swift, Unsloth and OrcaRouter use
their own tables. A mismatch keeps setup's original paths and reports the problem.

The table is part of the model, not a model selector. Switching loads another
model's own config and weights. The folder containing the lookup file is not a
place to put arbitrary models.

## Start and switch

Close a Strata server started from a normal run script once before using the
Models table to manage its lifecycle. The launcher refuses to stop a server it
did not start. Choose **Start** on an installed model, then **Open Chat**. To change
models, choose **Switch** on another installed model. The target files are checked
before the working model is stopped. Busy models are left running; wait for their
requests to finish and try again. Only one operation runs at a time.

The launcher starts the same `serve/server.py --engine strata --config ...`
command as a model's run script, preserving that config's settings. If loading a
new model fails, it attempts to restore the previous model once. Loading times
out after 10 minutes; the output remains available. **Stop model** unloads RAM
and VRAM and ends the server owned by this launcher. Closing the Models launcher
leaves its model server running, as with the existing MCP launcher.

State and logs live in `.strata-models/`, separate from `.strata-mcp/`. Keep one
Models launcher open per checkout. The launcher also refuses installs while the
MCP helper reports an active setup job.

## Local access and ports

The Models page binds only `127.0.0.1`. Management requests require its local
Host, same-origin JSON requests and a per-launch token. It accepts catalog ids
and reviewed plans, not shell commands or arbitrary file paths. It never changes
network access or API keys. Configs set to listen beyond localhost must be started
with their normal start script instead.

For different ports:

```
python tools/strata_models.py --port 8090 --model-port 8081 --open
```

The two ports must differ. If the model's config needs an API key, enter that key
in the normal chat page's About settings; the Models page does not display it.

## Validation

```
python -m unittest tools.test_strata_models tools.test_strata_mcp
node --check serve/web/models.js
```

The launcher tests use fake setup and server processes, without model downloads
or GPU inference. They cover official catalog/default delegation, download plans,
installation failures, disk rechecks, switching, busy requests, recovery of the
previous model, lookup-path preservation and local browser access checks.
