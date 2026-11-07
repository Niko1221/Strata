# Config snapshot

For every **machine class x model set x entry point** this records what the engine would be started with, and a test fails on any difference.

Nothing real is used: no GPU, no model file (the model files are empty), no engine (a text stub), no download (the drafter fetch is a recorder), no network,
no real user folder (every run has its own scratch HOME, data folder and install folder, removed afterwards).

## Run it

```
python -m pytest tests/snapshot -q -p no:cacheprovider      # the gate: about a minute; needs Python 3.11+ and Node 24
STRATA_SNAPSHOT_UPDATE=1 python -m pytest tests/snapshot     # after an intended change: rewrites golden/, then review the git diff of golden/
python -m tests.snapshot.generate --out DIR                  # write the snapshot of this tree somewhere else (to compare two branches)
python -m tests.snapshot.report --golden DIR_A --against DIR_B   # Markdown: where the entry points differ, what changed between A and B
```

`STRATA_SNAPSHOT_SKIP_APP=1` skips the two app entry points by name when Node is not installed (the test prints what it skipped).
The same command works on Windows, Linux and macOS; the golden files hold no path, user name or time of the machine that made them.

## What is simulated

* **machine classes** (`matrix.py`): NVIDIA all-resident (RTX PRO 5000 72 GB), NVIDIA 12 GB streaming (RTX 3060), DGX Spark (linux-aarch64, GB10),
  Strix Halo Windows (gfx1151), Radeon R9700 (gfx1201, Linux), Intel Arc A750 (SYCL), Mac M-series small (16 GB) and large (128 GB).
  A class sets the engine family, the card architecture, the card memory, the platform and the hardware report the Terminal's catalog and the app's host read.
* **model sets**: IQ3_S, UD-Q4_K_XL, UD-IQ4_XS-v31, DeepSeek (IST-DAS-2.00bpw).
* **entry points**: `setup`, `launch`, `app_plain`, `app_models`.

## Entry points

| entry | how it is driven | what is real | what is replaced |
|---|---|---|---|
| `launch` | `strata launch --model <set> --yes --no-tui` through `strata_cli.cli.main` | the whole command: settings, `Runner.run()`, the drafter step, config derivation and writing | the engine download, the model download (`ensure_model` returns the stand-in below), tuning, the server start, the drafter download |
| `app_plain` | `apps/host/server/native-control.mjs` `operation('load', {model})` in Node (`app_driver.mjs`) | the host's whole Load code, and Control's `drafting plan` / `drafting fetch` (run through `control_shim.py` on the same simulated machine) | the installer process (catalog, hardware), Control's model list / engine / profile save / start, the drafter download |
| `app_models` | the same with `values: {drafting: 'auto'}` (what the Models screen's Load passes) | same | same |
| `setup` | **not driven today** | | see below |

For `launch`, `app_plain` and `app_models` the last step is the same code the server runs when it starts the engine: `serve.server.engine_args(cfg, exe)` and
`serve.server.child_env(cfg)`.  The record has the run config the entry point left (`config_file`), the argument list (`engine.argv`), the variables the start adds to
the environment (`engine.env_set`), the notes the start prints, the drafter downloads that were asked for (`fetches`) and, for the app, the calls the host made to Control.

### The stand-in for setup's output, and why `setup` is blocked

`setup.py` writes the model's run config in one 1,300-line function (`run()`, setup.py lines ~10100-11400) that on the way probes the GPU, resolves or builds an engine,
downloads and hashes the model shards, packs the model and (DeepSeek) runs a smoke test.  It cannot run without a model, an engine and a GPU, and making it
run would change product code, which step 0 must not do.  So the entry points that start from an installed model all start from the same **stand-in**
(`World.base_config`, rebuilt from setup.py's own rules per class and set), and differences between entry points can never come from it.
The seam that would let `setup` join the table: a pure function `config_for(args, hardware_report, files) -> run config` in `strata_control`, called by `run()` just before
`write_setup_config()` (a move of existing lines, no behaviour change).

## Adding a class or a set

Add a row to `matrix.py`, run the update command, review the new golden file.  A golden file for something not in the matrix, or a matrix cell without one, fails
`test_every_golden_file_belongs_to_the_matrix`.

## CI

No workflow is added in step 0 (step 2.5/2.7).  A job needs `actions/setup-python` (3.11+, `pip install -r requirements.txt`) and `actions/setup-node` (24), then
`python -m pytest tests/snapshot -q`.

## Known limits

* The record is of a model that is installed and has **no drafter on disk yet**: the state a first Load / launch meets.  The state "drafter already downloaded" is not in the matrix.
* `engine_args` and `child_env` are called as the server calls them, but the server's other start-time steps (component engine resolution, vision reader choice, effort end arguments) are not.
* Host-OS code paths (a Windows-only branch) run on the host the test runs on; the golden files were checked to be identical when made on Windows and on Linux.
