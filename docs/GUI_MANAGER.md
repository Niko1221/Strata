# Strata Manager

An **optional, local GUI** for managing a Strata installation.  It manages - it does not replace - Strata.

The Manager is a thin layer **on top of the existing Strata setup system**:

```
Strata core (engine + serve/server.py)
      ↑
setup.py  (the existing setup / config / model-discovery logic)
      ↑
gui/manager.py  (the Manager: a small local HTTP server + supervisor)
      ↑
your browser   (gui/web/*: a dark, dependency-free page)
```

Everything the Manager knows about models, context, Vision, Low-RAM and the config format comes from
`setup.py` (imported, never duplicated).  Saving writes the **same `strata-*.json` files** that `setup.py`
writes.  Starting Strata runs the **same `serve/server.py` + config command** that the generated
`run-*.bat` scripts run.  There is no second engine, no second settings system, no downloads.

## What it does

- **Installed Models** — every `strata-*.json` in the Strata folder, most recently used first, with the
  model name, GGUF paths, quantization, context, KV, Vision, low-RAM, GPU and current/default status
  (only facts Strata itself knows, read with `setup.choices_from_config`).
- **Model selection** — pick any installed model without downloading anything.
- **Context** — presets (8K/16K/32K/64K/128K/256K/384K/512K) or any custom value.  Past the trained
  262144 the Manager applies the same automatic `yarn` rope scaling with the derived factor that
  `setup.py` uses (`setup.derived_factor`).
- **Vision** — Off / GPU / CPU, toggling the same `cfg["vision"]` structure and `--vision` /
  `--vram-reserve-mib` args that setup writes.  If the encoder was never installed, the Manager says so
  and leaves the job to `SETUP.bat --vision` (adding it downloads ~1 GB, which the Manager never does).
- **Low RAM** — the existing `--resident-experts` / `--mmap-experts` mode, choosing the variant with
  `setup.low_ram_resident` (the same decision setup makes).
- **Custom GGUF** — browse to a folder of existing GGUF shards; the Manager detects the family and size
  from the file names (`setup.py`'s own naming).  *Preparing* a new one runs the existing
  `setup.py --family … --model … --gguf-dir DIR --no-start --yes` pipeline in the background (its log is
  shown), so Strata's own packing is used.  Nothing is copied, moved, or downloaded by the Manager.
- **Advanced Settings** — a few safe, genuinely Strata-wide settings: KV cache precision (8-bit, 4-bit,
  K8V4 — shown from 8K context up), GPU (single-card choice; a multi-GPU layer split is detected and left
  to `SETUP.bat --gpus`), port, listen host, API key.
- **Save Configuration** — atomic (temp file + rename, so a crash cannot corrupt a config) with a dated
  `.bak` of the previous version kept (the last 5), exactly like setup's own backups.  Existing configs
  stay compatible.
- **Start / Stop / Restart** — a tiny supervisor launches the Strata server detached
  (`serve/server.py --engine strata --config … --port …`), remembers its PID in `logs/manager.json`, and
  checks the port so it can never launch a second instance (Strata's server itself also refuses a busy
  port).  **Stop** closes the server's process tree — the same outcome as closing Strata's console window.
  The UI never looks frozen: Stop/Restart switch to a visible Stopping/Restarting indicator with the real
  phase ("Gracefully shutting down the engine", measured "… 4.2s") from the backend's lifecycle state, and
  confirm with "Stopped in Xs".  Graceful shutdown is always tried first; **Force Stop** terminates the
  process tree immediately (separate button, shown during a stop), because graceful shutdown on Windows has
  no real external signal — the engine's own QUIT path still gets its grace time.
- **System** — GPU names, VRAM total/used and RAM total/used (measured, from `nvidia-smi` / the OS;
  nothing is estimated or invented).

## Launching

**Windows (double-click):** `START-MANAGER.bat`

**Linux (Arch / Omnarchy and others, double-click or terminal):** `./start-manager.sh`

**Anywhere:** (from the Strata folder, with its `.venv` Python)

```
.venv\Scripts\python.exe gui\manager.py      # Windows
.venv/bin/python gui/manager.py              # Linux
```

Then open http://127.0.0.1:8275/ (it opens itself).  The Manager only listens on 127.0.0.1.

Options: `--port N` (another local port), `--root DIR` (manage another Strata folder; mainly for tests),
`--no-browser`.

## Cross-platform

The Manager is one codebase for Windows and Linux - the web UI and all config/logic are shared.

- **Paths**: `pathlib` everywhere; no drive letters, no backslashes, no `C:\...` in the source.  A Linux
  model folder such as `/mnt/Storage/Model` is handled exactly like `E:\Model\Q2_0`.  (The generated
  `strata-*.json` configs contain absolute machine paths by design - that is Strata's own format; the
  source never hardcodes any.)
- **No Windows-only APIs in the core**: nothing depends on PowerShell or `.bat`.  The only platform code is
  the small launcher adapter (see below).
- **Browser**: `webbrowser` (opens your default browser on both platforms).
- **Launch/stop/restart** is abstracted in `gui/launcher.py` (supervisor: state, port checks, status) +
  `gui/platforms/` (the two operations that differ per OS):

  | platform | spawn | stop |
  |----------|-------|------|
  | `windows.py` | `CREATE_NEW_CONSOLE` with a console window **hidden from birth** (`STARTUPINFO` SW_HIDE) — the engine inherits it instead of popping a new blank console terminal of its own (a console child of a console-less parent gets a visible window; this is what the "extra blank terminal" was) | graceful attempt first (~3 s), then close the process tree (`taskkill /T /F`) — Strata's "close the window" outcome; **Force Stop** skips straight to `taskkill /T /F` |
  | `linux.py` | new session (`start_new_session=True`) | `SIGTERM` to the server's process group (serve/server.py handles it gracefully - QUIT to the engine), `SIGKILL` after a grace |

- **Startup convenience** (both just run the same `gui/manager.py`): `START-MANAGER.bat` (Windows) and
  `start-manager.sh` (Linux).

## How it relates to START-HERE.bat / SETUP.bat

|                          | START-HERE.bat / SETUP.bat / setup.sh | Strata Manager                 |
|--------------------------|----------------------------------------|--------------------------------|
| Installs Python, engine, model files | yes                             | no (never downloads)           |
| Installs / repairs a model           | yes (SETUP.bat / setup.sh)      | no — points to the setup       |
| Edits an installed model's config    | interactive questions           | yes (Save)                     |
| Starts / stops the model             | foreground console window       | detached, Start/Stop/Restart   |

`START-HERE.bat`, `SETUP.bat` and `setup.sh` are **untouched** and remain the source of truth for
installing, repairing, and updating models and the engine.  The Manager never changes how those run.

## Where configuration is stored

- Installed models: `strata-<name>.json` in the Strata folder (setup's own format — the Manager writes it).
- Backups of each edited config: `strata-<name>.json.bak-<timestamp>` (last 5 kept).
- The Manager's own runtime state (which server it launched): `logs/manager.json`.
- The Strata server's output when launched by the Manager: `strata-<name>.serve.log`.

All of these are gitignored machine data.

## Platform notes

- Run `START-HERE.bat` / `setup.sh` once first if `.venv` is missing.
- **Stop** ends the whole Strata process tree — on Windows a graceful attempt first (~3 s; the fast
  `taskkill /T /F` follows, so a normal Stop takes about 3 s) and on Linux a graceful SIGTERM to the
  server's process group (`serve/server.py` QUITs the engine) with a SIGKILL fallback.  If a stop is taking
  too long, **Force Stop** terminates the tree immediately.
- On Windows the launched server runs in a **hidden** console, so no extra blank terminal flashes next to
  the browser while the engine/server run (their output still lives in the log files the Manager tails).
- If Strata is running from its own console window (`START-HERE.bat` / `./setup.sh`), the Manager shows it
  as **running** but its **Stop** button only controls what the Manager itself started — close the other
  window instead.
- A VRAM/GPU reading appears only when `nvidia-smi` works (an NVIDIA driver is required, as for Strata).
- Custom GGUF folders can be anywhere: `E:\Model\Q2_0`, `/mnt/Storage/Model`, `~/models/...`.

## Limitations (first version)

- Vision can only be toggled per installed model; installing the encoder itself is SETUP.bat's job.
- Multi-GPU layer splits are detected and shown, but editing them is left to `SETUP.bat --gpus`.
- Custom GGUF *prepares* a new model through the existing setup pipeline (its log streams in the page);
  the heavy one-time pack step for some sizes can take a while and needs free disk.
- The Manager does not tune/calibrate the engine (`START-HERE.bat --calibrate` does) and does not
  estimate RAM/VRAM needs — it only shows measured values.
- The Manager supervises only servers it started itself; servers started from a console are left alone.
- HTTP only, localhost: no encryption or auth is needed or offered (127.0.0.1 only).

## Development

- Backend (config edits, discovery, HTTP, folder picker): `gui/manager.py`
- Supervisor (Start/Stop/Restart, state, log tail): `gui/launcher.py`
- OS adapters (the only platform code): `gui/platforms/windows.py`, `gui/platforms/linux.py`
- Frontend (no dependencies): `gui/web/index.html`, `gui/web/app.css`, `gui/web/app.js`
- Custom-GGUF field state (the field is the source of truth; no DOM): `gui/web/gguf_state.js`,
  tested with `node gui/web/test_gguf_state.mjs` (the unittest suite runs it when node is present)
- Launchers: `START-MANAGER.bat` (Windows), `start-manager.sh` (Linux)
- Tests: `python -m unittest gui.test_manager` (temp folders and a real local HTTP round-trip; no GPU,
  no network, no user files are touched).  The supervisor is tested against a fake platform adapter, so the
  same suite passes on Windows and Linux.