"""What `strata launch` does after the choices are made: get the engine and the model, write the run config, tune, start, open access.

Every step reports plain-word events through one callback (`emit`), so the TUI (progress screen), `--no-tui` (printed lines) and the
tests see the same thing. Nothing here prints or reads the keyboard. The steps reuse what is already in the Terminal:

* the engine comes from the signed feed (`strata update`, the one explicit step that downloads);
* the model comes from setup.py in the app's own installer mode (STRATA_CONTROL_INSTALL), the same job the desktop app runs;
* the run config the engine uses is a private copy of the installed one with this launch's choices applied; the profile
  `launch-<name>` (the shared profiles store) points at it, and carries host, port and parallel;
* tuning is `strata tune auto`, its result lives in the shared presets store and is applied at every start;
* start and stop are `strata runtime start|stop` (a background service: it keeps serving after this terminal closes).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from strata_cli.launch import access as acc
from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Catalog, CatalogUnavailable

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = os.name == "nt"


class LaunchError(RuntimeError):
    """A step could not be done. `text` says what it means; `next` says what to do."""

    def __init__(self, text: str, next: str = "", code: int = 6):
        super().__init__((text + " " + next).strip())
        self.text, self.next, self.code = text, next, code


class Cancelled(LaunchError):
    def __init__(self):
        super().__init__("Stopped. Nothing was changed on your computer beyond files already downloaded.", "", 130)


@dataclass
class Event:
    stage: str                      # engine | model | config | tune | start | access | done
    text: str
    pct: float | None = None
    level: str = "info"             # info | warn | error
    data: dict = field(default_factory=dict)


@dataclass
class Result:
    profile: str
    urls: list[str]
    key: str | None
    tune_word: str | None
    notices: list[str]
    config: str
    tunnel: str | None = None
    serving: bool = True                   # False: the model is ready but nothing was started (no engine yet)
    engine_problem: str | None = None      # the plain reason there is no engine, when serving is False
    final_text: str = ""                   # what to tell the person when serving is False


NO_ENGINE_NEXT = ("Get the engine that matches this computer's graphics card (from whoever sent you Strata, or with `strata update --feed <address>`), "
                  "then run `strata launch` again. Nothing was downloaded, and any model files you already have are kept.")

STAGES = ("engine", "model", "config", "tune", "start", "access")

NO_ENGINE_STEP = ("run `strata update --feed <address>` with the update source you were given, or ask whoever sent you Strata for the engine "
                  "that matches this computer's graphics card, then run `strata launch` again")


def no_engine_warning(reason: str) -> str:
    """The plain warning shown before a model is downloaded on a computer with no engine to run it."""
    reason = reason.strip().rstrip(".")
    return (f"Strata couldn't find an engine for this computer ({reason}). The model will download, but it may not run until you install "
            f"the right engine (next step: {NO_ENGINE_STEP}).")


def no_engine_final(reason: str, name: str = "") -> str:
    reason = reason.strip().rstrip(".")
    return (f"The model is ready, but Strata can't start it yet: it couldn't find an engine for this computer ({reason}). "
            f"Next step: {NO_ENGINE_STEP}. The model files stay on this computer.")


def tune_words(result: dict | None, code: int | None = None) -> tuple[str, str]:
    """(short word, plain sentence) for a tuner result: faster / already fastest / busy PC / failed / stopped."""
    if not result:
        return "failed", "The tuner did not finish, so the standard settings run. Nothing was changed."
    out = result.get("outcome")
    if out == "improved":
        gain = result.get("gain_pct")
        return "faster", (f"Done: answers are about {gain:.0f}% faster with the tuned settings." if isinstance(gain, (int, float))
                          else "Done: the tuned settings are faster.")
    if out == "default_confirmed":
        return "already fastest", "Your computer already runs at its best with the standard settings. Nothing needed changing."
    if out == "inconclusive":
        return "busy PC", ("The computer was too busy to tell a real gain from noise, so the standard settings run. "
                           "Close other programs and tune again later (strata tune auto).")
    if out == "cancelled":
        return "stopped", "Tuning was stopped. Your settings were not changed."
    return "failed", (result.get("message") or "Tuning could not run here.") + " The standard settings run."


# ----------------------------------------------------------------------------------------------- run config

def read_config(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def desc_matches(desc: dict, s: S.LaunchSettings) -> bool:
    """Does the installed model's config already carry the context, memory format and picture choice of these settings?
    (The graphics cards are not part of it: they go into each launch's own config, see apply_cards.)"""
    if s.context and desc.get("context") != s.context:
        return False
    want_kv = s.kv or ("int8" if (s.context or 32768) > 8192 else "fp16")
    if desc.get("kv") not in (None, want_kv) and (s.context or 32768) > 8192:
        return False
    images = desc.get("images") or "off"
    if s.vision is None:
        return True                                          # no preference yet: the run settles it before it compares
    if (s.vision == "off") != (images == "off"):
        return False
    if s.vision in ("gpu", "cpu") and images != s.vision:
        return False
    return True


def files_present(cfg: dict) -> list[str]:
    """The model files a run config names that are missing (empty list: all there)."""
    missing = []
    for tok in cfg.get("args") or []:
        if isinstance(tok, str) and tok.lower().endswith(".gguf") and not Path(tok).exists():
            missing.append(tok)
    for flag in ("--pack", "--mtp"):
        args = [str(a) for a in cfg.get("args") or []]
        if flag in args and args.index(flag) + 1 < len(args) and not Path(args[args.index(flag) + 1]).exists():
            missing.append(args[args.index(flag) + 1])
    vision = cfg.get("vision")
    if isinstance(vision, dict) and vision.get("mmproj") and not Path(str(vision["mmproj"])).exists():
        missing.append(str(vision["mmproj"]))                 # the picture reader's file: setup fetches it again
    return missing


def apply_choices(cfg: dict, s: S.LaunchSettings) -> dict:
    """A copy of the installed run config with this launch's context, memory format and picture choice on its engine command, so the command
    the Advanced step shows is the one these choices run (the installed config only holds the choices it was installed with). Only the command
    preview and the comparison use this; the start itself goes through setup, which writes the same values."""
    import copy
    cfg = copy.deepcopy(cfg)
    args = [str(a) for a in cfg.get("args") or []]

    def put(flag: str, value: str) -> None:
        if flag in args:
            i = args.index(flag)
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                args[i + 1] = value
                return
            args.insert(i + 1, value)
        else:
            args.extend([flag, value])

    if s.context:
        put("--max-context", str(int(s.context)))
    kv = s.kv or ("int8" if (s.context or 32768) > 8192 else "fp16")
    if "--kv" in args or s.kv:
        put("--kv", kv)
    if s.vision == "off":
        args = [a for a in args if a != "--vision"]
        cfg.pop("vision", None)                              # the picture reader's own settings: the server adds --vision for them
    elif s.vision and "--vision" not in args:
        args.append("--vision")
    cfg["args"] = args
    return cfg


def apply_cards(cfg: dict, gpus: list[int] | None) -> None:
    """The graphics cards of THIS launch go into its run config, whatever card the model was installed with (no reinstall): one card is a number,
    several are a list with the automatic layer split. None: the installed config's own choice stays."""
    if not gpus:
        return
    gpus = [int(g) for g in gpus]
    cfg["gpu"] = gpus[0] if len(gpus) == 1 else gpus
    cfg["gpus_asked"] = True
    if len(gpus) > 1:
        cfg["layer_split"] = cfg.get("layer_split") or "auto"
    else:
        cfg.pop("layer_split", None)


def cards_for_launch(s: S.LaunchSettings, catalog, ram_gb: float | None = None) -> list[int] | None:
    """The cards this launch uses: the ones named (--gpu / --gpus, the Cards step), else the recommended ones by the shared fit rule
    (strata_control/gpu_fit.py) when the computer has more than one card that counts, else None (one card: the installed config's own)."""
    if s.gpus:
        return [int(g) for g in s.gpus]
    try:
        from strata_control import gpu_fit
        comp = catalog.computer()
        if len(comp.usable_gpus) < 2:
            return None
        row = next((r for r in catalog.rows() if (r.family, r.model) == (s.family, s.model)), None)
        mem = float(row.memory_gb) if row is not None and row.memory_gb else 44.0
        return gpu_fit.recommended_cards(mem, comp.ram_gb, comp.gpus, comp.engine_vendor) or None
    except Exception:                                        # noqa: BLE001 - never block a launch over the choice of cards
        return None


def derive_config(base: dict, s: S.LaunchSettings, *, bool_flags: frozenset[str], log: str | None = None, gpus: list[int] | None = None) -> dict:
    """The private run config of this launch: the installed one plus the choices that are not setup's (host, port, key, cards, advanced)."""
    from strata_control import tune_knobs
    cfg = json.loads(json.dumps(base))
    cfg["host"] = acc.host_for(s.access)
    cfg["port"] = int(s.port)
    if s.api_key:
        cfg["api_key"] = s.api_key
    else:
        cfg.pop("api_key", None)
    cfg.setdefault("exe", "component:strata-engine")
    cfg["open_browser"] = False                              # strata launch never opens a browser by itself (only `--open` does, after the start)
    if isinstance(cfg.get("args"), list):
        cfg["args"] = [a for a in cfg["args"] if a != "--open"]
    if log:
        cfg["log"] = log
    apply_cards(cfg, gpus)
    if s.extra:
        a_changes, e_changes = S.as_argv_changes(s.extra, bool_flags)
        cfg["args"] = tune_knobs.apply_to_argv([str(x) for x in cfg.get("args") or []], a_changes)
        if e_changes:
            cfg["env"] = tune_knobs.apply_to_env(dict(cfg.get("env") or {}), e_changes)
    return cfg


def write_private(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".launch-cfg-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write((json.dumps(data, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    if WINDOWS:
        try:
            from strata_control import private_files
            private_files.restrict_to_current_user(path, directory=False)
        except Exception:                                    # noqa: BLE001
            pass


def launch_dir() -> Path:
    from strata_control import paths
    d = paths.state_base() / "launch"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------------------------- the runner

class Runner:
    def __init__(self, s: S.LaunchSettings, *, catalog: Catalog | None = None, emit: Callable[[Event], None] | None = None,
                 cancel: threading.Event | None = None, feed: str | None = None, python: str | None = None,
                 components_dir: str | None = None, hardware: dict | None = None,
                 confirm: Callable[[str, str], bool] | None = None, assume_yes: bool = False):
        self.s = s
        self.confirm = confirm                # asked (title, text) before a model is downloaded with no engine; True = go on
        self.assume_yes = assume_yes          # --yes: no questions
        self.engine_problem: str | None = None
        self._catalog = catalog
        self.emit_cb = emit or (lambda e: None)
        self.cancel = cancel or threading.Event()
        self.feed = feed
        self.python = python or sys.executable
        self.components_dir = components_dir
        self.hardware_override = hardware
        self.children: list[subprocess.Popen] = []
        self.notices: list[str] = []
        self.tune_result: dict | None = None
        self.config_path: Path | None = None

    # ---- small helpers
    def emit(self, stage: str, text: str, pct: float | None = None, level: str = "info", **data: Any) -> None:
        self.emit_cb(Event(stage, text, pct, level, data))

    def check_cancel(self) -> None:
        if self.cancel.is_set():
            raise Cancelled()

    @property
    def catalog(self) -> Catalog:
        if self._catalog is None:
            try:
                self._catalog = Catalog(hardware=self.hardware_override)
            except CatalogUnavailable as e:
                raise LaunchError(str(e), "", 5) from None
        return self._catalog

    @property
    def name(self) -> str:
        return S.slug(self.s.name or (self.s.model or "model") + "-" + S.context_label(self.s.context).lower())

    @property
    def profile(self) -> str:
        return "launch-" + self.name

    def _env(self) -> dict[str, str]:
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        env.pop("STRATA_NO_ACQUIRE", None)                  # launch is the explicit step that may download
        if self.components_dir:
            env["STRATA_COMPONENTS_DIR"] = self.components_dir
        return env

    def _spawn(self, argv: list[str], *, env: dict | None = None, cwd: str | None = None) -> subprocess.Popen:
        kw = {"creationflags": subprocess.CREATE_NO_WINDOW} if WINDOWS else {}
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", env=env or self._env(), cwd=cwd, bufsize=1, **kw)
        self.children.append(p)
        return p

    def _stop_children(self) -> None:
        """End only the processes this runner started (by the handles it holds)."""
        for p in self.children:
            if p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=15)
                except Exception:                            # noqa: BLE001
                    try:
                        p.kill()
                    except Exception:                        # noqa: BLE001
                        pass

    def _cli(self, *args: str) -> list[str]:
        return [self.python, str(ROOT / "strata.py"), *args]

    # ---- 1. the engine
    def ensure_engine(self) -> None:
        self.emit("engine", "Checking the engine for this computer…")
        from strata_control import engine_component as ec
        root = Path(self.components_dir) if self.components_dir else None
        report = ec.status(root)
        if report.get("usable"):
            eng = report["engine"]
            self.emit("engine", f"Engine {eng['version']} ({eng['backend']}) is installed.", 100)
            return
        feed = self.feed or os.environ.get("STRATA_UPDATE_FEED")
        self.emit("engine", "No engine is installed yet. Getting the one for your graphics card…", 0)
        argv = self._cli("update", "--no-control", "--output", "jsonl")
        if feed:
            argv += ["--feed", feed]
        if self.components_dir:
            argv += ["--components-dir", self.components_dir]
        env = self._env()
        for key, value in self.s.extra.items():              # the engine's own environment settings (device selectors, ...) apply to its test start too
            if not key.startswith("--") and value.lower() not in ("off", "unset", "none"):
                env[key] = value
        p = self._spawn(argv, env=env)
        last = None
        tail: list[str] = []
        for line in p.stdout:                                # type: ignore[union-attr]
            self.check_cancel()
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                tail.append(line)
                continue
            if ev.get("event") == "progress":
                self.emit("engine", f"Downloading the engine… {ev.get('percent', 0)}%", float(ev.get("percent", 0)))
            elif ev.get("event") in ("phase", "waiting"):
                self.emit("engine", str(ev.get("message") or ""))
            last = ev
        code = p.wait()
        if code == 0:
            # `strata update` exits 0 when the feed simply has no engine for this PC (or only one for another card family). That is not an
            # engine: refuse here, before the model (tens of GB) is touched.
            after = ec.status(root)
            if not after.get("usable"):
                self.no_engine(self._no_engine_words(last, after))
                return
        if code != 0:
            why = ""
            if isinstance(last, dict):
                why = str((last.get("error") or {}).get("message") or last.get("message") or "")
                if not why:
                    for res in last.get("results") or []:
                        if isinstance(res, dict) and res.get("status") == "failed" and res.get("error"):
                            why = str(res["error"])
                            break
            why = why or (tail[-1] if tail else "")
            raise LaunchError("The engine could not be installed." + (f" {why}" if why else ""),
                              "Check your internet connection, or point Strata at an update source with --feed, then try again.", 6)
        self.emit("engine", "The engine is installed.", 100)

    def no_engine(self, reason: str) -> None:
        """There is no engine for this computer and none to get. This is a warning, not a refusal: the model may still be downloaded, because
        the engine can be installed afterwards. Ask first (the answer is yes unless the person says no; --yes skips the question) when a
        download is about to happen; otherwise just remember the problem for the final message."""
        self.engine_problem = reason
        warning = no_engine_warning(reason)
        self.emit("engine", warning, 100, level="warn")
        if self.assume_yes or self.confirm is None:
            return
        try:
            gb = self.download_gb()
        except Exception:                                    # noqa: BLE001
            gb = 0.0
        if gb < 0.05:
            return                                           # nothing would be downloaded: no question, only the final message
        if not self.confirm("No engine for this computer yet", warning + f" Download the model now (about {gb:.0f} GB)?"):
            raise LaunchError("Stopped. Nothing was downloaded.", "Install the engine first, or run this again and answer yes.", 130)

    @staticmethod
    def _no_engine_words(last: object, report: dict) -> str:
        """The one plain sentence for "no usable engine for this PC": the update's own note (the wrong-card-family line, or "the feed has no
        engine package for this PC"), else the engine report's plain notice. Never the raw list of reasons."""
        note = ""
        if isinstance(last, dict):
            for res in last.get("results") or []:
                if isinstance(res, dict) and res.get("component") == "engine":
                    note = str(res.get("note") or "").strip()
        if not note:
            note = str(report.get("notice") or "").strip()
        if not note or note.startswith("No engine for this PC was found"):
            note = "There is no engine for this computer's graphics card in the update source."
        return note

    # ---- 2. the model
    def resolved_vision(self) -> str:
        """The picture setting of this launch: the chosen one, or (nothing chosen) the default for this model and engine (catalog.picture_default)."""
        s = self.s
        if s.vision is None:
            row = next((r for r in self.catalog.rows() if (r.family, r.model) == (s.family, s.model)), None)
            s.vision = self.catalog.picture_default(bool(row.images) if row is not None else False)
        return s.vision

    def vision_word(self) -> str:
        """What setup is told for the picture choice. "on" is setup's "yes": setup (Control mode) picks the CPU encoder on a card under 24 GB
        and the GPU encoder from 24 GB (any make of card), and downloads the pinned picture-reader file (mmproj) with the model."""
        s = self.s
        self.resolved_vision()
        if s.vision == "off":
            return "no"
        return "yes" if s.vision == "on" else s.vision

    def check_pictures(self, path: Path) -> None:
        """Pictures were asked for: say so in plain words when the installed settings still have them off."""
        s = self.s
        if s.vision == "off":
            return
        try:
            desc = self.catalog.strata.describe_config(path)
        except Exception:                                    # noqa: BLE001
            return
        if (desc or {}).get("images") in (None, "off"):
            note = ("Pictures could not be switched on: this engine does not carry the picture reader, or this model cannot read pictures, "
                    "so it will answer text only.")
            self.notices.append(note)
            self.emit("model", note, 100, level="warn")

    def setup_args(self) -> list[str]:
        cat, s = self.catalog, self.s
        fam, model = s.family, s.model
        args = ["--yes", "--no-start", "--family", str(fam), "--model", str(model)]
        ctx = s.context
        if ctx:
            args += ["--context", str(ctx)]
        args += ["--vision", self.vision_word()]
        if s.kv:
            args += ["--kv", s.kv]
        if fam in ("qwen", "coder"):
            args += ["--experimental-speed-projection", "off"]
        data = s.data_dir or self.existing_data_dir()
        if data:
            args += ["--data-dir", str(data)]
        if s.model_path:
            p = Path(s.model_path)
            args += ["--gguf-dir", str(p if p.is_dir() else p.parent)]
        if s.gpus:
            args += ["--gpus", ",".join(str(g) for g in s.gpus)] if len(s.gpus) > 1 else ["--gpu", str(s.gpus[0])]
        if sys.platform == "darwin":
            # a Mac's setup (Control mode, strata_control/mac_fit.py) takes fewer options (no --kv, --gpu, speed projection)
            keep = {"--yes", "--no-start", "--family", "--model", "--context", "--vision", "--data-dir", "--gguf-dir"}
            out, i = [], 0
            while i < len(args):
                if args[i] in keep:
                    out.append(args[i])
                    if args[i] not in ("--yes", "--no-start") and i + 1 < len(args):
                        out.append(args[i + 1])
                    i += 2 if args[i] not in ("--yes", "--no-start") else 1
                else:
                    i += 2 if i + 1 < len(args) and not args[i + 1].startswith("--") else 1
            args = out
        return args

    def installed_state(self) -> tuple[Path | None, dict | None]:
        tag = self.catalog.tag_of(self.s.family, self.s.model)
        path = self.catalog.installed_config(tag)
        if path is None:
            return None, None
        desc = self.catalog.strata.describe_config(path)
        return path, desc

    def existing_data_dir(self) -> Path | None:
        """The data folder to give setup when the person chose none: the folder the model already lives in (its run config names its files),
        else the folder the desktop app uses, so a helper file that is missing is fetched next to the rest, never into a second folder.
        None: setup's own default is right."""
        s = self.s
        try:
            if s.model_path:
                from strata_control import discovery
                first = next(iter(sorted(Path(s.model_path).glob("*.gguf"))), None) if Path(s.model_path).is_dir() else Path(s.model_path)
                found = discovery.prepared_beside(first)["data_dir"] if first else None
                return Path(found) if found else None
            if s.family and s.model:
                path = self.catalog.installed_config(self.catalog.tag_of(s.family, s.model))
                if path is not None:
                    from strata_control import discovery
                    found = discovery.footprint(read_config(path), path).data_dir
                    if found is not None:
                        return found
            found, source = self.catalog.data_dir_choice()
            return found if source != "default" else None
        except Exception:                                    # noqa: BLE001
            return None

    def model_state(self):
        """What this launch still needs for the model, in numbers and words (strata_cli/launch/modelstate.py)."""
        from strata_cli.launch import modelstate
        s = self.s
        self.resolved_vision()
        path, desc = (None, None)
        if s.family and s.model and not s.model_path:
            path, desc = self.installed_state()
        return modelstate.assess(self.catalog, s.family, s.model, vision=s.vision, model_path=s.model_path, data_dir=s.data_dir,
                                 config=path, desc=desc)

    def download_gb(self) -> float:
        """How much would be downloaded for this launch right now: 0 when everything is on disk (the model is installed with these choices
        or is the person's own files). A model whose helper files are missing counts only those files, never the whole model."""
        s = self.s
        if not (s.family and s.model) and not s.model_path:
            return 0.0
        try:
            return float(self.model_state().remaining_gb)
        except Exception:                                    # noqa: BLE001
            return 0.0

    def download_question(self) -> tuple[str, str]:
        """(title, text) of the question before a download starts, naming exactly what is fetched."""
        return self.model_state().question()

    def ensure_model(self) -> Path:
        s, cat = self.s, self.catalog
        if s.model_path and not (s.family and s.model):
            self.resolve_own_files()
        self.resolved_vision()
        if not (s.family and s.model):
            raise LaunchError("No model was chosen.", "Choose a model, or use --model NAME (strata launch --help lists them).", 2)
        path, desc = self.installed_state()
        if path is not None and desc is not None and desc_matches(desc, s) and not files_present(read_config(path)):
            self.emit("model", f"{s.family} {s.model} is already installed with these choices.", 100)
            return path
        title = f"{s.family} {s.model}"
        if sys.platform == "darwin" and not (cat.src / "strata_control" / "mac_fit.py").is_file():
            # a Terminal from before the Mac rules moved into strata_control/mac_fit.py: setup.py cannot install a Mac model in its Control
            # mode (no source build, no setup_mac.py), so use model files prepared on this Mac. Every other Terminal installs through setup.py.
            adopted = self.adopt_prepared_files("no Mac model installer in this Terminal")
            if adopted is not None:
                return adopted
        self.emit("model", f"Getting {title} ready (a download may be large; it continues where it stopped if interrupted)…", 0)
        argv = [self.python, str(cat.src / "setup.py"), *self.setup_args()]
        env = self._env()
        env.update(STRATA_CONTROL_INSTALL="1", STRATA_INSTALL_DIR=str(cat.root), PYTHONNOUSERSITE="1")
        env.pop("STRATA_NEED_ENGINE", None)
        if self.engine_problem:
            env["STRATA_ALLOW_NO_ENGINE"] = "1"              # the person said go on: setup warns about the engine instead of stopping
        else:
            env["STRATA_NEED_ENGINE"] = "1"
        for k in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
            env.pop(k, None)
        env["PYTHONPATH"] = str(cat.src)
        p = self._spawn(argv, env=env, cwd=str(cat.src))
        lines: list[str] = []
        folder = str(cat.data_dir() / "models" / cat.tag_of(s.family, s.model))
        total = float(next((r.download_gb for r in cat.rows() if r.family == s.family and r.model == s.model), 0) or 0)
        for line in p.stdout:                                # type: ignore[union-attr]
            if self.cancel.is_set():
                self._stop_children()
                raise Cancelled()
            line = line.rstrip()
            if not line:
                continue
            lines.append(line)
            lines[:] = lines[-400:]
            prog = cat.mcp.install_progress(lines)
            prog = cat.mcp.combined_progress(prog, {"folder": folder, "total_gb": total})
            dl = prog.get("download")
            if dl and not s.model_path and dl.get("total_gb", 0) >= 0.05:
                self.emit("model", f"Downloading {dl['done_gb']:.1f} of {dl['total_gb']:.1f} GB", float(dl["percent"]), step=prog.get("step"))
            elif prog.get("step"):
                self.emit("model", f"Setup step {prog['step']}", None, step=prog.get("step"))
        code = p.wait()
        if code != 0:
            err = ""
            for line in reversed(lines):
                if line.strip().startswith("[X]"):
                    err = line.strip()[3:].strip()
                    break
            hint = ""
            if err:
                for i, line in enumerate(lines):
                    if err in line and i + 1 < len(lines) and lines[i + 1].startswith("       "):
                        hint = lines[i + 1].strip()
            adopted = self.adopt_prepared_files(err)
            if adopted is not None:
                return adopted
            raise LaunchError("The model could not be installed." + (f" {err}." if err else ""),
                              hint or "Run it again: finished steps are skipped and a download continues where it stopped.", 6)
        path, _desc = self.installed_state()
        if path is None:
            raise LaunchError("Setup finished but no run config was written for this model.",
                              "Run `strata launch` again; if it repeats, the setup output above says why.", 6)
        self.emit("model", f"{title} is ready.", 100)
        self.check_pictures(path)
        return path

    def adopt_prepared_files(self, setup_error: str) -> Path | None:
        """The installer cannot set this graphics card up (Intel Arc, for one): if the model's files and prepared pack are already on disk,
        write the run config here (strata_cli/launch/adopt.py); else say what is missing."""
        from strata_cli.launch import adopt
        from strata_control import engine_component
        s, cat = self.s, self.catalog
        if not adopt.setup_cannot_handle_this_card(setup_error):
            return None
        _models, families, _ctx, _src = cat.tables()
        fam = families.get(s.family or "", {})
        tag = cat.tag_of(s.family, s.model)
        data_dir = Path(s.data_dir) if s.data_dir else cat.data_dir()
        model_dir = Path(s.model_path) if s.model_path else data_dir / "models" / (fam.get("tag", "") + s.model)
        if model_dir.is_file():
            model_dir = model_dir.parent
        try:
            backend = engine_component.detect_backend()[0]
        except Exception:                                    # noqa: BLE001
            backend = None
        cfg, problems = adopt.plan(family=s.family, model=s.model, model_dir=model_dir, data_dir=data_dir, src=cat.src,
                                   context=s.context or 32768, kv=s.kv, model_name=f"{fam.get('name', s.family)}-{s.model.lower()}", family_def=fam,
                                   log=cat.root / f"strata-{tag}.log", backend=backend)
        if cfg is None:
            raise LaunchError("The installer cannot set up this graphics card yet, and the files Strata would need to write the settings itself are not all there: "
                              + "; ".join(problems) + ".",
                              "Use model files that were prepared on this computer already (folder + pack), or install on a computer with an NVIDIA or AMD card.", 5)
        path = cat.root / f"strata-{tag}.json"
        adopt.write(path, cfg)
        self.notices.append("The downloader built into this Terminal sets up NVIDIA and AMD cards only (Intel Arc cards have a separate installer), so Strata wrote the "
                            f"run settings itself from the prepared model files already on this computer ({path.name}). Nothing was downloaded.")
        self.emit("model", f"Used your prepared model files and wrote the run settings ({path.name}); this card is not one the built-in downloader sets up.", 100, level="warn")
        return path

    def resolve_own_files(self) -> None:
        """"Use model files I already have": the fingerprint table says which catalog model the files are."""
        from strata_control import models
        s = self.s
        try:
            plan = models.plan_import(str(Path(s.model_path).expanduser().resolve()))
        except Exception as e:                               # noqa: BLE001
            raise LaunchError(f"These model files cannot be used: {e}.", "Pick the folder with every shard of one model, or a .gguf file of it.", 5) from None
        sel = plan.get("candidate_selection") or {}
        fam, model = sel.get("family"), sel.get("model")
        if not (fam and model) or plan.get("issues"):
            issues = "; ".join(plain_issue(str(x)) for x in (plan.get("issues") or [])[:3]) or "they are not a model Strata knows"
            raise LaunchError(f"These files cannot be used as they are ({issues}).",
                              "Choose the matching model in the list, or check that every shard is in the folder.", 5)
        s.family, s.model = fam, model
        self.emit("model", f"These files are {fam} {model}.", None)

    # ---- 3. config and profile
    def build_config(self, base_path: Path) -> Path:
        from strata_control import tune_knobs
        base = read_config(base_path)
        cfg = derive_config(base, self.s, bool_flags=tune_knobs.BOOL_FLAGS,
                            log=str(launch_dir() / f"{self.name}.engine.log"), gpus=cards_for_launch(self.s, self.catalog))
        out = launch_dir() / f"{self.name}.json"
        write_private(out, cfg)
        self.config_path = out
        return out

    def ensure_profile(self, config: Path) -> None:
        from strata_control import profiles
        name = self.profile
        parallel = self.s.parallel if self.s.parallel else self.catalog_parallel()
        sets = [f"host={acc.host_for(self.s.access)}", f"port={int(self.s.port)}"] + ([f"parallel={parallel}"] if parallel else [])
        try:
            current, _rev = profiles.load(name)
        except profiles.ProfileError:
            profiles.create(name, str(config))
        else:
            if current.get("source_config") != str(config.resolve()):
                proposed, rev = profiles.propose_source(name, str(config))
                profiles.save(name, proposed, expected_revision=rev)
        # the overrides are replaced as a set: a launch's host/port/parallel are exactly these
        cur, rev = profiles.load(name)
        want = {k: v for k, v in (profiles.parse_assignment(x) for x in sets)}
        if cur.get("overrides") != want:
            profiles.save(name, {**cur, "overrides": want}, expected_revision=rev)

    def catalog_parallel(self) -> int | None:
        try:
            comp = self.catalog.computer()
            return S.fit_parallel(comp.main_vram_gb) if comp.usable_gpus else None
        except Exception:                                    # noqa: BLE001
            return None

    # ---- 4. tuning
    def tune(self) -> None:
        s = self.s
        if s.tune == "skip":
            return
        from strata_control import presets
        if s.tuned_preset:
            try:
                presets.find(s.tuned_preset)
                self.emit("tune", "This preset is already tuned for this computer.", 100)
                return
            except presets.PresetError:
                pass
        mode = "thorough" if s.tune == "thorough" else "quick"
        status_file = launch_dir() / f"{self.name}.tune.json"
        try:
            status_file.unlink()
        except OSError:
            pass
        argv = self._cli("tune", "auto", self.profile, "--mode", mode, "--output", "json", "--status-file", str(status_file),
                         "--name", (s.name or self.name))
        if s.tune_allow_small:
            argv.append("--allow-small-differences")
        self.emit("tune", "Tuning for the best speed. Close other heavy programs while it measures.", 0)
        p = self._spawn(argv)
        out_lines: list[str] = []
        reader = threading.Thread(target=lambda: out_lines.extend(p.stdout.readlines()), daemon=True)   # type: ignore[union-attr]
        reader.start()
        while p.poll() is None:
            if self.cancel.is_set():
                self._stop_children()
                raise Cancelled()
            st = _read_json(status_file)
            if st:
                done, planned = st.get("runs_done") or 0, st.get("runs_planned") or 0
                pct = (100.0 * done / planned) if planned else None
                left = st.get("minutes_left")
                self.emit("tune", st.get("message") or "Measuring…", pct, speeds=st.get("speeds") or [],
                          round=(done, planned), minutes_left=left)
            time.sleep(1.0)
        reader.join(timeout=3)
        result = None
        for line in reversed(out_lines):
            try:
                result = json.loads(line).get("result")
                break
            except (ValueError, AttributeError):
                continue
        self.tune_result = result
        word, sentence = tune_words(result, p.returncode)
        self.tune_word = word
        preset = (result or {}).get("preset") or {}
        if preset.get("id"):
            s.tuned_preset, s.tuned_at = preset["id"], time.strftime("%Y-%m-%d")
        elif result and result.get("outcome") == "default_confirmed":
            s.tuned_at = time.strftime("%Y-%m-%d")
        self.emit("tune", f"Tuning result: {word}. {sentence}", 100, level="info" if word in ("faster", "already fastest") else "warn", word=word)

    tune_word: str | None = None

    # ---- 5. start
    def start(self) -> None:
        from strata_control import runtime
        from strata_control.profiles import ProfileError
        name = self.profile
        st = runtime.status(name)
        if st.get("state") in ("ready", "starting"):
            self.emit("start", "A server for this preset is already running; restarting it with these choices…")
            try:
                runtime.stop(name, timeout=120)
            except ProfileError as e:
                raise LaunchError(f"The running server could not be stopped: {e}.", "Run `strata launch --stop` first.", 6) from None
        self.emit("start", "Starting the engine and loading the model. The first start can take a few minutes.", None)
        done = threading.Event()
        box: dict[str, Any] = {}

        def go() -> None:
            try:
                box["r"] = runtime.start(name, wait=600)
            except Exception as e:                           # noqa: BLE001
                box["e"] = e
            finally:
                done.set()

        threading.Thread(target=go, daemon=True).start()
        t0 = time.monotonic()
        while not done.wait(1.0):
            if self.cancel.is_set():
                try:
                    runtime.stop(name, timeout=60)
                except Exception:                            # noqa: BLE001
                    pass
                raise Cancelled()
            self.emit("start", f"Loading the model… {int(time.monotonic() - t0)} s", None)
        if "e" in box:
            raise LaunchError(f"The server could not start: {box['e']}", _log_hint(self.config_path), 6)
        r = box["r"]
        if not r.get("ok"):
            raise LaunchError(f"The server did not become ready ({r.get('state')}: {r.get('error') or r.get('message') or 'see the engine log'}).",
                              _log_hint(self.config_path), 6)
        self.emit("start", "The model is loaded and answering.", 100)

    # ---- 6. access
    def open_access(self) -> str | None:
        s = self.s
        missing = acc.missing_helper(s.access)
        tunnel = None
        if s.access == "cloudflare":
            if missing:
                self.notices.append(missing + " The server is running on this computer meanwhile.")
                self.emit("access", missing, level="warn")
            else:
                self.emit("access", "Opening a Cloudflare quick tunnel…")
                tunnel, msg = acc.start_tunnel(self.profile, int(s.port))
                if tunnel is None:
                    self.notices.append(msg)
                self.emit("access", msg, level="info" if tunnel else "warn", url=tunnel)
        elif missing:
            self.notices.append(missing)
            self.emit("access", missing, level="warn")
        else:
            self.emit("access", "Ready: " + ", ".join(acc.urls_for(s)), 100)
        return tunnel

    # ---- all of it
    def plan(self) -> dict:
        s = self.s
        warns = [n.line() for n in (*S.check_access(s),)]
        return {"profile": self.profile, "settings": s.to_dict(with_key=False), "warnings": warns,
                "flags": S.flags_text(s), "steps": list(STAGES)}

    def run(self, *, start: bool = True) -> Result:
        try:
            self.ensure_engine()
            self.check_cancel()
            base = self.ensure_model()
            self.check_cancel()
            self.emit("config", "Writing this launch's settings…")
            cfg = self.build_config(base)
            self.ensure_profile(cfg)
            self.emit("config", "Settings are saved.", 100)
            if self.engine_problem:
                text = no_engine_final(self.engine_problem, self.s.name or self.name)
                self.emit("done", text, 100, level="warn")
                return Result(self.profile, [], self.s.api_key, None, self.notices + [text], str(cfg), serving=False,
                              engine_problem=self.engine_problem, final_text=text)
            self.tune()
            self.check_cancel()
            if not start:
                return Result(self.profile, [], self.s.api_key, self.tune_word, self.notices, str(cfg))
            self.start()
            tunnel = self.open_access()
            urls = acc.urls_for(self.s, tunnel_url=tunnel)
            res = Result(self.profile, urls, self.s.api_key, self.tune_word, self.notices, str(cfg), tunnel)
            self.emit("done", "Strata is serving.", 100, urls=urls)
            return res
        except Cancelled:
            self._stop_children()
            raise
        except LaunchError:
            self._stop_children()
            raise

    def stop(self) -> dict:
        from strata_control import runtime
        out = runtime.stop(self.profile, timeout=180)
        acc.stop_tunnel(self.profile)
        return out


ISSUE_WORDS = {
    "multiple_model_groups": "the folder holds more than one model",
    "incomplete_or_invalid_shard_set": "some shards are missing",
    "unrecognized_or_unsupported_model_variant": "this is not a model size Strata knows",
    "unsupported_or_missing_architecture": "this model type is not supported by the Strata engine (add it as a Custom model API with llama-server instead)",
    "unsupported_or_incomplete_tokenizer_metadata": "the file has no usable tokenizer",
    "missing_or_unexpected_expert_quantization": "its quantization is not one Strata knows",
}


def plain_issue(code: str) -> str:
    """One problem from the GGUF preview, in words (it may be `file.gguf: GGUF shard is not a nonempty regular file` and the like)."""
    if code in ISSUE_WORDS:
        return ISSUE_WORDS[code]
    low = code.lower()
    if "not a nonempty regular file" in low or "unreadable" in low:
        return "a file is empty, unreadable or not a regular file"
    if "does not begin with a gguf header" in low or "header version" in low:
        return "a file is not a GGUF model file"
    if "unsupported_architecture" in low:
        return "this model type is not supported by the Strata engine (add it as a Custom model API with llama-server instead)"
    if "symlink" in low:
        return "a file is a shortcut (symbolic link); point at the real file"
    return code.replace("_", " ")


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _log_hint(config: Path | None) -> str:
    log = None
    if config:
        try:
            log = read_config(config).get("log")
        except (OSError, ValueError):
            log = None
    return f"The engine's log has the reason: {log}" if log else "The engine's log has the reason (strata logs, or the log named in the run config)."
