"""launcher/api.py - what the launcher page asks Strata to do.

Every answer comes from the controller the MCP server uses (`tools/strata_mcp.py`): the model tables and their fit
checks, the install job, the start/stop of a model, the logs, the connection settings.  What is new here is only what
a page needs and an assistant does not: presets, and the calibration as a job that can be watched and stopped.

Nothing in this file prints or parses a console.  The errors are the controller's own (`ToolError`) or a
`PresetError`, and the HTTP layer turns them into a message the page shows.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import calibrate as C
from . import presets as P
from .controller import strata
from .jobs import Job

STATE_DIR = ".strata-launcher"
INSTALLED_PREFIX = "builtin-installed-"      # a built-in preset that mirrors an installed model's own config


class Launcher:
    """One Strata folder, its presets, and the operations the page can ask for."""

    def __init__(self, root=None, state_dir=None):
        self.root = Path(root).resolve() if root else Path(__file__).resolve().parents[1]
        self.m, self.s = strata(self.root)
        self.t = self.m.Tools(self.s)
        # tests point this at a temporary folder, so a test never writes next to a real install
        self.state_dir = Path(state_dir) if state_dir else self.root / STATE_DIR
        self.presets = P.Store(self.state_dir / "presets.json")
        self.calibrate_job = Job(self.state_dir, "calibrate")

    # ---- the tables
    def catalog(self) -> dict:
        models, families, contexts, source = self.s.tables()
        hw = self.s.hardware()
        return {"models": models, "families": families, "contexts": list(contexts), "source": source,
                # the sizes per family, asked of the same controller the page's model table uses: a size that names
                # no families in setup's table belongs to the fine-tunes too, and this is where that answer lives
                "family_sizes": {f: self.s.sizes_of(models, f) for f in families},
                "ram_gb": hw.get("ram_gb"), "gpus": [{k: g.get(k) for k in ("index", "name", "vram_gb", "usable")}
                                                     for g in hw.get("gpus", [])]}

    def schema(self) -> dict:
        """The form the page builds: every field, its choices in plain words, and the setup.py flag it writes."""
        fields = []
        for f in P.FIELDS:
            d = {k: v for k, v in f.items() if k != "choices"}
            if "choices" in f:
                d["choices"] = [{"value": k, "label": v} for k, v in f["choices"].items()]
            fields.append(d)
        return {"fields": fields, "trained_context": P.TRAINED_CONTEXT}

    def ui_catalog(self) -> dict:
        """The model list as the page wants it: the families with their sizes, and one entry per size with what it
        needs and whether it fits this PC - setup's own table (`Strata.model_table`), reshaped for a table view."""
        cat = self.catalog()
        sizes, families = {}, []
        for row in self.s.model_table(self.s.hardware()):
            fam = cat["families"][row["family"]]
            families.append({"id": row["family"], "title": row["title"], "about": row.get("about"),
                             "by": fam.get("by"), "tag": fam.get("tag", ""),
                             "images": row.get("images", True), "experimental": row.get("experimental", False),
                             "sizes": [s["model"] for s in row["sizes"]]})
            for s in row["sizes"]:
                sizes[s["model"]] = s
        return {"families": families, "sizeInfo": sizes, "contexts": P.offered_contexts(cat),
                "setup_contexts": cat["contexts"],
                "recommendation": self.recommendation(), "ram_gb": cat["ram_gb"], "gpus": cat["gpus"]}

    def recommendation(self) -> dict:
        return self.s.recommend(self.s.hardware())

    # ---- the page's first paint
    def state(self, hardware: bool = False) -> dict:
        out = self.t.strata_status(hardware=hardware)
        out["presets"] = self.presets.load()
        out["builtin"] = self.builtin_presets()
        out["calibrate"] = self.calibrate_job.status(lines=12)
        out["strata_folder"] = str(self.root)      # which Strata folder this page is reading
        return out

    # ---- the presets that are not stored
    def installed_presets(self) -> list:
        """One entry per model installed in this folder, exactly as its own strata-<model>.json has it: the page
        shows the setup this PC actually has, not only what setup would recommend for it."""
        cat = self.catalog()
        out = []
        for cfg in self.s.configs():
            model_id = cfg.stem[len("strata-"):]
            try:
                p = self._from_config(model_id)
            except Exception:                        # noqa: BLE001 - a half-written config is not the page's problem
                continue
            if not p.get("context"):
                rec = self.recommendation()
                p["context"] = rec.get("context") or cat["contexts"][0]
            fam = cat["families"].get(p["family"], {})
            p["id"] = INSTALLED_PREFIX + model_id
            p["builtin"] = True
            p["name"] = f"{fam.get('title', p['family'])} {p['model']} as it is set up here"
            p["note"] = f"what {cfg.name} says this model runs with"
            out.append(p)
        return out

    def builtin_presets(self) -> list:
        return P.builtin(self.catalog(), self.recommendation()) + self.installed_presets()

    # ---- presets
    def preset(self, pid: str) -> dict:
        """A stored preset, or one of the built-in ones (which are derived, never stored)."""
        pid = str(pid or "")
        if pid.startswith("builtin-"):
            found = next((b for b in self.builtin_presets() if b["id"] == pid), None)
            if found is None:
                raise self.m.ToolError(f"{pid} is not one of the built-in presets on this PC")
            return found
        p = self.presets.get(pid)
        if p is None:
            raise self.m.ToolError(f"no preset {pid!r}: the page lists the ones this folder has")
        return p

    def new_preset(self, seed: dict | None = None) -> dict:
        """An empty preset for the form, pre-filled with setup's own answers for this PC. Not validated and not
        stored: the page validates it when it is saved."""
        cat = self.catalog()
        return dict(P.blank(cat, self.recommendation()), **(seed or {}))

    def save_preset(self, raw: dict) -> dict:
        """Check a preset and store it. A built-in one is not stored: editing it makes a new id."""
        cat = self.catalog()
        p = P.clean(raw, cat, keep_id=raw.get("id"))
        if p["id"].startswith("builtin-"):
            p["id"] = P.slug(p["name"])
        return self.presets.put(p)

    def delete_preset(self, pid: str) -> dict:
        if str(pid).startswith("builtin-"):
            raise self.m.ToolError("a built-in preset follows this PC's hardware; duplicate it to change it")
        if not self.presets.delete(str(pid)):
            raise self.m.ToolError(f"no preset {pid!r}")
        return {"summary": f"the preset {pid} was removed"}

    def _from_config(self, model_id: str, name: str = "") -> dict:
        """A preset describing an installed model, as its own strata-<model>.json says it is set up."""
        cfg_path = self.s.find_config(model_id)
        cfg = self.s.read_config(cfg_path)
        return P.from_config(self.s.describe_config(cfg_path), cfg, self.catalog(), name or None)

    def preset_from_model(self, model_id: str, name: str = "") -> dict:
        """The same, checked: what the page edits and saves, so it can be changed and installed again."""
        return P.clean(self._from_config(model_id, name), self.catalog())

    def plan(self, pid: str) -> dict:
        """What installing this preset would do - the same plan the MCP tool returns before it downloads 60-110 GB."""
        p, cat = self.preset(pid), self.catalog()
        plan, args = self._plan_args(p)
        plan["preset"] = p
        plan["preset_id"] = p["id"]
        plan["setup_args"] = args
        plan["setup_command"] = ("START-HERE.bat " if self.m.WIN else "./setup.sh ") + " ".join(args)
        plan["model_id"] = P.model_id(p, cat)
        plan["calibrate"] = p.get("calibrate") or "ask"
        # setup's plan has no yes/no of its own: the page wants one. Only a full disk stops it; a RAM that is too
        # small is said in the plan, and setup can still do it with --low-ram.
        bad_ram = plan.get("ram") not in (None, "fits")
        plan["ok"] = not plan["disk_short"] and not bad_ram
        plan["why"] = plan["disk_short"] or (plan["ram"] if bad_ram else "")
        plan["changes"] = self._changes(plan["model_id"], p)
        # a length setup's own menu does not list: setup takes any --context, but its menu stops at these lengths,
        # and past the trained 262,144 it adds rope scaling
        plan["own_context"] = int(p["context"]) not in cat["contexts"]
        return plan

    # what an install would change on a model that is already installed here; name/note/calibrate are the launcher's
    # own, backend/api_key/port/host are not read back from a config in a way that can be compared, so they are left
    # out - an empty field in a preset means "setup decides", which for those is what happens anyway
    COMPARED = ("context", "kv", "vision", "kv_streaming", "low_ram",
                "resident_budget_gib", "vram_reserve_mib", "speed_projection", "gpu", "gpus",
                "layer_split")

    def preset_diff(self, raw: dict) -> dict:
        """What this preset differs from on the config of the model it names, and whether that model is installed
        here.  Starting a model runs its own strata-<model>.json, not the preset, so the page needs this to say so
        before Start: a preset with the speed projection off does not turn off a model whose config has it on."""
        cat = self.catalog()
        model_id = P.model_id(raw, cat)
        installed = any(c.stem == "strata-" + model_id for c in self.s.configs())
        return {"model_id": model_id, "installed": installed,
                "changes": self._changes(model_id, raw) if installed else []}

    # what setup does when a config carries nothing for a field: a preset that says the same thing is not a change
    DEFAULTS = {"vision": "no", "speed_projection": "off", "kv_streaming": "auto", "low_ram": "auto"}
    # The two settings setup decides by itself.  "auto" is not a setting: it is setup's RAM rule applied to this PC,
    # so a preset that says "auto" - and a config whose flags are that decision - is the model as it runs, and so is
    # a preset that spells the decision out ("off" on a PC that does not need the low-RAM mode).  Comparing the
    # words made every Start re-run setup to change nothing; the rules are setup's own (kv_streaming_wanted,
    # low_ram_wanted), not a second copy of them here.
    RULED = ("kv_streaming", "low_ram")

    def _decides(self, key: str, model: str, ctx, kv, ram) -> str:
        """What setup's rule makes of "auto" for one size on this PC: "on" or "off" ("" when it cannot tell)."""
        S = self.s.setup_module()
        if S is None or not model:
            return ""
        try:
            if key == "kv_streaming":
                return "on" if S.kv_streaming_wanted(model, int(ctx), kv, ram) else "off"
            return "on" if S.low_ram_wanted(model, ram, "auto") else "off"
        except (KeyError, TypeError, ValueError):        # a size this table does not know: nothing to resolve into
            return ""

    def _changes(self, model_id: str, chosen: dict) -> list:
        """What installing this preset would change on a model that is already installed here.  An install rewrites
        that model's strata-<model>.json, so a preset that carries a smaller context quietly replaces the one the
        model runs with now; the page is shown the difference before setup runs."""
        path = next((c for c in self.s.configs() if c.stem == "strata-" + model_id), None)
        if path is None:
            return []
        try:
            cfg = self.s.read_config(path)
            current = P.from_config(self.s.describe_config(path), cfg, self.catalog())
        except Exception:                        # noqa: BLE001 - a half-written config is not the page's problem
            return []
        args = cfg.get("args") if isinstance(cfg.get("args"), list) else []
        if "--kv-resident" in args:
            current["kv_streaming"] = "on"       # setup writes --kv-resident when it streams the KV cache
        labels = {f["key"]: f["label"] for f in P.FIELDS}
        ram = (self.s.hardware() or {}).get("ram_gb") or 0
        model = str(chosen.get("model") or "")
        # each side is read with its own context and KV precision: the config's flags are what setup decided for the
        # context it installed, the preset's for the context it asks for
        sides = {"config": (current.get("context") or 0, current.get("kv") or "int8"),
                 "preset": (chosen.get("context") or 0, chosen.get("kv") or "int8")}

        def text(v):
            return "" if v in ("", None) else str(v)

        def said(key, v, side):
            # "no images", "projection off", "auto": what a preset spells out and what a config leaves out are the
            # same setting, and listing them as a difference would make every start re-run setup
            t = text(v)
            if key == "low_ram" and t in ("resident", "mmap"):
                return "on"                       # the low-RAM mode's two variants are one setting here
            if t in ("", self.DEFAULTS.get(key, "\x00")):
                return self._decides(key, model, sides[side][0], sides[side][1], ram) if key in self.RULED else ""
            return t

        out = []
        for key in self.COMPARED:
            now, want = text(current.get(key)), text(chosen.get(key))
            if said(key, now, "config") == said(key, want, "preset"):     # the rules folded in, per side
                continue
            out.append({"key": key, "label": labels[key], "config": now or "setup's own choice",
                        "preset": want or "setup's own choice"})
        return out

    def _plan_args(self, p: dict):
        """setup's plan, with the preset's own flags merged in (the preset wins where both name a flag)."""
        def opt(key):
            v = p.get(key)
            return None if v in ("", None) else v
        plan, args = self.t.install_plan(
            family=p["family"], model=p["model"], context=int(p["context"]), vision=opt("vision"), kv=opt("kv"),
            gpu=opt("gpu"), gpus=opt("gpus"), backend=("auto" if p.get("backend") in ("", "auto") else p["backend"]),
            low_ram=opt("low_ram"), port=opt("port"), data_dir=None)
        return plan, P.merge_args(args, P.setup_args(p))

    def install(self, pid: str) -> dict:
        """Start the install: setup.py, non-interactive, in the background, resumable.  The plan is shown first -
        the page asks before it downloads 60-110 GB."""
        p = self.preset(pid)
        plan, args = self._plan_args(p)
        self._refuse_if_running()
        if plan["disk_short"]:
            raise self.m.ToolError(plan["disk_short"])
        out = self.t.launch_install(plan, args)
        self._mark_install(p, (self.s.state("install") or {}).get("job"))
        return {"plan": plan, "install": self.s.install_job() or out}

    APPLY_WAIT = 900          # setup re-applying an installed model: seconds here, more if it repairs a file

    def apply_preset(self, p: dict) -> dict:
        """Make a preset the model's own settings: setup itself, non-interactive, with --no-start.  For a model that
        is already installed this only re-checks the files and rewrites strata-<model>.json - a few seconds to a
        minute on this PC, nothing fetched again - which is why Start can wait for it and go straight on."""
        self._refuse_if_running()
        plan, args = self._plan_args(p)
        if plan["disk_short"]:
            raise self.m.ToolError(plan["disk_short"])
        self.t.launch_install(plan, args)
        job_id = (self.s.state("install") or {}).get("job")
        # a re-apply does not start a measurement by itself: "measure this PC: always" means after a download
        self._mark_install(dict(p, calibrate="never"), job_id)
        end = time.time() + self.APPLY_WAIT
        job = {}
        while True:
            job = self.s.install_job() or {}
            if not job.get("running"):
                break
            if time.time() > end:
                raise self.m.ToolError(f"setup is still running after {self.APPLY_WAIT // 60} minutes: the page "
                                       "shows its log, and you can stop it there")
            time.sleep(1.0)
        if job.get("result") != "installed":
            raise self.m.ToolError("setup stopped while applying the preset (" + str(job.get("result") or
                                       "no exit code recorded") + "): its log is on the page under Download")
        return {"plan": plan, "install": job}

    def apply_by_id(self, pid: str) -> dict:
        """The page's "Update / re-apply": make a preset the model's settings without starting it and without
        downloading anything, since the model is already here."""
        p, cat = self.preset(pid), self.catalog()
        model_id = P.model_id(p, cat)
        if model_id not in [c.stem[len("strata-"):] for c in self.s.configs()]:
            raise self.m.ToolError(f"{P.summary(p, cat)} is not installed here: download it first "
                                   "(the page shows the plan and the size)")
        changes = self._changes(model_id, p)
        if not changes:
            return {"install": None, "applied": [], "summary": f"{model_id} already runs with these settings"}
        out = self.apply_preset(p)
        return {"install": out["install"], "applied": changes,
                "summary": f"applied to {model_id}: " + ", ".join(f"{c['label']} {c['preset']}" for c in changes)}

    def _refuse_if_running(self) -> None:
        tracked = self.s.tracked_server()
        if tracked and not tracked.get("ended"):
            raise self.m.ToolError("Strata is running (started by the launcher): stop it first - setup may update "
                                   "the engine files it uses")

    def install_status(self) -> dict:
        job = self.s.install_job()
        if job and not job["running"] and job.get("exit_code") == 0:
            self._maybe_calibrate(job)
        return {"install": job}

    def install_cancel(self) -> dict:
        return self.t.strata_install(cancel=True)

    def _mark_install(self, p: dict, job_id) -> None:
        """Remember which preset an install came from, so a preset that says "measure this PC" can run its
        calibration when the install finishes."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "last-install.json").write_text(
            json.dumps({"job": job_id, "preset": p.get("id"), "calibrate": p.get("calibrate"),
                        "model": P.model_id(p, self.catalog())}, indent=1), encoding="utf-8")

    def _maybe_calibrate(self, job: dict) -> None:
        try:
            mark = json.loads((self.state_dir / "last-install.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not mark or mark.get("job") != job.get("job") or mark.get("calibrate") != "always":
            return
        mark["calibrate"] = "done"
        (self.state_dir / "last-install.json").write_text(json.dumps(mark, indent=1), encoding="utf-8")
        if self.calibrate_job.running():
            return
        try:
            self.calibrate(mark["model"])
        except Exception:                               # noqa: BLE001 - a finished install is not undone by this
            pass

    # ---- running it
    def start(self, pid: str, wait_seconds: int = 10, apply_first: bool = True) -> dict:
        """Start a model.  A model runs what its own strata-<model>.json says, so when the preset differs from that
        config, applying it is part of starting: setup re-writes the config first (nothing is downloaded again for a
        model that is already here) and the model then starts with the preset's settings."""
        p, cat = self.preset(pid), self.catalog()
        model_id = P.model_id(p, cat)
        if model_id not in [c.stem[len("strata-"):] for c in self.s.configs()]:
            raise self.m.ToolError(f"{P.summary(p, cat)} is not installed here: download it first "
                                   "(the page shows the plan and the size)")
        changes = self._changes(model_id, p)
        applied = None
        if apply_first and changes:
            self.apply_preset(p)
            applied = changes
        out = self.t.strata_start(model=model_id, port=None, gpu=None, wait_seconds=wait_seconds)
        if applied:
            out["applied"] = applied
            what = ", ".join(f"{c['label']} {c['preset']}" for c in applied)
            out["summary"] = f"applied the preset to {model_id} first ({what}); " + out["summary"]
        return out

    def stop(self, force: bool = False) -> dict:
        return self.t.strata_stop(force=force)

    def logs(self, source: str = "", model: str = "", lines: int = 80) -> dict:
        return self.t.strata_logs(source=source or None, model=model or None, lines=int(lines))

    def connect(self) -> dict:
        return self.t.strata_connect_info()

    # ---- measuring this PC
    def calibrate(self, model_id: str, then_start: bool = False) -> dict:
        """setup's --calibrate for one installed model, as a job: it starts its own engine, so nothing else may run.
        `then_start` is the page's answer to a preset's "ask": measure first, then start the model."""
        running = [x for x in (self.t.strata_status(hardware=False).get("running") or [])]
        if running:
            raise self.m.ToolError(f"Strata is running on port {running[0]['port']}: stop it first - the "
                                   "measurement starts its own engine and needs the VRAM and RAM to itself")
        if self.calibrate_job.running():
            raise self.m.ToolError("a measurement is already running (its progress is on the page)")
        cfg = self.s.find_config(model_id)
        py = self.s.run_python()
        if not Path(py).exists():
            raise self.m.ToolError("Strata's Python environment (.venv) is missing: run the install again")
        result = self.state_dir / "calibrate-result.json"
        out = self.calibrate_job.start([py, "-m", "launcher.calibrate", str(cfg), str(result)], cwd=self.root,
                                       label=f"measuring this PC for {cfg.name}", result=result)
        # only once the job really runs: a measurement that never started must not leave a "then start" behind
        self._save_hook({"model": model_id, "start": bool(then_start)} if then_start else None)
        return out

    def _hook(self) -> dict:
        try:
            raw = json.loads((self.state_dir / "calibrate-hook.json").read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_hook(self, d: dict | None) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        f = self.state_dir / "calibrate-hook.json"
        if d is None:
            try:
                f.unlink()
            except OSError:
                pass
        else:
            f.write_text(json.dumps(d, indent=1), encoding="utf-8")

    def calibrate_status(self) -> dict:
        """The measurement's progress; a measurement the page asked for before a start ends with that start."""
        status = self.calibrate_job.status()
        hook = self._hook()
        if hook.get("start") and not status.get("running") and hook.get("model"):
            self._save_hook(None)
            try:
                if not (self.t.strata_status(hardware=False).get("running") or []):
                    status["then_start"] = self.t.strata_start(model=hook["model"], wait_seconds=5)
            except Exception as e:                      # noqa: BLE001 - said on the page, not swallowed
                status["then_start"] = {"summary": f"it did not start afterwards: {e}"}
        return {"calibrate": status}

    def calibrate_cancel(self) -> dict:
        return {"calibrate": self.calibrate_job.cancel()}

    # ---- what the model itself is on this PC
    def model_card(self, p: dict, model_id: str | None = None) -> dict:
        """The model behind this preset on this PC: its own config when it is installed (which is what starting it
        runs), the files it misses, what setup's calibration recorded for it, and which model the server serves now.
        No speed numbers: a model runs what its own config says, and the engine's own page shows how fast."""
        cat = self.catalog()
        model_id = model_id or P.model_id(p, cat)
        name = P.model_name(p, cat)
        configs = {c.stem[len("strata-"):]: c for c in self.s.configs()}
        installed = self.s.describe_config(configs[model_id]) if model_id in configs else None
        card = {"model": model_id, "model_name": name, "preset_id": p.get("id"),
                "settings": P.summary(p, cat), "installed": bool(installed),
                "calibrations": [c for c in C.calibrations(self.s.settings_path())
                                 if c.get("model_name") == name][:5]}
        if installed:
            card["model_settings"] = installed
        running = self.t.strata_status(hardware=False).get("running") or []
        if running:
            card["live"] = {"running": True, "model": running[0].get("model"), "port": running[0].get("port")}
        return card

    def model_for_preset(self, pid: str) -> dict:
        """The same for a preset, or for an installed model named directly (the page can select either)."""
        try:
            p = self.preset(pid)
        except self.m.ToolError:
            p = self._from_config(pid)          # not a preset id: an installed model's
        return self.model_card(p)
