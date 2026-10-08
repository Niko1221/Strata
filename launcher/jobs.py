"""launcher/jobs.py - a long job the launcher starts, shows progress for, and can stop.

setup's install already has this shape (tools/strata_mcp.py runs it detached, keeps its log and reads the progress
back out of setup's own output).  The calibration needs the same thing - 5-10 minutes of measuring, a log to watch,
a Cancel button - so it uses this small runner, which follows the same rules: the job is a separate process, its
output goes to a log file, its state is a JSON file, and it keeps running if the launcher window is closed.

One job per name at a time; the state file names the process, so a launcher started again finds the job that is
still running.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .controller import strata_module


class Job:
    """One named job: start it, read its status, cancel it."""

    def __init__(self, state_dir: Path, name: str):
        self.dir = Path(state_dir)
        self.name = name
        self.m = strata_module()

    @property
    def state_path(self) -> Path:
        return self.dir / f"{self.name}-job.json"

    @property
    def log_path(self) -> Path:
        return self.dir / f"{self.name}.log"

    def state(self) -> dict:
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, d: dict | None) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if d is None:
            try:
                self.state_path.unlink()
            except OSError:
                pass
            return
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def running(self) -> bool:
        s = self.state()
        return bool(s) and self.m.proc_alive(s.get("pid"), s.get("ident"))

    def start(self, cmd: list, cwd: Path, label: str, result: Path | None = None,
              env: dict | None = None) -> dict:
        """Run `cmd` detached; `result` is the file the job writes its own summary to when it ends."""
        if self.running():
            raise RuntimeError(f"{label} is already running (started {self.state().get('started')})")
        self.dir.mkdir(parents=True, exist_ok=True)      # the first job in a fresh folder writes its log here
        with open(self.log_path, "ab") as f:      # the child writes its own output here: bytes, not str
            f.write(f"\n===== launcher: {label} at {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n".encode("utf-8"))
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", **(env or {}))
        p = self.m.spawn_detached([str(x) for x in cmd], cwd, self.log_path, env)
        time.sleep(0.2)
        self._save({"pid": p.pid, "ident": self.m.proc_identity(p.pid), "label": label, "log": str(self.log_path),
                    "result": str(result) if result else None,
                    "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cmd": [str(x) for x in cmd]})
        return self.status()

    def cancel(self) -> dict:
        s = self.state()
        if not s or not self.m.proc_alive(s.get("pid"), s.get("ident")):
            self._save(None)
            return {"summary": f"the {self.name} job is not running"}
        self.m.proc_kill_tree(s["pid"], s.get("ident"))
        self.m.wait_gone(s["pid"], s.get("ident"), 15)
        self._save(None)
        return {"summary": f"the {self.name} job was stopped"}

    def status(self, lines: int = 40) -> dict:
        s = self.state()
        if not s:
            return {"running": False}
        alive = self.m.proc_alive(s.get("pid"), s.get("ident"))
        result = {}
        try:                                          # the job writes its own result when it ends
            result = json.loads(Path(s.get("result", "")).read_text(encoding="utf-8")) if s.get("result") else {}
        except (OSError, ValueError):
            result = {}
        tail = self.m.read_tail(Path(s.get("log", "")), lines)
        return {"running": alive, "label": s.get("label"), "started": s.get("started"), "log": s.get("log"),
                "lines": tail[-25:], "result": result,
                "result_note": (result.get("summary") or "ended without a result") if not alive else "running"}
