"""Opt-in local model selection using Strata's existing process manager."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
import uuid

from strata_mcp import Strata, proc_alive, proc_identity, spawn_detached

MODELS = {
    "original": ("iq2_xs", "Qwen3.8 Original"),
    "quality": ("iq3_s", "Qwen3.8 Quality (IQ3_S)"),
    "coder": ("coder-iq1_m", "Qwen3.8 Coder"),
    "swift": ("swift-iq2_xs", "Swift 1.5"),
    "uncensored": ("uncensored-iq2_xs", "Qwen3.8 Uncensored (experimental)"),
}
PENDING = {"starting", "restoring"}


def save_switch(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class ModelSwitcher:
    def __init__(self, root: Path):
        self.root = root
        self.manager = Strata(root)
        self.path = root / ".strata-mcp" / "model-switch.json"
        self.lock = threading.Lock()

    def models(self) -> list[dict]:
        models = []
        for name, (tag, label) in MODELS.items():
            path = self.root / f"strata-{tag}.json"
            cfg = self.manager.read_config(path)
            ready = False
            if cfg:
                desc = self.manager.describe_config(path)
                ready = (desc["ready"] and desc["host"] == "127.0.0.1" and desc["port"] == 8080
                         and cfg.get("model_switch") is True
                         and (Path(cfg.get("tokenizer", "")) / "tokenizer.json").is_file())
            models.append({"id": name, "name": label, "model": cfg.get("model_name"), "available": bool(ready)})
        return models

    def _state(self, svc) -> dict:
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                return {}
        except (OSError, ValueError):
            return {}
        if state.get("status") in PENDING and not proc_alive(state.get("pid"), state.get("ident")):
            target = next((m for m in self.models() if m["id"] == state.get("target")), {})
            state.update(status="ready" if target.get("model") == svc.model and svc.loaded() else "failed",
                         message="The switch process exited. Check the current model status.")
            save_switch(self.path, state)
        return state

    def _owned(self) -> bool:
        tracked = self.manager.tracked_server()
        # Windows' venv python.exe is a redirector: the tracked launcher is our direct parent.
        owners = {os.getpid(), os.getppid()} if os.name == "nt" else {os.getpid()}
        return bool(tracked and not tracked.get("ended") and tracked.get("pid") in owners)

    def snapshot(self, svc) -> dict:
        with self.lock:
            models, state = self.models(), self._state(svc)
            with svc.status_lock:
                busy = bool(svc.status.get("busy") or svc.status.get("queued"))
            current = next((m["id"] for m in models if m["model"] == svc.model), None)
            return {"models": models, "current": current, "loaded": svc.loaded(), "busy": busy,
                    "can_switch": self._owned(),
                    "switch": {k: state[k] for k in ("id", "status", "target", "message", "restored") if k in state}}

    def begin(self, svc, name) -> tuple[int, dict]:
        if not isinstance(name, str) or name not in MODELS:
            return 400, {"error": {"message": "Choose a model from the list."}}
        with self.lock:
            if self._state(svc).get("status") in PENDING:
                return 409, {"error": {"message": "A model is loading. Wait for it to finish."}}
            models = self.models()
            target = next(m for m in models if m["id"] == name)
            if not target["available"]:
                return 404, {"error": {"message": "This model is not installed or configured."}}
            current = next((m["id"] for m in models if m["model"] == svc.model), None)
            if current == name:
                return 200, {"status": "current", "target": name}
            if not self._owned():
                return 409, {"error": {"message": "Start Strata with START-Strata.bat before switching models."}}
            if not svc.fifo.acquire(blocking=False):
                return 409, {"error": {"message": "Wait for response generation to finish before switching."}}
            try:
                with svc.status_lock:
                    if svc.status.get("busy") or svc.status.get("queued"):
                        return 409, {"error": {"message": "Wait for response generation to finish before switching."}}
                state = {"id": uuid.uuid4().hex, "status": "starting", "target": name, "previous": current,
                         "started": time.time()}
                worker = self.root / "serve" / "model_switch_worker.py"
                if not worker.is_file():
                    return 503, {"error": {"message": "The model switch worker is missing."}}
                process = spawn_detached([self.manager.run_python(), str(worker), name, current or "", state["id"]],
                                         self.root, self.root / ".strata-mcp" / "model-switch.log")
                state.update(pid=process.pid, ident=proc_identity(process.pid))
                save_switch(self.path, state)
                return 202, {"status": "starting", "target": name, "id": state["id"]}
            finally:
                svc.fifo.release()
