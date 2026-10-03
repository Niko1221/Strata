"""Convenience launchers for installed native models, using Strata's process manager."""
import json
from pathlib import Path
import sys
import time
import webbrowser

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from strata_mcp import Strata, Tools
from start_models import model_config

MODELS = {
    "original": ("iq2_xs", "Qwen3.8 Original"),
    "quality": ("iq3_s", "Qwen3.8 Quality (IQ3_S)"),
    "coder": ("coder-iq1_m", "Qwen3.8 Coder"),
    "swift": ("swift-iq2_xs", "Swift 1.5"),
    "uncensored": ("uncensored-iq2_xs", "Qwen3.8 Uncensored (experimental projection)"),
}


def start_model(manager, requested):
    desc = manager.s.describe_config(model_config(manager.s, requested))
    requested = desc['model']
    if not desc["ready"]:
        raise RuntimeError(f"Model is not ready: {requested}")
    status = manager.call("strata_status", {"hardware": False})
    running = status.get("running", [])
    if status.get("loading"):
        raise RuntimeError("A model is loading. Wait for it before switching models.")
    if running and running[0]["model"] != desc["model_name"]:
        print("Stopping the previous model before switching...", flush=True)
        manager.call("strata_stop", {})
    print(f"Starting {desc['model_name']}...", flush=True)
    manager.call("strata_start", {"model": requested, "wait_seconds": 0})
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        info = manager.s.probe(8080, deep=False)
        if info and not info.get("other") and info.get("loaded"):
            if info.get("model") != desc["model_name"]:
                raise RuntimeError("Another model is already running on port 8080.")
            print("Ready: http://localhost:8080/v1", flush=True)
            return info
        tracked = manager.s.tracked_server()
        if not tracked or tracked.get("ended"):
            raise RuntimeError("Server stopped while loading. See .strata-mcp/server.log.")
        time.sleep(2)
    raise RuntimeError("The model is still loading. Check STATUS-Strata.bat.")


def main():
    action = sys.argv[1] if len(sys.argv) >= 2 else "status"
    manager = Tools(Strata(ROOT))
    if action == "open":
        info = manager.s.probe(8080, deep=False)
        if not info:
            try:
                state = json.loads((ROOT / ".strata-mcp/model-switch.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                state = {}
            chosen = state.get("previous") if state.get("status") == "failed" else state.get("target")
            requested = MODELS[chosen][0] if chosen in MODELS else None
            info = start_model(manager, requested)
        elif info.get("other"):
            raise RuntimeError("Port 8080 is already being used by another application.")
        print(json.dumps(info, ensure_ascii=False, indent=2))
        webbrowser.open("http://127.0.0.1:8080")
    elif action == "start":
        requested = sys.argv[2] if len(sys.argv) >= 3 and not sys.argv[2].startswith('--') else None
        requested = MODELS.get(requested, (requested,))[0]
        print(json.dumps(start_model(manager, requested), ensure_ascii=False, indent=2))
        if "--no-browser" not in sys.argv:
            webbrowser.open("http://127.0.0.1:8080")
    elif action in ("stop", "status"):
        args = {"hardware": False} if action == "status" else {}
        print(json.dumps(manager.call("strata_" + action, args), ensure_ascii=False, indent=2))
    else:
        raise ValueError("Expected open, start, stop, or status")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
