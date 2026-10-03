"""A detached worker survives the old HTTP server's orderly shutdown."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from tools.personal_control import start_model
from serve.model_switch import MODELS, save_switch
from strata_mcp import Strata, Tools


def main():
    name, previous, operation = sys.argv[1:4]
    path = ROOT / ".strata-mcp" / "model-switch.json"
    time.sleep(1)  # Allow the handler to return 202 and save the worker identity.
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("id") != operation or state.get("target") != name:
        raise RuntimeError("Model switch operation changed before starting.")
    manager = Tools(Strata(ROOT))
    try:
        result = start_model(manager, MODELS[name][0])
        state.update(status="ready", model=result["model"], finished=time.time())
    except Exception as error:
        print(f"Model switch failed: {error}", flush=True)
        state.update(status="restoring", message="Loading failed. Restoring the previous model.")
        save_switch(path, state)
        restored = False
        if previous in MODELS:
            try:
                start_model(manager, MODELS[previous][0])
                restored = True
            except Exception as rollback_error:
                print(f"Restore failed: {rollback_error}", flush=True)
        state.update(status="failed", restored=restored, finished=time.time(),
                     message="Loading failed. The previous model was restored." if restored else
                     "Loading failed. Start Strata with START-Strata.bat.")
    save_switch(path, state)


if __name__ == "__main__":
    main()
