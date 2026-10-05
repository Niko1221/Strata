"""launcher/calibrate.py - setup's --calibrate for one model, as a job the launcher can watch.

    python -m launcher.calibrate <strata-<model>.json> [result.json]

It runs setup.py's own `calibrate_config()`, so the measurement is exactly the one `START-HERE.bat --calibrate`
does: measure this PC's decode speed with a few engine settings (the PCIe share, the draft depth, the CPU threads),
write the winners into the model's config, and remember them per PC and model in setup's settings file.  Nothing is
reimplemented here - only the way it is started, so the launcher can show its progress and stop it.

The last line written to the result file (by default `.strata-launcher/calibrate-result.json`) is what the
launcher's Measuring panel shows afterwards.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / ".strata-launcher" / "calibrate-result.json"

# setup keys its calibration by the PC and the model: "GPU|VRAM|CPU|RAM|model|context|images"
CAL_KEY = re.compile(r"^(?P<gpu>[^|]*)\|(?P<vram>[^|]*)\|(?P<cpu>[^|]*)\|(?P<ram>[^|]*)\|"
                     r"(?P<model>[^|]*)\|(?P<context>[^|]*)\|(?P<images>[^|]*)$")


def calibrations(settings_path: Path) -> list[dict]:
    """What setup's --calibrate recorded on this PC (its settings file keeps the decode speed it measured, keyed by
    the PC and the model).  Read-only: the launcher shows them, it never writes that file."""
    try:
        raw = json.loads(Path(settings_path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return []
    raw = raw.get("calibration") if isinstance(raw, dict) else None
    out = []
    if not isinstance(raw, dict):
        return out
    for key, v in raw.items():
        if not isinstance(v, dict):
            continue
        m = CAL_KEY.match(key)
        g = m.groupdict() if m else {}
        out.append({"model_name": g.get("model") or None, "context": g.get("context") or None,
                    "gpu": g.get("gpu") or None, "decode_tok_s": v.get("tok_s"), "date": v.get("date"),
                    "settings": v.get("settings") or {}, "key": key})
    return sorted(out, key=lambda r: str(r.get("date") or ""), reverse=True)


def write_result(d: dict) -> None:
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULT.with_name(RESULT.name + ".tmp")
    tmp.write_text(json.dumps(dict(d, at=time.strftime("%Y-%m-%d %H:%M")), indent=1), encoding="utf-8")
    import os
    os.replace(tmp, RESULT)


def main(argv=None) -> int:
    global RESULT
    argv = sys.argv[1:] if argv is None else argv
    if not 1 <= len(argv) <= 2:
        print("usage: python -m launcher.calibrate <strata-<model>.json> [result.json]")
        return 2
    if len(argv) == 2:                       # the launcher names the file its job reads back
        RESULT = Path(argv[1]).expanduser()
    cfg_path = Path(argv[0]).expanduser()
    if not cfg_path.is_file():
        print(f"{cfg_path} is not a model config")
        return 2
    from .controller import strata                      # the same loader the launcher's API uses
    _m, s = strata(ROOT)
    S = s.setup_module()
    if S is None:
        print("setup.py could not be read: run START-HERE.bat --calibrate (Linux: ./setup.sh --calibrate) instead")
        write_result({"ok": False, "summary": "setup.py could not be read"})
        return 2
    if not hasattr(S, "calibrate_config"):
        print("this Strata's setup.py has no calibration (it needs engine 0.1.13 or newer setup)")
        write_result({"ok": False, "summary": "this setup has no calibration"})
        return 2
    print(f"measuring this PC for {cfg_path.name} - about 5-10 minutes, the PC is busy meanwhile", flush=True)
    try:
        ok = bool(S.calibrate_config(cfg_path))
    except Exception as e:                              # noqa: BLE001 - a failed measurement keeps the defaults
        print(f"the tuning did not finish ({e}): the default settings stay", flush=True)
        write_result({"ok": False, "summary": f"the tuning did not finish ({e})", "config": cfg_path.name})
        return 1
    write_result({"ok": ok, "config": cfg_path.name,
                  "summary": ("this PC is tuned: the settings are in " + cfg_path.name) if ok
                             else "the tuning failed: the default settings stay"})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
