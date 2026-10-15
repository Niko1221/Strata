"""Paths and settings the KV measurement harnesses share, so no script carries a machine-specific path.

Every value has an environment override and a default that follows the repo's own layout:

  ROOT      the repository (this file's parent's parent).
  DATA      $STRATA_DATA, else the Strata-data folder next to the repository - the layout SETUP and
            tools/strata_mcp.py use for model files.
  PY        the interpreter running the harness: the serve it launches runs under the same one.
  KEY       $STRATA_API_KEY, else DATA/strata_api_key.txt if it exists, else "" (a serve without an api_key).
  CONFIG    $KV_CONFIG, else ROOT/strata-swift-iq3_xxs.json - the strata-<model>.json the serve runs. The
            harnesses copy it and override only the arguments an arm changes (--kv, --max-context, ...).
  engine()  an engine binary: $<env> if set, else ROOT/<relative>/strata(.exe).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("STRATA_DATA") or (ROOT.parent / "Strata-data"))
PY = Path(sys.executable)
EXE = "strata.exe" if os.name == "nt" else "strata"
CONFIG = Path(os.environ.get("KV_CONFIG") or (ROOT / "strata-swift-iq3_xxs.json"))


def _key() -> str:
    if os.environ.get("STRATA_API_KEY"):
        return os.environ["STRATA_API_KEY"].strip()
    f = DATA / "strata_api_key.txt"
    return f.read_text(encoding="utf-8").strip() if f.is_file() else ""


KEY = _key()


def base_config() -> dict:
    """The serve config the arms are derived from (a fresh copy each call)."""
    if not CONFIG.is_file():
        sys.exit(f"KV harness: no serve config at {CONFIG} - set KV_CONFIG to your strata-<model>.json")
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def engine(env: str, relative: str) -> str:
    """$env if set, else ROOT/relative/strata(.exe) - e.g. engine("KV_DEV_ENGINE", "build")."""
    return os.environ.get(env) or str(ROOT / relative / EXE)
