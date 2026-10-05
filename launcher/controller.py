"""launcher/controller.py - the launcher's connection to Strata itself.

The launcher does not re-implement "what is installed", "start this model", "what does this PC have": those answers
already exist in `tools/strata_mcp.py`, the stdlib-only controller the MCP server uses so an AI assistant can install,
start, check and stop Strata.  A second front-end should ask the same questions, so this file only imports that
controller and hands it out.  The web page is the only new front-end; the model tables, the fit checks, the install
job and the start/stop logic stay where they are.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_mcp = None


def strata_module():
    """tools/strata_mcp.py as a module (it is a script with a main() guard, and stdlib-only)."""
    global _mcp
    if _mcp is None:
        if str(ROOT / "tools") not in sys.path:
            sys.path.insert(0, str(ROOT / "tools"))
        import strata_mcp                                  # noqa: WPS433 - the shared controller
        _mcp = strata_mcp
    return _mcp


def strata(root=None):
    """(the controller module, a Strata folder bound to `root`)."""
    m = strata_module()
    return m, m.Strata(Path(root).resolve() if root else ROOT)
