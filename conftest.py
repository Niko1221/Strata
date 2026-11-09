r"""pytest: setup.py's data-folder migration (#198) saves the data folder it settled on in the user's settings file
(%APPDATA%\Strata\settings.json, ~/.config/strata/settings.json) and can MOVE model files from a folder it finds there.
A test that reaches it must never touch the real file, so the whole session gets its own config home."""
import os
import tempfile

_home = tempfile.mkdtemp(prefix="strata-pytest-config-")
os.environ["APPDATA"] = _home
os.environ["XDG_CONFIG_HOME"] = _home
# The per-user STATE and DATA folders too (%LOCALAPPDATA%\Strata: tuning/recheck.json, the engine store, the update records; ~/.local/state, ~/.local/share).
# A test that installs a stub engine marks saved tuning presets for a re-check there: on 2026-10-24 one such run left an entry in the owner's real
# tuning\recheck.json. A test that wants a place of its own still names it (STRATA_COMPONENTS_DIR, state_base=...).
os.environ["LOCALAPPDATA"] = _home
os.environ["XDG_STATE_HOME"] = _home
os.environ["XDG_DATA_HOME"] = _home
# Run from PowerShell 7 (GitHub's Windows runners), PSModulePath names PowerShell 7's module folders first: a Windows PowerShell 5.1 child that a
# test starts (install.ps1, fetch-node.ps1) then finds their Microsoft.PowerShell.Utility/Security and cannot load them ("Get-FileHash is not
# recognized"). Keep only the folders that are not PowerShell 7's, as in a Windows PowerShell or cmd window.
if os.name == "nt" and os.environ.get("PSModulePath"):
    _keep = [p for p in os.environ["PSModulePath"].split(os.pathsep)
             if p and not ("\\powershell\\" in p.lower().replace("/", "\\") and "\\windowspowershell\\" not in p.lower().replace("/", "\\"))]
    os.environ["PSModulePath"] = os.pathsep.join(_keep)
