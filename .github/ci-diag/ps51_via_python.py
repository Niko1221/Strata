"""DIAGNOSTIC ONLY (ci/checks-green; dropped before main). Windows PowerShell 5.1 children started by PYTHON (as setup, strata_cli, strata_control
and strata_mcp do), from a pwsh step, with the environment as inherited (raw) and with PowerShell 7's PSModulePath entries dropped (kept)."""
import base64
import json
import os
import subprocess
from pathlib import Path

probes = json.loads((Path(__file__).with_name("ps51_probes.json")).read_text(encoding="utf-8"))
exe = str(Path(os.environ["SystemRoot"]) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe")


def ps7(p):
    low = p.lower().replace("/", "\\")
    return "\\powershell\\" in low and "\\windowspowershell\\" not in low


kept = os.pathsep.join(p for p in os.environ.get("PSModulePath", "").split(os.pathsep) if p and not ps7(p))
print("python parent; PSModulePath as inherited:", os.environ.get("PSModulePath"), flush=True)
for name, env in (("raw", dict(os.environ)), ("kept", {**os.environ, "PSModulePath": kept})):
    print(f"=== python -> powershell.exe, {name}", flush=True)
    for what, script in probes:
        enc = base64.b64encode(script.encode("utf-16-le")).decode()
        r = subprocess.run([exe, "-NoProfile", "-NonInteractive", "-EncodedCommand", enc], env=env, capture_output=True, text=True,
                           errors="replace", timeout=60, stdin=subprocess.DEVNULL)
        first = next((x for x in r.stderr.splitlines() if x.strip()), "")
        print(f"{name:<6} rc={r.returncode} out={' '.join(r.stdout.split())[:300]} err={first[:200]}", flush=True)
        print(f"       {what}", flush=True)
