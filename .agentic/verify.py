#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
if os.environ.get("AGENTIC_VERIFY_ACTIVE"):
    sys.exit("Recursive verification refused")
try:
    checks = json.loads((root / ".agentic/checks.json").read_text())["checks"]
    if not isinstance(checks, list) or not checks:
        raise ValueError("No checks configured; edit .agentic/checks.json")
    validated = []
    for c in checks:
        name, argv = c["name"], c["argv"]
        if not isinstance(name, str) or not name or not isinstance(argv, list) or not argv or not all(isinstance(a, str) and a for a in argv):
            raise ValueError("Each check needs a name and nonempty argv string array")
        cwd = (root / c.get("cwd", ".")).resolve()
        cwd.relative_to(root)
        timeout = c.get("timeout", 900)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        validated.append((name, argv, cwd, timeout))
except (ValueError, KeyError, TypeError, OSError) as e:
    print("UNCONFIGURED/INVALID:", e, file=sys.stderr)
    sys.exit(2)
failed = False
for name, argv, cwd, timeout in validated:
    print("RUN", name, repr(argv), flush=True)
    try:
        result = subprocess.run(argv, cwd=cwd, timeout=timeout,
                                env=dict(os.environ, AGENTIC_VERIFY_ACTIVE="1"))
        code = result.returncode
    except (OSError, subprocess.TimeoutExpired) as e:
        print(e, file=sys.stderr)
        code = 1
    print("PASS" if code == 0 else "FAIL", name, "exit", code, flush=True)
    failed |= code != 0
sys.exit(1 if failed else 0)
