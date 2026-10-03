"""Run selected CPU contract/regression suites in separate processes; save raw evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ["serve.test_security", "serve.test_server", "serve.test_lifecycle",
            "serve.test_structured", "serve.test_monitor", "serve.test_mcp", "serve.test_detok"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("modules", nargs="*", default=BASELINE)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    results = []
    for module in a.modules:
        command = [sys.executable, "-m", "unittest", module, "-v"]
        run = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             encoding="utf-8", errors="replace", timeout=240)
        filename = module.replace(".", "-") + ".txt"
        (a.out / filename).write_text(run.stdout.replace(str(ROOT), "<WORKTREE>"), encoding="utf-8")
        results.append({"module": module, "command": ["python", *command[1:]],
                        "return_code": run.returncode, "evidence": filename})
        print(f"{module}: exit {run.returncode}", flush=True)
    (a.out / "test-results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(any(r["return_code"] for r in results))


if __name__ == "__main__":
    main()
