"""Record the selected Git base, source drift and offline handoff checks (not Strata tests)."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = "99f3dbd0b21d1401b3769e0c0d963913607f380b"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--handoff", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    def git(*args):
        return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}",
                                        "-C", str(ROOT), *args])

    remote = git("ls-remote", "upstream", "refs/heads/main").decode().split()[0]
    if remote != BASE or git("rev-parse", "HEAD").decode().strip() != BASE:
        raise SystemExit("The chosen base or remote main changed; inspect before proceeding")
    baseline = json.loads((a.handoff / "research/baseline.json").read_text(encoding="utf-8"))
    files = []
    for f in baseline["files"]:
        blob = git("show", f"{BASE}:{f['path']}")
        working = (ROOT / f["path"]).read_bytes()
        sha = hashlib.sha256(blob).hexdigest()
        files.append({"path": f["path"], "snapshot_sha256": f["sha256"], "git_blob_sha256": sha,
                      "worktree_sha256": hashlib.sha256(working).hexdigest(),
                      "snapshot_status": "match" if sha == f["sha256"] else "changed",
                      "working_bytes_equal_git_blob": working == blob})
    report = {"time_utc": datetime.now(timezone.utc).isoformat(), "base_ref": "upstream/main",
              "base_commit": BASE, "remote_main_commit": remote, "branch": "work/responses-api",
              "worktree": "<WORKSPACE>/strata-responses-stateless",
              "source_checkout": "<WORKSPACE>/strata-responses", "source_checkout_preserved": True,
              "remote_operations": "ls-remote only; no fetch/push",
              "remotes": git("remote", "-v").decode().splitlines(),
              "worktrees": git("worktree", "list", "--porcelain").decode().replace(
                  str(ROOT.parent).replace("\\", "/"), "<WORKSPACE>"),
              "snapshot_archive": baseline["archive"], "snapshot_is_trusted_commit": False,
              "files": files, "python": platform.python_version(), "os": platform.system(),
              "native_engine_tested": False, "gpu_contacted": False}
    (a.out / "source-audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (a.out / "python-lock.txt").write_text(freeze, encoding="utf-8")
    checks = []
    for name, args in [("handoff-pack", ["tools/check_pack.py"]),
                       ("handoff-tests", ["-m", "unittest", "discover", "-s", "tests", "-v"])]:
        run = subprocess.run([sys.executable, *args], cwd=a.handoff, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=120)
        (a.out / (name + ".txt")).write_text((run.stdout + run.stderr).replace(str(a.handoff), "<HANDOFF>"),
                                             encoding="utf-8")
        checks.append({"command": ["python", *args], "return_code": run.returncode,
                       "scope": "handoff tooling only, not Strata implementation"})
    (a.out / "handoff-checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(f"Base {BASE}; snapshot drift in {sum(f['snapshot_status'] != 'match' for f in files)}/{len(files)} files")
    print(json.dumps(checks, indent=2))
    raise SystemExit(any(r["return_code"] for r in checks))


if __name__ == "__main__":
    main()
