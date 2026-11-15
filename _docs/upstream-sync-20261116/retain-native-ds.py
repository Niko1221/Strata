"""Restore the fork's native DS extension by reversing only named retirement deltas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("H:/Strata/evidence/upstream-sync-20261116/native-ds")
FORK = "6f13c7555a7d50289f0e3587ab780627c65a8000"
BASE = "f33c8c30588ff00e24b6b63bad0bf6a14da19081"
RETIREMENTS = ("e5806740a", "c6f84f281")
OWNED = ("src/", "include/", "metal/", "cmake/", "tests/dsv41/")


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, timeout=60)


def blob(ref: str, path: str) -> bytes | None:
    value = git("show", ref + ":" + path)
    return value.stdout if value.returncode == 0 else None


def merge(current: bytes, base: bytes, incoming: bytes) -> tuple[bytes, int]:
    with tempfile.TemporaryDirectory(prefix="native-diff3-", dir=OUT) as folder:
        paths = []
        for name, value in (("current", current), ("base", base), ("incoming", incoming)):
            value.decode("utf-8")
            path = Path(folder) / name
            path.write_bytes(value.replace(b"\r\n", b"\n"))
            paths.append(str(path))
        done = git("merge-file", "-p", "--diff3", *paths)
        if not 0 <= done.returncode <= 127:
            raise RuntimeError("Native compatibility diff3 failed")
        return done.stdout, done.returncode


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = OUT / "restoration.json"
    if receipt.exists():
        raise ValueError("Preserve restoration receipt")
    rows = []
    for commit in RETIREMENTS:
        changes = git("diff", "--name-status", commit + "^", commit).stdout.decode().splitlines()
        for line in changes:
            status, path = line.split("\t", 1)
            if not path.startswith(OWNED) or status not in ("D", "M"):
                continue
            target = ROOT / path
            if target.is_symlink() or target.is_junction():
                raise ValueError("Linked source refused")
            before = blob(commit + "^", path)
            if before is None:
                raise ValueError("Missing retirement parent")
            if status == "D":
                ours, base = blob(FORK, path), blob(BASE, path)
                if ours is not None and base is not None:
                    candidate, conflicts = merge(ours, base, before)
                else:
                    candidate, conflicts = before, 0
            else:
                current = target.read_bytes()
                if b"<<<<<<<" in current:
                    raise ValueError("Resolve current native conflict first: " + path)
                after = blob(commit, path)
                candidate, conflicts = merge(current, after, before)
            row = {"retirement": commit, "path": path, "operation": status,
                   "conflicts": conflicts, "sha256": hashlib.sha256(candidate).hexdigest()}
            if conflicts:
                review = OUT / commit / (path + ".diff3")
                review.parent.mkdir(parents=True, exist_ok=True)
                with review.open("xb") as stream:
                    stream.write(candidate)
                row["review_copy"] = str(review)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(candidate)
            rows.append(row)
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump({"fork": FORK, "base": BASE, "rows": rows}, stream, indent=2)
    print(json.dumps({"paths": len(rows), "conflicted": [r["path"] for r in rows if r["conflicts"]]}))


if __name__ == "__main__":
    main()
