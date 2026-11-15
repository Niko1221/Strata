"""Python-driven diff3 from immutable base/fork/upstream blobs; review copies only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REFS = {
    "base": "f33c8c30588ff00e24b6b63bad0bf6a14da19081",
    "fork": "6f13c7555a7d50289f0e3587ab780627c65a8000",
    "upstream": "09b603886925994d0d4461857b01321ef3aa431c",
}
OUT = Path("H:/Strata/evidence/upstream-sync-20261116/merge")


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, timeout=60
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="repository-relative text file")
    parser.add_argument("--normalize-eol", action="store_true")
    args = parser.parse_args()
    path = PurePosixPath(args.path.replace("\\", "/"))
    if path.is_absolute() or any(p in ("..", ".") for p in path.parts) or ":" in str(path):
        parser.error("contained repository-relative path required")
    OUT.mkdir(parents=True, exist_ok=True)
    rows, blobs = {}, {}
    for label, sha in REFS.items():
        value = git("show", f"{sha}:{path}")
        if value.returncode:
            rows[label] = {"ref": sha, "status": "ABSENT"}
            continue
        blobs[label] = value.stdout
        rows[label] = {
            "ref": sha,
            "status": "PRESENT",
            "bytes": len(value.stdout),
            "sha256": hashlib.sha256(value.stdout).hexdigest(),
        }
    report = {"path": str(path), "inputs": rows, "writes_source": False,
              "normalized_eol": args.normalize_eol}
    if len(blobs) == 3:
        with tempfile.TemporaryDirectory(prefix="diff3-", dir=OUT) as folder:
            files = {}
            for label, blob in blobs.items():
                files[label] = Path(folder) / label
                if args.normalize_eol:
                    blob.decode("utf-8")  # Never normalize binary or opaque inputs.
                    blob = blob.replace(b"\r\n", b"\n")
                files[label].write_bytes(blob)
            result = git(
                "merge-file", "-p", "--diff3", "-L", "fork", "-L", "base",
                "-L", "upstream", str(files["fork"]), str(files["base"]),
                str(files["upstream"]),
            )
            if result.returncode < 0 or result.returncode > 127:
                raise RuntimeError("diff3 failed")
            target = OUT / (str(path) + (".normalized" if args.normalize_eol else ""))
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(result.stdout)
            report.update(review_copy=str(target), conflict_count=result.returncode)
    else:
        report.update(reason="add_delete_requires_semantic_resolution")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
