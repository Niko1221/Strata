"""Prepare and apply reviewed per-file diff3 results for the fixed official intake.

This adapts intake-hotfix.py's immutable-blob merge. Preparation writes only a
fresh private review directory; application requires a complete decision record
and unchanged physical originals. Neither action modifies Git's index or refs.
Run only after the controller has admitted the current resource conditions.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("H:/Strata/private-backups/cleanup-20261116/upstream-c775-intake-v1")
REFS = {
    "base": "baf16363b4403ef0ed4f538f5d5ed11c43fb9535",
    "fork": "d552b56ab0218a613dc6c287947022adf4bd4164",
    "official": "c77572806996349b49995e4f98d41b5294d36a21",
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                          check=check, timeout=120)


def contained(root: Path, name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("Unsafe repository-relative path")
    target = root.joinpath(*path.parts)
    for item in (target, *target.parents):
        if item.is_symlink() or item.is_junction():
            raise ValueError("Linked intake path")
        if item == root:
            break
    return target


def normalized(raw: bytes) -> bytes:
    raw.decode("utf-8")
    if b"\0" in raw:
        raise ValueError("Opaque input requires a separate explicit decision")
    return raw.replace(b"\r\n", b"\n")


def exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("Existing evidence differs: " + str(path))
        return  # Resume only identical immutable preparation artifacts.
    with path.open("xb") as stream:
        stream.write(raw)


def check_head() -> None:
    if git("rev-parse", "HEAD").stdout.decode().strip() != REFS["fork"]:
        raise ValueError("Fork source changed")
    if git("diff", "--cached", "--name-only").stdout.strip():
        raise ValueError("Index must remain unchanged during intake")


def prepare() -> None:
    check_head()
    for value in REFS.values():
        git("cat-file", "-e", value + "^{commit}")
    git("merge-base", "--is-ancestor", REFS["base"], REFS["official"])
    if git("rev-list", "--count", REFS["base"] + ".." + REFS["official"]).stdout.strip() != b"45":
        raise ValueError("Unexpected official intake history")
    if git("diff", "--name-only").stdout.strip():
        raise ValueError("Tracked source must be clean before originals are frozen")
    if (OUT / "applied.json").exists():
        raise ValueError("This intake attempt has already been applied")
    OUT.mkdir(parents=True, exist_ok=True)
    fields = git("diff", "--name-status", "--no-renames", "-z", REFS["base"], REFS["official"]).stdout.decode().split("\0")
    rows = {}
    for status, name in zip(fields[0:-1:2], fields[1::2]):
        if status not in ("A", "M"):
            raise ValueError("Unexpected removal/mode transition requires separate review")
        directory = contained(OUT / "merge", name)
        target = contained(ROOT, name)
        original = target.read_bytes() if target.exists() else None
        if original is not None:
            exclusive(contained(OUT / "original", name), original)
        inputs = {}
        for label, ref in REFS.items():
            result = git("show", ref + ":" + name, check=False)
            present = result.returncode == 0
            if not present and git("ls-tree", ref, "--", name).stdout.strip():
                raise ValueError("Cannot read identified input blob")
            raw = result.stdout if present else b""
            exclusive(directory / (label + ".raw"), raw)
            exclusive(directory / label, normalized(raw))
            inputs[label] = {"ref": ref, "present": present, "bytes": len(raw), "sha256": digest(raw)}
        if original is not None and normalized(original) != (directory / "fork").read_bytes():
            raise ValueError("Physical source differs from frozen fork: " + name)
        if original is None and inputs["fork"]["present"]:
            raise ValueError("Missing physical fork source: " + name)
        merged = git("merge-file", "-p", "--diff3", "-L", "fork", "-L", "base", "-L", "official",
                     str(directory / "fork"), str(directory / "base"), str(directory / "official"), check=False)
        if not 0 <= merged.returncode <= 127:
            raise RuntimeError("diff3 failed: " + name)
        exclusive(directory / "candidate", merged.stdout)
        rows[name] = {"status": status, "inputs": inputs, "conflicts": merged.returncode,
                      "originalSHA256": digest(original) if original is not None else None,
                      "originalBytes": len(original) if original is not None else None,
                      "candidateSHA256": digest(merged.stdout),
                      "eol": "preserve_matching_original_lines"}
    exclusive(OUT / "preview.json", json.dumps({"refs": REFS, "rows": rows}, indent=2).encode())
    print(json.dumps({"files": len(rows), "conflicts": {n: r["conflicts"] for n, r in rows.items() if r["conflicts"]}}, indent=2))


def apply() -> None:
    check_head()
    if (OUT / "applied.json").exists():
        raise ValueError("This intake attempt has already been applied")
    manifest = json.loads((OUT / "preview.json").read_text())
    decisions = json.loads((OUT / "decisions.json").read_text())
    if manifest["refs"] != REFS or set(decisions) != set(manifest["rows"]):
        raise ValueError("Exact source and complete per-path decisions required")
    writes = []
    for name, row in manifest["rows"].items():
        target = contained(ROOT, name)
        current = target.read_bytes() if target.exists() else None
        if (digest(current) if current is not None else None) != row["originalSHA256"]:
            raise ValueError("Original changed after preview: " + name)
        decision = decisions[name]
        if not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
            raise ValueError("A semantic review ruling is required: " + name)
        resolved = contained(OUT / "resolved", name).read_bytes()
        if digest(resolved) != decision.get("sha256"):
            raise ValueError("Reviewed candidate changed: " + name)
        text = normalized(resolved)
        if any(line.startswith((b"<<<<<<< ", b"||||||| ", b">>>>>>> ")) for line in text.splitlines()):
            raise ValueError("Unresolved conflict: " + name)
        # Some committed fork files have mixed EOLs. Retain every matching
        # original line byte-for-byte; new/changed lines use the majority EOL.
        before = current.splitlines(keepends=True) if current is not None else []
        after = text.splitlines(keepends=True)
        crlf = sum(line.endswith(b"\r\n") for line in before)
        lf = sum(line.endswith(b"\n") and not line.endswith(b"\r\n") for line in before)
        final_lines = [line.replace(b"\n", b"\r\n") if crlf > lf else line for line in after]
        matcher = difflib.SequenceMatcher(None, [normalized(line) for line in before], after, autojunk=False)
        for block in matcher.get_matching_blocks():
            final_lines[block.b:block.b + block.size] = before[block.a:block.a + block.size]
        final = b"".join(final_lines)
        writes.append((target, final, name, decision["reason"]))
    results = {}
    for target, raw, name, reason in writes:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        results[name] = {"bytes": len(raw), "sha256": digest(raw), "reason": reason}
    exclusive(OUT / "applied.json", json.dumps({"refs": REFS, "files": results}, indent=2).encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "apply"))
    args = parser.parse_args()
    {"prepare": prepare, "apply": apply}[args.action]()
