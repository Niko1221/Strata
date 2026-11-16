"""Explicit, source/compiler-bound per-entry CUDA resource measurements."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOTS = ("src", "include", "cmake", "spin8/src", "spin8/include", "spin8/third_party", "tqcodec/src", "tqcodec/include")
TOP = ("CMakeLists.txt", "spin8/CMakeLists.txt", "tqcodec/CMakeLists.txt")

def source_digest(root):
    root = Path(root).resolve()
    if not (root / "CMakeLists.txt").is_file():
        raise ValueError("CUDA baseline source root has no CMakeLists.txt")
    paths = [p for name in ROOTS if (root/name).exists() for p in (root/name).rglob("*") if p.is_file()]
    paths += [root/name for name in TOP if (root/name).is_file()]
    rows = {}
    for path in paths:
        if path.is_symlink():
            raise ValueError("Linked baseline source input")
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        raw = path.read_bytes()
        if b"\0" not in raw:
            raw = raw.replace(b"\r\n", b"\n")
        rows[path.relative_to(root).as_posix()] = hashlib.sha256(raw).hexdigest()
    data = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()

def dependency_commit(root):
    root = Path(root)
    # Official builds now use a vendored, patched ggml tree. Bind its actual
    # contents rather than accidentally using the enclosing Strata Git HEAD.
    if (root / "UPSTREAM.md").is_file() and (root / "ggml/CMakeLists.txt").is_file():
        for path in (root, *root.parents):
            if path.is_symlink() or path.is_junction():
                raise ValueError("Linked vendored ggml source input")
        rows = {}
        for directory, folders, files in os.walk(root, followlinks=False):
            for name in folders + files:
                path = Path(directory) / name
                if path.is_symlink() or path.is_junction():
                    raise ValueError("Linked vendored ggml source input")
            for name in files:
                path = Path(directory) / name
                raw = path.read_bytes()
                if b"\0" not in raw:
                    raw = raw.replace(b"\r\n", b"\n")
                rows[path.relative_to(root).as_posix()] = hashlib.sha256(raw).hexdigest()
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        return "vendored-sha256:" + digest
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    if dirty or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("CUDA baseline requires an identified clean ggml source")
    return revision

def compiler_key(path):
    key = Path(path).read_text(encoding="utf-8-sig").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", key):
        raise ValueError("Invalid generated CUDA compiler/configuration key")
    return key

def entry_key(library, obj, sm, function):
    base = lambda value: str(value).replace("\\", "/").rsplit("/", 1)[-1]
    return "|".join((base(library), base(obj), sm, function))

def bindings(source_root, key_file, ggml_source):
    return {"sourceDigest": source_digest(source_root), "compilerKey": compiler_key(key_file),
            "ggmlCommit": dependency_commit(ggml_source)}

def load(path, expected):
    path = Path(path)
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("CUDA baseline is unexpectedly large")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != 1 or record.get("bindings") != expected:
        raise ValueError("CUDA baseline source, compiler/configuration or ggml identity changed; remeasure and review")
    entries = record.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("CUDA baseline has no measurements")
    result = {}
    for key, value in entries.items():
        if not isinstance(key, str) or not isinstance(value, list) or len(value) != 2:
            raise ValueError("Invalid CUDA baseline entry")
        if any(type(v) is not int or v < 0 or v > 1048576 for v in value):
            raise ValueError("Invalid CUDA baseline resource amount")
        result[key] = tuple(value)
    return result

def save(path, expected, entries, libraries):
    record = {"schema": 1, "bindings": expected, "entries": dict(sorted(entries.items())),
              "measuredLibraries": libraries,
              "scope": "Explicit per-library/object/kernel/SM measurements. No automatic waiver; changes require remeasurement and review."}
    with Path(path).open("x", encoding="utf-8", newline="\n") as output:
        json.dump(record, output, indent=2, sort_keys=True)
        output.write("\n")
