#!/usr/bin/env python3
"""Refresh hashes for the distributable report package."""
from pathlib import Path
import hashlib
import json

BASE = Path(__file__).resolve().parent.parent
MANIFEST = BASE / "SOURCE-MANIFEST.json"


def include(path: Path) -> bool:
    rel = path.relative_to(BASE)
    if rel.as_posix() == "SOURCE-MANIFEST.json":
        return False
    if "__pycache__" in rel.parts or path.suffix == ".pyc":
        return False
    if rel.parts[0] == "qa" and (path.name.startswith("page-") or path.name.startswith("final-panel-")) and path.suffix == ".png":
        return False
    return True


files = []
for path in sorted((p for p in BASE.rglob("*") if p.is_file() and include(p)), key=lambda p: p.relative_to(BASE).as_posix()):
    data = path.read_bytes()
    files.append({
        "path": path.relative_to(BASE).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    })

manifest = {
    "schema_version": 1,
    "note": "Hashes cover distributed bytes; volatile QA page/panel rasters and this manifest are excluded.",
    "files": files,
}
MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"manifest": str(MANIFEST), "files": len(files), "bytes": sum(x["bytes"] for x in files)}, sort_keys=True))
