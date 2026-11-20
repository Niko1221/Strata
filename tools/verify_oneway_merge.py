#!/usr/bin/env python3
"""Verify a one-way merge result: fork features intact, upstream fixes present.

Reads the manifest emitted by ``oneway_upstream_merge.py`` and checks the
applied worktree.  Exits non-zero when a required fork feature or an adopted
upstream file is missing, or when a written file's digest disagrees.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

LOG = logging.getLogger("strata.verify_oneway")
LOG_FMT = "%(asctime)s [%(levelname)s] [strata.verify_oneway] %(message)s"

# Anchors that must survive in the merged tree.  Paths are in the NEW layout.
FORK_ANCHORS: List[str] = [
    "engine/q38/src/engine.cu",
    "engine/q38/src/ops.cu",
    "engine/q38/include/strata/qwen38/admission.hpp",
    "engine/dsv41/src/ds4_main.cpp",
    "engine/q38/src/spin8/",
    "engine/q38/src/tqcodec/",
    "app/host/src/components/ChatModelPicker.tsx",
    "app/host/src/components/HakuaLlamaChat.tsx",
    "app/host/server/q38-dense-contract.mjs",
    "app/host/server/memory.mjs",
    "terminal/strata_control/qwen38_dense.py",
    "terminal/strata_control/backends/",
    "terminal/serve/memory_harness.py",
    "schemas/fork-extension.schema.json",
    "tools/packaging/full_windows.py",
    "_docs/upstream-intake/fork-contracts.json",
]

# Upstream v0.2.16 security / correctness files that must be adopted.
UPSTREAM_ANCHORS: List[str] = [
    "engine/CMakeLists.txt",
    "engine/include/strata/artifact/model_family.hpp",
    "engine/include/strata/core/family_layout.hpp",
    "engine/include/strata/core/kpack_path.hpp",
    "terminal/strata_control/signing.py",
    "terminal/strata_control/update_keys.json",
    "terminal/strata_control/engine_component.py",
    "terminal/strata_control/netpolicy.py",
    "terminal/strata_control/paths.py",
    "terminal/serve/owner_guard.py",
    "terminal/tests/terminal/test_doctor_signed_engine.py",
    "app/host/server/control-runtime.mjs",
    "app/host/src/components/GuidedSetup.tsx",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="applied isolated worktree")
    parser.add_argument("--manifest", required=True, help="manifest.json from the merge")
    parser.add_argument("--json", dest="json_out", help="write verification report here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format=LOG_FMT, stream=sys.stderr)
    repo, manifest_path = Path(args.repo).resolve(), Path(args.manifest).resolve()
    manifest: Dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: List[str] = []
    checks: Dict[str, Any] = {}

    missing_fork = [p for p in FORK_ANCHORS if not (repo / p).exists()]
    missing_up = [p for p in UPSTREAM_ANCHORS if not (repo / p).exists()]
    checks["fork_anchors"] = {"total": len(FORK_ANCHORS), "missing": missing_fork}
    checks["upstream_anchors"] = {"total": len(UPSTREAM_ANCHORS), "missing": missing_up}
    failures += [f"fork anchor missing: {p}" for p in missing_fork]
    failures += [f"upstream anchor missing: {p}" for p in missing_up]

    mismatches: List[str] = []
    checked = 0
    for row in manifest.get("decisions", []):
        # Only decisions that actually won their target are written.  Two
        # source paths can legitimately map to one target (rename + copy).
        if not row.get("applied", False) or not row.get("sha256"):
            continue
        target = repo / row["target"]
        if not target.exists():
            mismatches.append(f"missing: {row['target']}")
            continue
        checked += 1
        if sha256(target) != row["sha256"]:
            mismatches.append(f"digest mismatch: {row['target']}")
    checks["digest_check"] = {"checked": checked, "failures": mismatches[:40],
                              "count": len(mismatches)}
    failures += mismatches[:40]

    conflicts = [r["path"] for r in manifest.get("decisions", []) if r.get("conflicts")]
    checks["conflicts"] = {"count": len(conflicts), "paths": conflicts[:40]}

    report = {"repo": str(repo), "manifest": str(manifest_path),
              "ok": not failures, "failures": failures, "checks": checks}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "failures": len(failures),
                      "checks": {k: (len(v) if isinstance(v, list) else v)
                                 for k, v in checks.items()}}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
