"""strata_control/packaging_verifier.py - Verifies packaging integrity, binary digests, and distribution isolation.
Strictly conforms to MILSPEC-Codex standard: logging only, zero raw print, strict types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("strata.packaging")

DISALLOWED_HASHES = frozenset({
    "native-sha",
    "placeholder",
    "todo",
    "none",
    "unknown",
    "",
})


@dataclass
class ReleasePackageManifest:
    version: str
    commit_sha: str
    binaries: Dict[str, Dict[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PackageVerificationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    verified_binaries: Dict[str, str] = field(default_factory=dict)


def verify_package_distribution(
    manifest: ReleasePackageManifest,
    extract_root: Path
) -> PackageVerificationResult:
    """Verify that all extracted binaries in a release package match declared digests and lack developer path leaks."""
    errors: List[str] = []
    verified: Dict[str, str] = {}

    for name, bin_spec in manifest.binaries.items():
        rel_path_str = bin_spec.get("path", "")
        declared_sha = str(bin_spec.get("sha256", "")).strip().lower()

        # 1. Reject placeholder hashes
        if declared_sha in DISALLOWED_HASHES or len(declared_sha) != 64 or not re.fullmatch(r"[0-9a-f]{64}", declared_sha):
            err = f"Placeholder or invalid sha256 for binary '{name}': {declared_sha!r}"
            logger.error(err)
            errors.append(err)
            continue

        # 2. Reject absolute or developer paths
        p = Path(rel_path_str)
        if p.is_absolute() or re.match(r"^[A-Za-z]:", rel_path_str) or "Users" in rel_path_str or "build" in rel_path_str or ".." in rel_path_str:
            err = f"Absolute or developer path leak in binary '{name}': {rel_path_str}"
            logger.error(err)
            errors.append(err)
            continue

        # 3. Check physical presence under extract_root
        target_path = extract_root / p
        if not target_path.is_file():
            err = f"Missing binary file under extraction root: {p} (full path: {target_path})"
            logger.error(err)
            errors.append(err)
            continue

        # 4. Stream and verify SHA-256
        h = hashlib.sha256()
        try:
            with open(target_path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    h.update(chunk)
            actual_sha = h.hexdigest().lower()
            if actual_sha != declared_sha:
                err = f"Binary digest mismatch for '{name}': declared {declared_sha} != actual {actual_sha}"
                logger.error(err)
                errors.append(err)
            else:
                verified[name] = actual_sha
                logger.info("Binary verified: %s -> %s", name, actual_sha)
        except OSError as e:
            err = f"Failed to read binary '{name}' at {target_path}: {e}"
            logger.error(err)
            errors.append(err)

    return PackageVerificationResult(
        valid=len(errors) == 0,
        errors=errors,
        verified_binaries=verified,
    )
