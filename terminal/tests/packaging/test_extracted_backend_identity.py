"""tests/packaging/test_extracted_backend_identity.py - TDD tests for Task R18 Distribution Integrity.
Conforms strictly to MILSPEC-Codex standard: logging only, zero print, strict types.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
from typing import Any, Dict

from strata_control.packaging_verifier import (
    ReleasePackageManifest,
    verify_package_distribution,
    PackageVerificationResult,
)


def test_missing_or_stale_backend_binary_rejected(tmp_path: Path) -> None:
    """Advertised backend binary missing from extraction directory must be rejected."""
    manifest = ReleasePackageManifest(
        version="0.3.0",
        commit_sha="45b8628226f35051195520ab418e55be92093ccf",
        binaries={
            "strata_engine": {"path": "bin/strata_engine.exe", "sha256": "a" * 64},
            "strata_dsv41": {"path": "bin/strata_dsv41.exe", "sha256": "b" * 64},
        },
    )
    # staging directory is empty
    result = verify_package_distribution(manifest, extract_root=tmp_path)
    assert result.valid is False
    assert any("missing binary" in err.lower() for err in result.errors)


def test_placeholder_hash_in_release_manifest_rejected(tmp_path: Path) -> None:
    """Placeholder or invalid hash (e.g. 'native-sha') in release manifest must fail verification."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    fake_exe = bin_dir / "strata_engine.exe"
    fake_exe.write_bytes(b"MZ\x90\x00" + b"\x00" * 100)

    manifest = ReleasePackageManifest(
        version="0.3.0",
        commit_sha="45b8628226f35051195520ab418e55be92093ccf",
        binaries={
            "strata_engine": {"path": "bin/strata_engine.exe", "sha256": "native-sha"},
        },
    )
    result = verify_package_distribution(manifest, extract_root=tmp_path)
    assert result.valid is False
    assert any("placeholder" in err.lower() or "invalid sha256" in err.lower() for err in result.errors)


def test_dev_path_leak_rejected(tmp_path: Path) -> None:
    """Packaging manifest cannot contain absolute developer paths (e.g. C:\\Users\\... or H:\\...)."""
    manifest = ReleasePackageManifest(
        version="0.3.0",
        commit_sha="45b8628226f35051195520ab418e55be92093ccf",
        binaries={
            "strata_engine": {"path": "C:\\Users\\downl\\build\\strata_engine.exe", "sha256": "a" * 64},
        },
    )
    result = verify_package_distribution(manifest, extract_root=tmp_path)
    assert result.valid is False
    assert any("absolute or developer path" in err.lower() for err in result.errors)


def test_verified_package_distribution_passes(tmp_path: Path) -> None:
    """Proper relative paths and cryptographic SHA-256 match pass package verification."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    exe_file = bin_dir / "strata_engine.exe"
    payload = b"MZ\x90\x00RealStrataBinaryPayload"
    exe_file.write_bytes(payload)

    import hashlib
    actual_hash = hashlib.sha256(payload).hexdigest()

    manifest = ReleasePackageManifest(
        version="0.3.0",
        commit_sha="45b8628226f35051195520ab418e55be92093ccf",
        binaries={
            "strata_engine": {"path": "bin/strata_engine.exe", "sha256": actual_hash},
        },
    )
    result = verify_package_distribution(manifest, extract_root=tmp_path)
    assert result.valid is True
    assert len(result.errors) == 0
