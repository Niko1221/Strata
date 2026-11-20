"""tests/models/test_bundle_identity.py - Tests for streaming verification and drift detection."""
from __future__ import annotations

from pathlib import Path
import pytest

from strata_control.bundle_resolver import ImportSelection, resolve_bundle
from strata_control.bundle_identity import verify_bundle, VerifyPolicy


def test_bundle_identity_verification(tmp_path: Path) -> None:
    f = tmp_path / "test.gguf"
    f.write_bytes(b"content-v1")

    bundle = resolve_bundle(ImportSelection(primary_path=f))
    verified = verify_bundle(bundle)

    assert verified.model_set_id == bundle.model_set_id
    assert len(verified.canonical_hash) == 64
    assert verified.verified_files[0].sha256 != ""


def test_drift_detected_on_file_change(tmp_path: Path) -> None:
    f = tmp_path / "drift.gguf"
    f.write_bytes(b"original-content")

    bundle = resolve_bundle(ImportSelection(primary_path=f))
    # Pre-record expected sha256
    verified = verify_bundle(bundle)

    # Modify file content behind the back
    f.write_bytes(b"tampered-content")

    # Re-verifying a bundle that has fixed expected sha256
    drifted_bundle = resolve_bundle(ImportSelection(primary_path=f))
    # Replace file with old sha expecting original
    from strata_control.bundle_resolver import FileMember, ModelBundle
    tampered_bundle = ModelBundle(
        schema_version="1.0",
        model_set_id=bundle.model_set_id,
        artifact_kind=bundle.artifact_kind,
        architecture=bundle.architecture,
        dialect=bundle.dialect,
        split_total=1,
        files=[FileMember(role="shard", ordinal=0, path=str(f), file_size=f.stat().st_size, sha256=verified.verified_files[0].sha256)],
    )
    with pytest.raises(ValueError, match="Content drift detected"):
        verify_bundle(tampered_bundle)
