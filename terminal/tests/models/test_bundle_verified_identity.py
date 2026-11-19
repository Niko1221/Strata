from pathlib import Path
import pytest

from strata_control.bundle_identity import (
    VerifyPolicy,
    compute_file_sha256,
    verify_bundle,
)
from strata_control.bundle_resolver import FileMember, ImportSelection, resolve_bundle
from tests.fixtures.gguf_builder import MinimalGgufBuilder


def test_streaming_sha256_computation(tmp_path: Path):
    """Verify streaming SHA-256 calculation matches hashlib."""
    import hashlib
    data = b"GGUF_TEST_DATA_STREAM" * 10000
    f = tmp_path / "model.gguf"
    f.write_bytes(data)

    expected = hashlib.sha256(data).hexdigest()
    actual = compute_file_sha256(f, chunk_size=1024)
    assert actual == expected


def test_empty_sha256_with_no_full_hash_rejected(tmp_path: Path):
    """Verify that verify_bundle rejects creating a VerifiedBundle if full_sha256 is False and no trusted sha is present."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 32)

    selection = ImportSelection(primary_path=f)
    bundle = resolve_bundle(selection)

    # If policy disables full hash but member has no sha256, it must fail
    policy = VerifyPolicy(full_sha256=False)
    with pytest.raises(ValueError, match="sha256"):
        verify_bundle(bundle, policy=policy)


def test_content_drift_detected(tmp_path: Path):
    """Verify that if an expected sha256 is specified and file differs, ValueError is raised."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 32)

    selection = ImportSelection(primary_path=f)
    bundle = resolve_bundle(selection)

    # Manually provide an expected sha256 that does not match
    tampered_files = [
        FileMember(
            role=fm.role,
            ordinal=fm.ordinal,
            path=fm.path,
            file_size=fm.file_size,
            sha256="0" * 64,
        )
        for fm in bundle.files
    ]
    tampered_bundle = type(bundle)(
        schema_version=bundle.schema_version,
        model_set_id=bundle.model_set_id,
        artifact_kind=bundle.artifact_kind,
        architecture=bundle.architecture,
        dialect=bundle.dialect,
        split_total=bundle.split_total,
        files=tampered_files,
    )

    with pytest.raises(ValueError, match="Content drift detected"):
        verify_bundle(tampered_bundle)


def test_symlink_rejection_when_disallowed(tmp_path: Path):
    """Verify that symlinks are rejected when allow_symlinks=False."""
    real_file = tmp_path / "real.gguf"
    real_file.write_bytes(b"GGUF" + b"\x00" * 32)

    symlink_file = tmp_path / "symlink.gguf"
    try:
        symlink_file.symlink_to(real_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported on this filesystem or permission denied")

    selection = ImportSelection(primary_path=symlink_file)
    bundle = resolve_bundle(selection)

    policy = VerifyPolicy(allow_symlinks=False)
    with pytest.raises(ValueError, match="[Ss]ymlink"):
        verify_bundle(bundle, policy=policy)
