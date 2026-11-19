"""tests/backends/test_dsv41_adapter.py - Tests for strata-dsv41 adapter contracts."""
from __future__ import annotations

from pathlib import Path
import pytest

from strata_control.bundle_resolver import ImportSelection, resolve_bundle
from strata_control.bundle_identity import verify_bundle
from strata_control.backends.dsv41 import (
    validate_dsv41,
    prepare_dsv41,
    PrepareRequest,
)
from tests.fixtures.gguf_builder import MinimalGgufBuilder


def _make_dsv41_gguf(path: Path, arch: str = "deepseek41", include_all_dense: bool = True) -> None:
    builder = MinimalGgufBuilder(version=3)
    builder.add_string("general.architecture", arch)
    if include_all_dense:
        builder.add_tensor("token_embd.weight", [8, 1], 0, 0, b"\x00" * 32)
        builder.add_tensor("output_norm.weight", [8, 1], 0, 0, b"\x00" * 32)
        builder.add_tensor("output.weight", [8, 1], 0, 0, b"\x00" * 32)
    else:
        # Missing output_norm
        builder.add_tensor("token_embd.weight", [8, 1], 0, 0, b"\x00" * 32)
    path.write_bytes(builder.build())


def test_renamed_dsv41_variant(tmp_path: Path) -> None:
    """A DeepSeek variant with any filename passes if required tensors and geometry match."""
    f = tmp_path / "custom_uncensored_maya_s.gguf"
    _make_dsv41_gguf(f, arch="deepseek41", include_all_dense=True)

    bundle = resolve_bundle(
        ImportSelection(primary_path=f, architecture="dsv41", dialect="standard")
    )
    verified = verify_bundle(bundle)

    # Valid with all required dense tensors from real inspection
    report = validate_dsv41(verified, caps=None)
    assert report.decision == "QUALIFIED"
    assert len(report.errors) == 0


def test_dense_required_tensor_missing(tmp_path: Path) -> None:
    """Missing embed_tokens or norm weights is reported as incompatible."""
    f = tmp_path / "incomplete.gguf"
    _make_dsv41_gguf(f, arch="deepseek41", include_all_dense=False)

    bundle = resolve_bundle(
        ImportSelection(primary_path=f, architecture="dsv41", dialect="standard")
    )
    verified = verify_bundle(bundle)

    report = validate_dsv41(verified, caps=None)
    assert report.decision == "INCOMPATIBLE"
    assert any("output_norm.weight" in e or "dense tensors" in e for e in report.errors)


def test_unsupported_arch_rejected(tmp_path: Path) -> None:
    """A llama or qwen model is rejected by the dsv41 adapter."""
    f = tmp_path / "qwen.gguf"
    _make_dsv41_gguf(f, arch="qwen2", include_all_dense=True)

    bundle = resolve_bundle(
        ImportSelection(primary_path=f, architecture="qwen2", dialect="standard")
    )
    verified = verify_bundle(bundle)

    report = validate_dsv41(verified, caps=None)
    assert report.decision == "INCOMPATIBLE"
    assert any("Architecture 'qwen2' is not supported" in e for e in report.errors)


def test_prepare_creates_clean_staging(tmp_path: Path) -> None:
    """Preparation writes isolated staging manifest without mutating original files."""
    f = tmp_path / "ds.gguf"
    _make_dsv41_gguf(f, arch="deepseek41", include_all_dense=True)
    orig_bytes = f.read_bytes()

    bundle = resolve_bundle(
        ImportSelection(primary_path=f, architecture="dsv41", dialect="standard")
    )
    verified = verify_bundle(bundle)

    out_dir = tmp_path / "staging_ds4"
    prepared = prepare_dsv41(
        verified,
        PrepareRequest(output_dir=out_dir),
    )
    assert Path(prepared.pack_manifest_path).exists()
    assert f.read_bytes() == orig_bytes
