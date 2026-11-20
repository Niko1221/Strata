from pathlib import Path
import pytest

from strata_control.backends.dsv41 import (
    REQUIRED_DEEPSEEK_DENSE_TENSORS,
    PrepareRequest,
    prepare_dsv41,
    validate_dsv41,
)
from strata_control.bundle_identity import verify_bundle
from strata_control.bundle_resolver import ImportSelection, resolve_bundle
from tests.fixtures.gguf_builder import MinimalGgufBuilder


def test_validate_dsv41_with_real_tensors(tmp_path: Path):
    """Verify validate_dsv41 succeeds when real GGUF contains required dense tensors."""
    builder = MinimalGgufBuilder(version=3)
    builder.add_string("general.architecture", "deepseek41")
    # Add real required dense tensors (using 8 elements * 4 bytes = 32 bytes)
    builder.add_tensor("token_embd.weight", [8, 1], 0, 0, b"\x00" * 32)
    builder.add_tensor("output_norm.weight", [8, 1], 0, 0, b"\x00" * 32)
    builder.add_tensor("output.weight", [8, 1], 0, 0, b"\x00" * 32)
    raw = builder.build()

    f = tmp_path / "model.gguf"
    f.write_bytes(raw)

    bundle = resolve_bundle(ImportSelection(primary_path=f, architecture="deepseek41"))
    verified = verify_bundle(bundle)

    report = validate_dsv41(verified)
    assert report.decision == "QUALIFIED"
    assert report.qualification_status == "PASS"
    assert not report.errors


def test_validate_dsv41_missing_tensor_rejected(tmp_path: Path):
    """Verify validate_dsv41 rejects GGUF when a required dense tensor is missing."""
    builder = MinimalGgufBuilder(version=3)
    builder.add_string("general.architecture", "deepseek41")
    # Only add embedding, missing norm and output
    builder.add_tensor("token_embd.weight", [8, 1], 0, 0, b"\x00" * 32)
    raw = builder.build()

    f = tmp_path / "model.gguf"
    f.write_bytes(raw)

    bundle = resolve_bundle(ImportSelection(primary_path=f, architecture="deepseek41"))
    verified = verify_bundle(bundle)

    report = validate_dsv41(verified)
    assert report.decision == "INCOMPATIBLE"
    assert report.qualification_status == "FAIL"
    assert any("Missing required DeepSeek dense tensors" in e for e in report.errors)


def test_simulated_tensors_parameter_removed():
    """Verify that simulated_tensors argument is removed and rejected by validate_dsv41."""
    import inspect
    sig = inspect.signature(validate_dsv41)
    assert "simulated_tensors" not in sig.parameters

    sig_prep = inspect.signature(prepare_dsv41)
    assert "simulated_tensors" not in sig_prep.parameters
