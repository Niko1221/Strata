from pathlib import Path
import pytest

from strata_control.backend_contracts import LoadRequest
from strata_control.backend_registry import BackendRegistry, create_default_registry
from strata_control.bundle_identity import verify_bundle
from strata_control.bundle_resolver import ImportSelection, resolve_bundle


def test_routing_qwen4exp_to_native(tmp_path: Path):
    """Verify that qwen4exp (Swift-1.5 / Qwen3.8-Flash-Next) routes to strata-native."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 32)
    bundle = resolve_bundle(ImportSelection(primary_path=f, architecture="qwen4exp"))
    verified = verify_bundle(bundle)

    reg = create_default_registry()
    routed = reg.select_backend(verified, LoadRequest())
    assert routed.backend_id == "strata-native"


def test_routing_deepseek41_to_dsv41(tmp_path: Path):
    """Verify that deepseek41 architecture routes to strata-dsv41."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 32)
    bundle = resolve_bundle(ImportSelection(primary_path=f, architecture="deepseek41"))
    verified = verify_bundle(bundle)

    reg = create_default_registry()
    routed = reg.select_backend(verified, LoadRequest())
    assert routed.backend_id == "strata-dsv41"


def test_manual_routing_unsupported_architecture_rejected(tmp_path: Path):
    """Verify that manual selection of a backend that doesn't support the architecture is rejected without fallback."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 32)
    bundle = resolve_bundle(ImportSelection(primary_path=f, architecture="deepseek41"))
    verified = verify_bundle(bundle)

    reg = create_default_registry()
    # Force manual routing of deepseek41 to strata-native (which only supports qwen)
    with pytest.raises(ValueError, match="Substitution forbidden"):
        reg.select_backend(verified, LoadRequest(preferred_backend="strata-native"))
