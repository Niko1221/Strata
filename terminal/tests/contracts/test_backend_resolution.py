"""tests/contracts/test_backend_resolution.py - Tests for backend capability contracts and routing."""
from __future__ import annotations

from pathlib import Path
import pytest

from strata_control.bundle_resolver import ImportSelection, resolve_bundle
from strata_control.bundle_identity import verify_bundle
from strata_control.backend_contracts import (
    BackendCapabilities,
    LoadRequest,
    ResolvedBackend,
)
from strata_control.backend_registry import BackendRegistry


@pytest.fixture
def populated_registry() -> BackendRegistry:
    reg = BackendRegistry()
    reg.register_backend(
        BackendCapabilities(
            schema_version="1.0",
            backend_id="strata-native",
            binary_path="/bin/strata-native",
            binary_sha256="native-sha",
            supported_architectures=("qwen2", "qwen"),
            supported_dialects=("standard", "tq4_1s", "tq3_0s"),
            supports_spin8=True,
        )
    )
    reg.register_backend(
        BackendCapabilities(
            schema_version="1.0",
            backend_id="strata-dsv41",
            binary_path="/bin/strata-dsv41",
            binary_sha256="dsv41-sha",
            supported_architectures=("dsv41", "deepseek", "deepseek2"),
            supported_dialects=("standard", "iq4_nl"),
        )
    )
    reg.register_backend(
        BackendCapabilities(
            schema_version="1.0",
            backend_id="prism",
            binary_path="/bin/prism",
            binary_sha256="prism-sha",
            supported_architectures=("qwen2", "llama"),
            supported_dialects=("pq2_0", "ptq1_0"),
        )
    )
    return reg


def test_no_name_allowlist(tmp_path: Path, populated_registry: BackendRegistry) -> None:
    """Routing is governed strictly by architecture and dialect, not filename or uncensored tag."""
    f = tmp_path / "custom_uncensored_supermodel.gguf"
    f.write_bytes(b"model-data")

    # A DeepSeek model with an arbitrary filename
    bundle = resolve_bundle(
        ImportSelection(primary_path=f, architecture="dsv41", dialect="standard")
    )
    verified = verify_bundle(bundle)

    resolved = populated_registry.select_backend(verified, LoadRequest())
    assert resolved.backend_id == "strata-dsv41"


def test_manual_backend_no_substitution(tmp_path: Path, populated_registry: BackendRegistry) -> None:
    """Manual selection of an incompatible backend must fail closed rather than silently substitute."""
    f = tmp_path / "deepseek.gguf"
    f.write_bytes(b"data")

    bundle = resolve_bundle(
        ImportSelection(primary_path=f, architecture="dsv41", dialect="standard")
    )
    verified = verify_bundle(bundle)

    # Forcing strata-native for DeepSeek must raise ValueError
    with pytest.raises(ValueError, match="does not support architecture 'dsv41'. Substitution forbidden."):
        populated_registry.select_backend(
            verified, LoadRequest(preferred_backend="strata-native")
        )


def test_no_cloud_fallback(tmp_path: Path, populated_registry: BackendRegistry) -> None:
    """An unsupported architecture raises an error; it must never fall back to cloud."""
    f = tmp_path / "alien.gguf"
    f.write_bytes(b"data")

    bundle = resolve_bundle(
        ImportSelection(primary_path=f, architecture="unsupported_arch", dialect="standard")
    )
    verified = verify_bundle(bundle)

    with pytest.raises(ValueError, match="Cloud fallback is forbidden"):
        populated_registry.select_backend(verified, LoadRequest())
