"""tests/mutations/test_critical_mutants.py - Mutation tests for critical security, budgeting, and routing invariants."""
from __future__ import annotations

from pathlib import Path
import pytest

from strata_control.gguf_preflight import check_tensor_dimensions
from strata_control.backend_registry import BackendRegistry
from strata_control.backend_contracts import BackendCapabilities, LoadRequest
from strata_control.bundle_resolver import ImportSelection, resolve_bundle
from strata_control.bundle_identity import verify_bundle


def test_mutant_tensor_overflow_inverted_guard() -> None:
    """Mutant: If dimension overflow check is inverted or bypassed, it fails."""
    # Must reject safely
    with pytest.raises(ValueError):
        check_tensor_dimensions([2**35, 2**35], type_size=4)


def test_mutant_silent_cloud_fallback_blocked(tmp_path: Path) -> None:
    """Mutant: If unsupported architecture silently proxies to cloud, it fails."""
    f = tmp_path / "alien.gguf"
    f.write_bytes(b"data")
    bundle = verify_bundle(resolve_bundle(ImportSelection(primary_path=f, architecture="unknown_alien")))

    reg = BackendRegistry()
    # No backend supports alien architecture
    with pytest.raises(ValueError, match="Cloud fallback is forbidden"):
        reg.select_backend(bundle, LoadRequest())


def test_mutant_split_ordinal_tamper(tmp_path: Path) -> None:
    """Mutant: If shard ordinal sequence is corrupted, bundle resolution detects gap."""
    s1 = tmp_path / "model-00001-of-00002.gguf"
    s1.write_bytes(b"shard-1")
    # Missing shard-2
    with pytest.raises(FileNotFoundError, match="Missing required shard"):
        resolve_bundle(ImportSelection(primary_path=s1))
