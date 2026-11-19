"""tests/models/test_registry_transaction.py - Tests for atomic manifest registration and crash recovery."""
from __future__ import annotations

from pathlib import Path
import pytest

from strata_control.bundle_resolver import ImportSelection, resolve_bundle
from strata_control.bundle_identity import verify_bundle
from strata_control.model_registry import ModelRegistry


def test_atomic_registry_transaction(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    registry = ModelRegistry(registry_dir)

    f = tmp_path / "model.gguf"
    f.write_bytes(b"model-payload-data")

    bundle = resolve_bundle(ImportSelection(primary_path=f))
    verified = verify_bundle(bundle)

    receipt = registry.register(verified)
    assert receipt.registered is True
    assert receipt.model_set_id == bundle.model_set_id

    # Retrieve from registry
    record = registry.get(bundle.model_set_id)
    assert record is not None
    assert record["canonical_hash"] == verified.canonical_hash
    assert len(record["files"]) == 1
    assert record["files"][0]["path"] == str(f)
