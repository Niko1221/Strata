from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest

from strata_control.bundle_identity import verify_bundle
from strata_control.bundle_resolver import ImportSelection, resolve_bundle
from strata_control.model_registry import ModelRegistry


def test_registry_register_and_get(tmp_path: Path):
    """Verify basic register and retrieval from registry."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 32)

    selection = ImportSelection(primary_path=f)
    bundle = resolve_bundle(selection)
    verified = verify_bundle(bundle)

    reg_dir = tmp_path / "registry"
    reg = ModelRegistry(reg_dir)
    receipt = reg.register(verified)

    assert receipt.registered is True
    assert receipt.model_set_id == verified.model_set_id
    assert receipt.canonical_hash == verified.canonical_hash

    entry = reg.get(verified.model_set_id)
    assert entry is not None
    assert entry["canonical_hash"] == verified.canonical_hash
    assert len(entry["files"]) == 1


def test_registry_conflict_on_hash_mismatch(tmp_path: Path):
    """Verify that registering different content under the same model_set_id raises a conflict error."""
    f1 = tmp_path / "model1.gguf"
    f1.write_bytes(b"GGUF" + b"\x00" * 32)
    bundle1 = resolve_bundle(ImportSelection(primary_path=f1))
    verified1 = verify_bundle(bundle1)

    reg_dir = tmp_path / "registry"
    reg = ModelRegistry(reg_dir)
    reg.register(verified1)

    # Create a second verified bundle with the same model_set_id but different canonical_hash
    tampered_verified = type(verified1)(
        model_set_id=verified1.model_set_id,
        canonical_hash="f" * 64,
        bundle=verified1.bundle,
        verified_files=verified1.verified_files,
        policy=verified1.policy,
    )

    with pytest.raises(ValueError, match="[Cc]onflict|[Mm]ismatch"):
        reg.register(tampered_verified)


def test_concurrent_registry_writes(tmp_path: Path):
    """Verify thread-safe registration without race condition or corruption."""
    reg_dir = tmp_path / "registry"
    reg = ModelRegistry(reg_dir)

    def register_item(i: int):
        f = tmp_path / f"model_{i}.gguf"
        f.write_bytes(b"GGUF" + bytes([i % 256]) * 64)
        bundle = resolve_bundle(ImportSelection(primary_path=f))
        verified = verify_bundle(bundle)
        return reg.register(verified)

    with ThreadPoolExecutor(max_workers=4) as ex:
        receipts = list(ex.map(register_item, range(10)))

    assert len(receipts) == 10
    assert all(r.registered for r in receipts)

    # Verify that all 10 distinct entries are present in registry
    manifest = reg._read()
    assert len(manifest) == 10
