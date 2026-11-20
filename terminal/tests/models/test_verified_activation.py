from pathlib import Path
import pytest

from strata_control.backend_contracts import LoadRequest
from strata_control.backend_registry import create_default_registry
from strata_control.bundle_identity import verify_bundle
from strata_control.bundle_resolver import ImportSelection, resolve_bundle
from strata_control.model_memory import (
    HardwareSnapshot,
    capture_live_hardware_snapshot,
    plan_memory,
)
from strata_control.verified_activation import (
    ActivationReceipt,
    activate_model,
)


def test_verified_activation_success(tmp_path: Path):
    """Verify clean verified activation of a valid bundle."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 32)
    bundle = resolve_bundle(ImportSelection(primary_path=f, architecture="qwen4exp"))
    verified = verify_bundle(bundle)

    # Ample memory snapshot
    hw = HardwareSnapshot(
        total_physical_ram_bytes=32 * 1024**3,
        available_physical_ram_bytes=24 * 1024**3,
        commit_limit_bytes=36 * 1024**3,
        available_commit_bytes=28 * 1024**3,
        total_vram_bytes=16 * 1024**3,
        free_vram_bytes=12 * 1024**3,
    )
    mem_plan = plan_memory(verified, caps=None, hardware=hw)

    receipt = activate_model(
        verified=verified,
        mem_plan=mem_plan,
        request=LoadRequest(),
        previous_server_state={"status": "stopped"},
    )

    assert receipt.activated is True
    assert receipt.model_set_id == verified.model_set_id
    assert receipt.backend_id == "strata-native"
    assert receipt.restored_previous is False


def test_verified_activation_rejects_unlaunchable_memory(tmp_path: Path):
    """Verify activation fails immediately if memory plan is unlaunchable."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 32)
    bundle = resolve_bundle(ImportSelection(primary_path=f, architecture="qwen4exp"))
    verified = verify_bundle(bundle)

    # Insufficient memory snapshot
    hw = HardwareSnapshot(
        total_physical_ram_bytes=32 * 1024**3,
        available_physical_ram_bytes=2 * 1024**3,  # only 2 GB available
        commit_limit_bytes=36 * 1024**3,
        available_commit_bytes=2 * 1024**3,
        total_vram_bytes=16 * 1024**3,
        free_vram_bytes=12 * 1024**3,
    )
    mem_plan = plan_memory(verified, caps=None, hardware=hw)

    with pytest.raises(ValueError, match="Cannot activate unlaunchable memory plan"):
        activate_model(
            verified=verified,
            mem_plan=mem_plan,
            request=LoadRequest(),
        )


def test_drift_detection_rejects_modified_files(tmp_path: Path):
    """Verify that file modification after verification is detected as drift and rejected."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 32)
    bundle = resolve_bundle(ImportSelection(primary_path=f, architecture="qwen4exp"))
    verified = verify_bundle(bundle)

    hw = HardwareSnapshot(
        total_physical_ram_bytes=32 * 1024**3,
        available_physical_ram_bytes=24 * 1024**3,
        commit_limit_bytes=36 * 1024**3,
        available_commit_bytes=28 * 1024**3,
        total_vram_bytes=16 * 1024**3,
        free_vram_bytes=12 * 1024**3,
    )
    mem_plan = plan_memory(verified, caps=None, hardware=hw)

    # Tamper with the file after verification
    f.write_bytes(b"TAMPERED_GGUF" + b"\x00" * 32)

    with pytest.raises(ValueError, match="[Dd]rift|[Mm]ismatch"):
        activate_model(
            verified=verified,
            mem_plan=mem_plan,
            request=LoadRequest(),
        )
