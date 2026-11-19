from pathlib import Path
import pytest

from strata_control.bundle_identity import verify_bundle
from strata_control.bundle_resolver import ImportSelection, resolve_bundle
from strata_control.model_memory import (
    HardwareSnapshot,
    MemoryPlan,
    ResourcePolicy,
    capture_live_hardware_snapshot,
    plan_memory,
)


def test_reject_when_available_ram_insufficient(tmp_path: Path):
    """Verify that available RAM (12.97 GB) < planned RAM (18.79 GB) is rejected for launch."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 32)
    bundle = resolve_bundle(ImportSelection(primary_path=f))
    verified = verify_bundle(bundle)

    # Snapshot with 32 GB total RAM, but only 12.97 GB available
    hw = HardwareSnapshot(
        total_physical_ram_bytes=32 * 1024**3,
        available_physical_ram_bytes=int(12.97 * 1024**3),
        commit_limit_bytes=36 * 1024**3,
        available_commit_bytes=14 * 1024**3,
        total_vram_bytes=16 * 1024**3,
        free_vram_bytes=12 * 1024**3,
    )

    # Planned host: 18.79 GB
    plan = plan_memory(
        verified,
        caps=None,
        hardware=hw,
        dense_bytes=int(10.0 * 1024**3),
        expert_resident_bytes=int(7.79 * 1024**3),
        scratch_bytes=int(1.0 * 1024**3),
    )

    assert plan.is_launchable is False
    assert "Insufficient available RAM" in str(plan.rejection_reason)


def test_reject_when_vram_insufficient(tmp_path: Path):
    """Verify that free VRAM (1.39 GB) < planned VRAM (8.80 GB) is rejected for launch."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 32)
    bundle = resolve_bundle(ImportSelection(primary_path=f))
    verified = verify_bundle(bundle)

    hw = HardwareSnapshot(
        total_physical_ram_bytes=32 * 1024**3,
        available_physical_ram_bytes=25 * 1024**3,
        commit_limit_bytes=36 * 1024**3,
        available_commit_bytes=28 * 1024**3,
        total_vram_bytes=16 * 1024**3,
        free_vram_bytes=int(1.39 * 1024**3),  # only 1.39 GB free
    )

    plan = plan_memory(
        verified,
        caps=None,
        hardware=hw,
        dense_bytes=4 * 1024**3,
        expert_resident_bytes=4 * 1024**3,
        scratch_bytes=1 * 1024**3,
        planned_vram_bytes=int(8.80 * 1024**3),
    )

    assert plan.is_launchable is False
    assert "Insufficient free VRAM" in str(plan.rejection_reason)


def test_static_qualification_vs_dynamic_launch(tmp_path: Path):
    """Verify that static qualification can pass for 32GB PC even when dynamic launch is blocked by current memory pressure."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 32)
    bundle = resolve_bundle(ImportSelection(primary_path=f))
    verified = verify_bundle(bundle)

    hw = HardwareSnapshot(
        total_physical_ram_bytes=32 * 1024**3,
        available_physical_ram_bytes=int(12.97 * 1024**3),  # under heavy load
        commit_limit_bytes=36 * 1024**3,
        available_commit_bytes=14 * 1024**3,
        total_vram_bytes=16 * 1024**3,
        free_vram_bytes=12 * 1024**3,
    )

    plan = plan_memory(
        verified,
        caps=None,
        hardware=hw,
        dense_bytes=int(10.0 * 1024**3),
        expert_resident_bytes=int(7.0 * 1024**3),
        scratch_bytes=int(1.0 * 1024**3),
    )

    # Static qualification passes because total hardware is 32GB RAM / 16GB VRAM
    assert plan.is_qualifiable is True
    # Dynamic launch fails because available RAM is currently only 12.97GB
    assert plan.is_launchable is False


def test_capture_live_hardware_snapshot():
    """Verify that live snapshot returns positive values for Windows host."""
    snap = capture_live_hardware_snapshot()
    assert snap.total_physical_ram_bytes > 0
    assert snap.available_physical_ram_bytes > 0
    assert snap.commit_limit_bytes >= snap.total_physical_ram_bytes
