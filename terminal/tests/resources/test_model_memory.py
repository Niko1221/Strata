"""tests/resources/test_model_memory.py - Tests for RAM32 hardware budgeting and preflight launch guards."""
from __future__ import annotations

from pathlib import Path
import pytest

from strata_control.bundle_resolver import ImportSelection, resolve_bundle
from strata_control.bundle_identity import verify_bundle
from strata_control.model_memory import (
    HardwareSnapshot,
    ResourcePolicy,
    plan_memory,
)


def test_host_vram_not_unified(tmp_path: Path) -> None:
    """Verifies that RAM and VRAM are treated as separate disjoint pools."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"data")
    bundle = verify_bundle(resolve_bundle(ImportSelection(primary_path=f)))

    # Machine has 8GB RAM available and 14GB free VRAM
    # Host needs 11GB RAM -> Must FAIL even though total memory (8 + 14 = 22GB) > 11GB
    hw = HardwareSnapshot(
        total_physical_ram_bytes=32 * 1024**3,
        available_physical_ram_bytes=8 * 1024**3,  # Only 8GB RAM free
        commit_limit_bytes=64 * 1024**3,
        available_commit_bytes=10 * 1024**3,
        total_vram_bytes=16 * 1024**3,
        free_vram_bytes=14 * 1024**3,
    )
    plan = plan_memory(bundle, None, hw)
    assert plan.is_launchable is False
    assert "Insufficient available RAM" in (plan.rejection_reason or "")


def test_dense_floor_rejection(tmp_path: Path) -> None:
    """Dense floor exceeding budget cap is rejected before spawn."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"data")
    bundle = verify_bundle(resolve_bundle(ImportSelection(primary_path=f)))

    hw = HardwareSnapshot(
        total_physical_ram_bytes=32 * 1024**3,
        available_physical_ram_bytes=28 * 1024**3,
        commit_limit_bytes=64 * 1024**3,
        available_commit_bytes=30 * 1024**3,
        total_vram_bytes=16 * 1024**3,
        free_vram_bytes=14 * 1024**3,
    )
    # Huge dense floor of 26 GB exceeds 24 GB budget cap
    plan = plan_memory(bundle, None, hw, dense_bytes=26 * 1024**3)
    assert plan.is_launchable is False
    assert "exceeds budget cap" in (plan.rejection_reason or "")


def test_valid_plan_with_adequate_headroom(tmp_path: Path) -> None:
    """A realistic allocation within 24GB cap and with 4GB OS headroom passes."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"data")
    bundle = verify_bundle(resolve_bundle(ImportSelection(primary_path=f)))

    # Real machine snapshot: ~16GB free RAM, 14GB free VRAM
    hw = HardwareSnapshot(
        total_physical_ram_bytes=32 * 1024**3,
        available_physical_ram_bytes=16 * 1024**3,
        commit_limit_bytes=64 * 1024**3,
        available_commit_bytes=20 * 1024**3,
        total_vram_bytes=16 * 1024**3,
        free_vram_bytes=14 * 1024**3,
    )
    # Plan needs: dense 6GB + expert 4GB + scratch 1GB = 11GB host RAM
    plan = plan_memory(bundle, None, hw)
    assert plan.is_launchable is True
    assert plan.planned_host_bytes == 11 * 1024**3
    assert plan.headroom_bytes == 5 * 1024**3
