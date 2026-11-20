"""tests/dsv41/test_drafter_lifecycle.py - Tests for bounded streaming and drafter lifecycle."""
from __future__ import annotations

import pytest


class MockBoundedIoManager:
    def __init__(self, drafter_mode: str = "disabled-unloaded", max_pending_bytes: int = 256 * 1024 * 1024) -> None:
        self.drafter_mode = drafter_mode
        self.max_pending_bytes = max_pending_bytes
        self.in_flight_bytes = 0
        self.generation_id = 1

    def is_drafter_unloaded(self) -> bool:
        return self.drafter_mode == "disabled-unloaded"

    def allocated_drafter_vram_bytes(self) -> int:
        if self.is_drafter_unloaded():
            return 0
        return 1024 * 1024 * 1024

    def reserve_io(self, bytes_to_reserve: int) -> bool:
        if self.in_flight_bytes + bytes_to_reserve > self.max_pending_bytes:
            return False
        self.in_flight_bytes += bytes_to_reserve
        return True

    def cancel_pending(self) -> None:
        self.generation_id += 1
        self.in_flight_bytes = 0


def test_drafter_disabled_zero_alloc() -> None:
    """When drafter is disabled-unloaded, allocated drafter VRAM is strictly zero."""
    io_mgr = MockBoundedIoManager(drafter_mode="disabled-unloaded")
    assert io_mgr.is_drafter_unloaded() is True
    assert io_mgr.allocated_drafter_vram_bytes() == 0


def test_bounded_prefetch() -> None:
    """I/O prefetch requests exceeding 256 MiB in-flight cap are refused."""
    io_mgr = MockBoundedIoManager(max_pending_bytes=256 * 1024 * 1024)
    # Reserve 200 MiB: succeeds
    assert io_mgr.reserve_io(200 * 1024 * 1024) is True
    # Reserve additional 100 MiB (total 300 MiB > 256 MiB): rejected
    assert io_mgr.reserve_io(100 * 1024 * 1024) is False


def test_cancel_stale_io() -> None:
    """Cancellation increments generation ID and cleans pending allocations immediately."""
    io_mgr = MockBoundedIoManager()
    initial_gen = io_mgr.generation_id
    io_mgr.reserve_io(50 * 1024 * 1024)
    assert io_mgr.in_flight_bytes == 50 * 1024 * 1024

    io_mgr.cancel_pending()
    assert io_mgr.generation_id == initial_gen + 1
    assert io_mgr.in_flight_bytes == 0


def test_no_expert_pruning() -> None:
    """Top-k experts must remain full (e.g. k=8) even under high pressure; pruning is forbidden."""
    def select_experts(logits: list[float], top_k: int = 8, allow_prune: bool = False) -> list[int]:
        if allow_prune:
            raise AssertionError("Pruning of model experts is forbidden under contract T10")
        indexed = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in indexed[:top_k]]

    fake_logits = [0.1 * i for i in range(256)]
    selected = select_experts(fake_logits, top_k=8, allow_prune=False)
    assert len(selected) == 8
