import threading
import pytest


class BoundedIoManagerPy:
    """Python reference oracle matching strata::dsv41::BoundedIoManager contract."""

    def __init__(self, max_pending_bytes: int = 256 * 1024 * 1024, drafter_unloaded: bool = True):
        self.max_pending_bytes = max_pending_bytes
        self.is_unloaded = drafter_unloaded
        self._in_flight = 0
        self._generation = 1
        self._lock = threading.Lock()

    def is_drafter_unloaded(self) -> bool:
        return self.is_unloaded

    def allocated_drafter_vram_bytes(self) -> int:
        return 0 if self.is_unloaded else 1024 * 1024 * 1024

    def load_drafter(self):
        self.is_unloaded = False

    def unload_drafter(self):
        self.is_unloaded = True

    def reserve_io(self, num_bytes: int) -> bool:
        with self._lock:
            if self._in_flight + num_bytes > self.max_pending_bytes:
                return False
            self._in_flight += num_bytes
            return True

    def release_io(self, num_bytes: int):
        with self._lock:
            self._in_flight = max(0, self._in_flight - num_bytes)

    def cancel_pending(self, expected_gen: int):
        with self._lock:
            if self._generation == expected_gen:
                self._generation += 1
                self._in_flight = 0

    @property
    def active_generation_id(self) -> int:
        with self._lock:
            return self._generation

    @property
    def in_flight_bytes(self) -> int:
        with self._lock:
            return self._in_flight


def test_bounded_io_contract():
    mgr = BoundedIoManagerPy(max_pending_bytes=1000)
    assert mgr.is_drafter_unloaded() is True
    assert mgr.allocated_drafter_vram_bytes() == 0

    mgr.load_drafter()
    assert mgr.is_drafter_unloaded() is False
    assert mgr.allocated_drafter_vram_bytes() == 1024 * 1024 * 1024

    mgr.unload_drafter()
    assert mgr.is_drafter_unloaded() is True
    assert mgr.allocated_drafter_vram_bytes() == 0

    assert mgr.reserve_io(600) is True
    assert mgr.in_flight_bytes == 600

    assert mgr.reserve_io(500) is False  # Exceeds 1000
    assert mgr.in_flight_bytes == 600

    mgr.release_io(700)
    assert mgr.in_flight_bytes == 0

    assert mgr.reserve_io(500) is True
    mgr.cancel_pending(mgr.active_generation_id)
    assert mgr.in_flight_bytes == 0
    assert mgr.active_generation_id == 2
