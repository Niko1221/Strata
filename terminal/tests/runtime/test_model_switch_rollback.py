"""tests/runtime/test_model_switch_rollback.py - Tests for automatic rollback to original working model upon activation failure."""
from __future__ import annotations

import pytest


class RuntimeSessionManager:
    def __init__(self, initial_model: str) -> None:
        self.active_model = initial_model
        self.last_known_good = initial_model

    def switch_model(self, new_model: str, should_fail: bool = False) -> str:
        saved_good = self.last_known_good
        self.active_model = new_model
        if should_fail:
            # Immediate rollback to last known good
            self.active_model = saved_good
            raise RuntimeError(f"Activation of {new_model} failed. Restored {saved_good}.")
        self.last_known_good = new_model
        return self.active_model


def test_failed_activation_restores_original() -> None:
    mgr = RuntimeSessionManager(initial_model="Qwen3.8-Flash-Next-Q2_0")
    assert mgr.active_model == "Qwen3.8-Flash-Next-Q2_0"

    # Attempting to load faulty model
    with pytest.raises(RuntimeError, match="failed. Restored Qwen3.8-Flash-Next-Q2_0"):
        mgr.switch_model("Corrupted-DeepSeek", should_fail=True)

    # Must be back on the original model
    assert mgr.active_model == "Qwen3.8-Flash-Next-Q2_0"


def test_no_model_silent_switch() -> None:
    """The runtime must never silently fall back without raising an error."""
    mgr = RuntimeSessionManager(initial_model="Qwen3.8-Flash-Next-Q2_0")
    with pytest.raises(RuntimeError):
        mgr.switch_model("Missing-Weights-Model", should_fail=True)
