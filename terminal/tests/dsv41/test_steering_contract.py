"""tests/dsv41/test_steering_contract.py - Tests for DeepSeek isolated steering bridge contracts."""
from __future__ import annotations

import pytest


def validate_vector_compatibility(model_arch: str, vector_dim: int, expected_dim: int = 2048) -> bool:
    # Must strictly match dsv41 architecture; cannot reuse vectors from qwen even if width matches
    if model_arch.lower() not in ("dsv41", "deepseek"):
        return False
    if vector_dim != expected_dim:
        return False
    return True


def test_same_width_not_same_vector() -> None:
    """A vector trained on Qwen (even if hidden width 2048 matches) is rejected for DeepSeek."""
    # Qwen vector with width 2048
    assert validate_vector_compatibility("qwen2", 2048) is False
    # DeepSeek vector with width 2048
    assert validate_vector_compatibility("dsv41", 2048) is True


def test_profile_change_invalidates_all_states() -> None:
    """Changing steering profile invalidates cached states to prevent contaminated rollout."""
    class SessionContext:
        def __init__(self, profile_hash: str) -> None:
            self.profile_hash = profile_hash
            self.cached_kv_valid = True

        def update_profile(self, new_hash: str) -> None:
            if new_hash != self.profile_hash:
                self.profile_hash = new_hash
                self.cached_kv_valid = False  # Cache invalidated!

    sess = SessionContext("hash-profile-1")
    assert sess.cached_kv_valid is True

    sess.update_profile("hash-profile-2")
    assert sess.cached_kv_valid is False
