"""tests/runtime/test_session_identity.py - Tests for session cache identity and resume invalidation."""
from __future__ import annotations

import pytest

from strata_control.session_identity import SessionIdentity, validate_session_resume


def test_any_semantic_change_invalidates() -> None:
    base = SessionIdentity(
        model_set_id="bundle-1",
        backend_id="strata-dsv41",
        kv_cache_dialect="standard",
        spin8_mode="off",
        steering_hash="hash-0",
        geometry_hash="geo-0",
    )
    # Identical session resumes cleanly
    assert validate_session_resume(base, base) is True

    # Altering model_set_id invalidates
    drift_model = SessionIdentity(
        model_set_id="bundle-2",
        backend_id=base.backend_id,
        kv_cache_dialect=base.kv_cache_dialect,
        spin8_mode=base.spin8_mode,
        steering_hash=base.steering_hash,
        geometry_hash=base.geometry_hash,
    )
    with pytest.raises(ValueError, match="Session identity mismatch"):
        validate_session_resume(base, drift_model)

    # Altering spin8_mode invalidates
    drift_spin8 = SessionIdentity(
        model_set_id=base.model_set_id,
        backend_id=base.backend_id,
        kv_cache_dialect=base.kv_cache_dialect,
        spin8_mode="cuda",
        steering_hash=base.steering_hash,
        geometry_hash=base.geometry_hash,
    )
    with pytest.raises(ValueError, match="Session identity mismatch"):
        validate_session_resume(base, drift_spin8)

    # Altering steering_hash invalidates
    drift_steering = SessionIdentity(
        model_set_id=base.model_set_id,
        backend_id=base.backend_id,
        kv_cache_dialect=base.kv_cache_dialect,
        spin8_mode=base.spin8_mode,
        steering_hash="hash-diff",
        geometry_hash=base.geometry_hash,
    )
    with pytest.raises(ValueError, match="Session identity mismatch"):
        validate_session_resume(base, drift_steering)
