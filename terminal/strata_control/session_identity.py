"""strata_control/session_identity.py - Comprehensive cryptographic session identity and cache validation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any


@dataclass(frozen=True)
class SessionIdentity:
    model_set_id: str
    backend_id: str
    kv_cache_dialect: str
    spin8_mode: str
    steering_hash: str
    geometry_hash: str

    def canonical_digest(self) -> str:
        s = f"{self.model_set_id}:{self.backend_id}:{self.kv_cache_dialect}:{self.spin8_mode}:{self.steering_hash}:{self.geometry_hash}"
        return hashlib.sha256(s.encode("utf-8")).hexdigest()


def validate_session_resume(
    saved_identity: SessionIdentity,
    current_identity: SessionIdentity,
) -> bool:
    """Validate that every semantic component of the session matches; reject cache resume on any drift."""
    if saved_identity.canonical_digest() != current_identity.canonical_digest():
        raise ValueError(
            "Session identity mismatch: semantic parameters (model, dialect, spin8, or steering) changed. Cache invalidated."
        )
    return True
