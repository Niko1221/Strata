"""strata_control/backend_contracts.py - Backend capabilities and routing contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

from strata_control.bundle_identity import VerifiedBundle


@dataclass(frozen=True)
class BackendCapabilities:
    schema_version: str
    backend_id: str
    binary_path: str
    binary_sha256: str
    supported_architectures: tuple[str, ...]
    supported_dialects: tuple[str, ...]
    supports_cuda: bool = True
    supports_cpu: bool = True
    supports_spin8: bool = False
    supports_steering: bool = False
    caps_hash: str = ""


@dataclass(frozen=True)
class LoadRequest:
    preferred_backend: str | None = None
    force_device: str | None = None  # cuda, cpu
    allow_fallback: bool = False


@dataclass(frozen=True)
class ResolvedBackend:
    backend_id: str
    capabilities: BackendCapabilities
    decision_reason: str
    launch_command: list[str] = field(default_factory=list)
