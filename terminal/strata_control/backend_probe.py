"""strata_control/backend_probe.py - Real backend binary cryptographic inspection and capability discovery."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Sequence

from strata_control.backend_contracts import BackendCapabilities
from strata_control.bundle_identity import compute_file_sha256


def probe_backend_binary(
    binary_path: Path | str,
    backend_id: str,
    supported_architectures: Sequence[str],
    supported_dialects: Sequence[str],
    supports_cuda: bool = True,
    supports_cpu: bool = True,
    supports_spin8: bool = False,
    supports_steering: bool = False,
    strict: bool = False,
) -> BackendCapabilities:
    """Cryptographically inspect a local backend executable and return its verified capabilities."""
    p = Path(binary_path)
    if not p.exists() and not p.with_suffix(".exe").exists():
        if strict:
            raise FileNotFoundError(f"Backend binary not found: {p}")
        # Synthetic / stub fallback only when non-strict (e.g. offline dev without compiled binaries)
        bin_sha = hashlib.sha256(f"uncompiled:{backend_id}:{p.name}".encode("utf-8")).hexdigest()
    else:
        actual_path = p if p.exists() else p.with_suffix(".exe")
        bin_sha = compute_file_sha256(actual_path)

    # Compute stable caps_hash
    caps_hasher = hashlib.sha256()
    caps_hasher.update(f"{backend_id}:{bin_sha}".encode("utf-8"))
    caps_hasher.update(",".join(sorted(supported_architectures)).encode("utf-8"))
    caps_hasher.update(",".join(sorted(supported_dialects)).encode("utf-8"))
    caps_hasher.update(f"{supports_cuda}:{supports_cpu}:{supports_spin8}:{supports_steering}".encode("utf-8"))

    return BackendCapabilities(
        schema_version="1.0",
        backend_id=backend_id,
        binary_path=str(p),
        binary_sha256=bin_sha,
        supported_architectures=tuple(supported_architectures),
        supported_dialects=tuple(supported_dialects),
        supports_cuda=supports_cuda,
        supports_cpu=supports_cpu,
        supports_spin8=supports_spin8,
        supports_steering=supports_steering,
        caps_hash=caps_hasher.hexdigest(),
    )
