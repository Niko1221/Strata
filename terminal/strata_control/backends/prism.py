"""strata_control/backends/prism.py - Prism backend adapter for Bonsai PQ2_0/PTQ1_0 dialects."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from terminal.strata_control.backend_contracts import BackendCapabilities
from terminal.strata_control.bundle_identity import VerifiedBundle, compute_file_sha256


@dataclass(frozen=True)
class PrismLaunchPlan:
    backend_id: str = "prism"
    binary_path: str = "prism-server"
    args: tuple[str, ...] = field(default_factory=tuple)
    supports_pq2_0: bool = True
    verified_sha256: str = ""


def build_prism_plan(
    bundle: VerifiedBundle,
    caps: BackendCapabilities | None = None,
    port: int = 18503,
) -> PrismLaunchPlan:
    dialect = bundle.bundle.dialect.lower()
    if dialect not in ("pq2_0", "ptq1_0", "standard"):
        raise ValueError(f"Prism backend does not support dialect '{dialect}'")

    bin_path = caps.binary_path if caps else "prism-server"
    verified_sha = ""
    if caps and Path(caps.binary_path).exists():
        actual_sha = compute_file_sha256(Path(caps.binary_path))
        if caps.binary_sha256 and actual_sha != caps.binary_sha256:
            raise ValueError(
                f"Binary hash mismatch for {caps.binary_path}: expected {caps.binary_sha256}, actual {actual_sha}"
            )
        verified_sha = actual_sha

    args = (
        "--model", str(bundle.verified_files[0].path),
        "--dialect", dialect,
        "--port", str(port),
        "--host", "127.0.0.1",
    )
    return PrismLaunchPlan(
        backend_id="prism",
        binary_path=bin_path,
        args=args,
        verified_sha256=verified_sha,
    )
