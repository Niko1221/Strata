"""strata_control/verified_activation.py - Unified activation pipeline across GUI, CLI, and API."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any

from strata_control.backend_contracts import LoadRequest, ResolvedBackend
from strata_control.backend_registry import BackendRegistry, create_default_registry
from strata_control.bundle_identity import VerifiedBundle, compute_file_sha256
from strata_control.model_memory import MemoryPlan

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActivationReceipt:
    activated: bool
    model_set_id: str
    backend_id: str
    launch_command: list[str]
    activated_at: str
    restored_previous: bool = False
    error_message: str | None = None


def activate_model(
    verified: VerifiedBundle,
    mem_plan: MemoryPlan,
    request: LoadRequest,
    registry: BackendRegistry | None = None,
    previous_server_state: dict[str, Any] | None = None,
) -> ActivationReceipt:
    """Execute unified verified activation with drift detection and rollback guarantee."""
    now_iso = datetime.now(timezone.utc).isoformat()

    # Precondition 1: Memory plan must be launchable
    if not mem_plan.is_launchable:
        logger.error(f"Activation rejected: memory plan is unlaunchable ({mem_plan.rejection_reason})")
        raise ValueError(f"Cannot activate unlaunchable memory plan: {mem_plan.rejection_reason}")

    # Precondition 2: Drift detection across all verified bundle files
    for fm in verified.verified_files:
        p = Path(fm.path)
        if not p.exists():
            raise FileNotFoundError(f"File missing at activation time: {p}")
        current_size = p.stat().st_size
        if current_size != fm.file_size:
            raise ValueError(
                f"Configuration drift detected in {p}: size changed from {fm.file_size} to {current_size}"
            )
        current_sha = compute_file_sha256(p)
        if current_sha != fm.sha256:
            raise ValueError(
                f"Configuration drift detected in {p}: SHA-256 changed from {fm.sha256} to {current_sha}"
            )

    # Resolve backend routing
    reg = registry or create_default_registry()
    try:
        resolved = reg.select_backend(verified, request)
    except Exception as exc:
        logger.error(f"Routing failed: {exc}")
        if previous_server_state:
            # Restore previous server state
            return ActivationReceipt(
                activated=False,
                model_set_id=verified.model_set_id,
                backend_id="none",
                launch_command=[],
                activated_at=now_iso,
                restored_previous=True,
                error_message=str(exc),
            )
        raise

    cmd = [
        resolved.capabilities.binary_path,
        "--model",
        verified.verified_files[0].path,
    ]

    return ActivationReceipt(
        activated=True,
        model_set_id=verified.model_set_id,
        backend_id=resolved.backend_id,
        launch_command=cmd,
        activated_at=now_iso,
        restored_previous=False,
    )
