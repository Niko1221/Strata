"""tests/runtime/test_custom_model_activation.py - Tests for model activation and preflight drift abort."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pytest


@dataclass(frozen=True)
class LaunchPlan:
    model_set_id: str
    backend_id: str
    binary_path: str
    expected_source_sha256: str
    port: int = 18501


@dataclass(frozen=True)
class ActivationReceipt:
    activated: bool
    model_set_id: str
    backend_id: str
    nonce: str


def activate_model(plan: LaunchPlan, actual_source_sha256: str) -> ActivationReceipt:
    """Preflight check: if source sha256 changed after validation, immediately abort spawn."""
    if actual_source_sha256 != plan.expected_source_sha256:
        raise ValueError(
            f"Source drifted before spawn! Expected {plan.expected_source_sha256}, actual {actual_source_sha256}. Spawn aborted."
        )
    return ActivationReceipt(
        activated=True,
        model_set_id=plan.model_set_id,
        backend_id=plan.backend_id,
        nonce="nonce-abc-123",
    )


def test_preview_replaced_before_spawn() -> None:
    plan = LaunchPlan(
        model_set_id="bundle-1",
        backend_id="strata-dsv41",
        binary_path="/bin/strata-dsv41",
        expected_source_sha256="sha-original",
    )
    # File changed to 'sha-drifted' behind the back
    with pytest.raises(ValueError, match="Source drifted before spawn"):
        activate_model(plan, actual_source_sha256="sha-drifted")


def test_successful_activation() -> None:
    plan = LaunchPlan(
        model_set_id="bundle-1",
        backend_id="strata-dsv41",
        binary_path="/bin/strata-dsv41",
        expected_source_sha256="sha-matching",
    )
    receipt = activate_model(plan, actual_source_sha256="sha-matching")
    assert receipt.activated is True
    assert receipt.model_set_id == "bundle-1"
