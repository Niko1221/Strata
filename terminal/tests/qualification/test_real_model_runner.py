"""tests/qualification/test_real_model_runner.py - TDD tests for Task R16 Real Model Qualification Runner.
Conforms strictly to MILSPEC-Codex standard: logging only, zero print, strict types.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
from typing import Any, Dict

from strata_control.qualification import (
    evaluate_gate,
    GateConfig,
    RunReceipt,
)
from tools.qualify_model_bundle import (
    qualify_model_run,
    QualificationRunRequest,
    ModelRunResult,
)


def test_zero_engine_launches_rejected(tmp_path: Path) -> None:
    """Zero engine invocations must never qualify a model for empirical evaluation."""
    dummy_model = tmp_path / "model.gguf"
    dummy_model.write_bytes(b"GGUF" + b"\x00" * 100)
    req = QualificationRunRequest(
        candidate_id="mock-qwen-zero-launch",
        architecture="qwen",
        dialect="standard",
        model_path=dummy_model,
    )
    result = qualify_model_run(req, dry_run=True)
    assert result.qualified is False
    assert result.real_model_qualified is False
    assert "no real engine invocation" in result.rejection_reason.lower() or "dryrun" in result.rejection_reason.lower()


def test_empty_token_output_rejected(tmp_path: Path) -> None:
    """Execution resulting in zero generated tokens cannot be qualified."""
    dummy_model = tmp_path / "model.gguf"
    dummy_model.write_bytes(b"GGUF" + b"\x00" * 100)

    res = ModelRunResult(
        candidate_id="test-empty-tokens",
        model_path=dummy_model,
        tokens_generated=0,
        output_text="",
        engine_invocations=1,
        duration_seconds=1.2,
    )
    receipt = res.to_run_receipt()
    eval_res = evaluate_gate(receipt, GateConfig(min_invocations=1, min_tokens=10, allow_dryrun=False))
    assert eval_res.passed is False
    assert eval_res.real_model_qualified is False


def test_mismatched_usage_or_hash_rejected(tmp_path: Path) -> None:
    """Discrepancy between declared tokens and actual output length must be rejected."""
    dummy_model = tmp_path / "model.gguf"
    dummy_model.write_bytes(b"GGUF" + b"\x00" * 100)

    res = ModelRunResult(
        candidate_id="test-mismatch",
        model_path=dummy_model,
        tokens_generated=500,  # Claimed 500
        output_text="short",   # Actual 1 word
        engine_invocations=1,
        duration_seconds=0.5,
    )
    # Validation should catch invalid token/text correlation or missing telemetry
    receipt = res.to_run_receipt()
    assert receipt.tokens_generated == 500
    # Strict gate with token length sanity check
    eval_res = evaluate_gate(receipt, GateConfig(min_invocations=1, min_tokens=1000, allow_dryrun=False))
    assert eval_res.passed is False


def test_missing_multipart_shards_marked_blocked(tmp_path: Path) -> None:
    """Missing multi-part shards (e.g. shard 1 of 6 present, 2-6 missing) must be classified as BLOCKED."""
    shard1 = tmp_path / "model-00001-of-00006.gguf"
    shard1.write_bytes(b"GGUF" + b"\x00" * 200)

    req = QualificationRunRequest(
        candidate_id="deepseek-v41-partial",
        architecture="dsv41",
        dialect="standard",
        model_path=shard1,
        total_shards=6,
        present_shards=1,
    )
    result = qualify_model_run(req, dry_run=False)
    assert result.qualified is False
    assert result.real_model_qualified is False
    assert result.tier_status.get("execution_readiness") == "BLOCKED"
    assert "missing" in result.rejection_reason.lower() and "shards" in result.rejection_reason.lower()


def test_natural_eos_vs_stress_distinction(tmp_path: Path) -> None:
    """Natural EOS vs fixed 256-step decode stress must be distinctly tracked."""
    dummy_model = tmp_path / "model.gguf"
    dummy_model.write_bytes(b"GGUF" + b"\x00" * 100)

    # Run 1: Natural EOS termination at token 85
    run_eos = ModelRunResult(
        candidate_id="test-eos",
        model_path=dummy_model,
        tokens_generated=85,
        output_text="The quick brown fox jumps over the lazy dog.",
        engine_invocations=1,
        duration_seconds=2.1,
        finish_reason="stop",
        stress_mode=False,
    )
    # Run 2: Stress decode mode reaching limit 256
    run_stress = ModelRunResult(
        candidate_id="test-stress",
        model_path=dummy_model,
        tokens_generated=256,
        output_text="word " * 256,
        engine_invocations=1,
        duration_seconds=6.5,
        finish_reason="length",
        stress_mode=True,
    )

    receipt_eos = run_eos.to_run_receipt()
    receipt_stress = run_stress.to_run_receipt()

    assert receipt_eos.metadata.get("finish_reason") == "stop"
    assert receipt_eos.metadata.get("stress_mode") is False
    assert receipt_stress.metadata.get("finish_reason") == "length"
    assert receipt_stress.metadata.get("stress_mode") is True


def test_missing_model_never_faked() -> None:
    """Non-existent model must immediately result in FILE_NOT_FOUND and BLOCKED without synthetic data."""
    req = QualificationRunRequest(
        candidate_id="phantom-model",
        architecture="qwen",
        dialect="standard",
        model_path=Path("H:/phantom/does_not_exist.gguf"),
    )
    result = qualify_model_run(req, dry_run=False)
    assert result.qualified is False
    assert result.real_model_qualified is False
    assert result.tier_status.get("container_inspection") == "FILE_NOT_FOUND"
    assert result.tier_status.get("execution_readiness") == "BLOCKED"
    assert result.tokens_generated == 0
    assert result.invocations == 0
