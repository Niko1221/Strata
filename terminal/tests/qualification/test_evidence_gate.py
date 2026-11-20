import json
from pathlib import Path
import pytest

from strata_control.qualification import evaluate_gate, GateStatus


def test_dryrun_never_qualifies():
    """Verify that old receipts marked as DRYRUN_QUALIFICATION_RECEIPT are rejected as real_model_qualified."""
    receipt = {
        "schema_version": "2.0",
        "candidate_id": "deepseek-v4.1-flash-ista-standard",
        "verification_method": "DRYRUN_QUALIFICATION_RECEIPT",
        "execution_readiness": "READY_FOR_INFERENCE",
        "invocations": 0,
        "raw_log_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "binary_sha256": "45b8628226f35051195520ab418e55be92093ccf",
        "tokens_generated": 0,
    }
    result = evaluate_gate(receipt)
    assert result.status == GateStatus.REJECTED
    assert result.real_model_qualified is False
    assert "dryrun" in result.reason.lower()


def test_zero_invocations_never_pass():
    """Verify that having zero invocations of the real model binary rejects qualification."""
    receipt = {
        "schema_version": "2.0",
        "candidate_id": "deepseek-v4.1-flash-ista-standard",
        "verification_method": "REAL_MODEL_EXECUTION",
        "execution_readiness": "READY_FOR_INFERENCE",
        "invocations": 0,
        "raw_log_sha256": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
        "binary_sha256": "ca37ff9f0120fccac47ddb8e8b1b7f3341889aab4c32e9b2ef9ccf63e1497a9a",
        "tokens_generated": 0,
    }
    result = evaluate_gate(receipt)
    assert result.status == GateStatus.REJECTED
    assert result.real_model_qualified is False
    assert "invocations" in result.reason.lower()


def test_tampered_log_hash(tmp_path: Path):
    """Verify that mismatch between actual log hash and recorded hash fails gate."""
    log_file = tmp_path / "inference.log"
    log_file.write_text("actual token output stream: hello world", encoding="utf-8")
    tampered_hash = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

    receipt = {
        "schema_version": "2.0",
        "candidate_id": "deepseek-v4.1-flash-ista-standard",
        "verification_method": "REAL_MODEL_EXECUTION",
        "execution_readiness": "READY_FOR_INFERENCE",
        "invocations": 1,
        "log_path": str(log_file),
        "raw_log_sha256": tampered_hash,
        "binary_sha256": "ca37ff9f0120fccac47ddb8e8b1b7f3341889aab4c32e9b2ef9ccf63e1497a9a",
        "tokens_generated": 10,
    }
    result = evaluate_gate(receipt)
    assert result.status == GateStatus.REJECTED
    assert result.real_model_qualified is False
    assert "hash mismatch" in result.reason.lower()


def test_wrong_binary_same_model():
    """Verify that unexpected or placeholder binary hash is rejected."""
    receipt = {
        "schema_version": "2.0",
        "candidate_id": "deepseek-v4.1-flash-ista-standard",
        "verification_method": "REAL_MODEL_EXECUTION",
        "execution_readiness": "READY_FOR_INFERENCE",
        "invocations": 1,
        "raw_log_sha256": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
        "binary_sha256": "native-sha",  # placeholder hash
        "tokens_generated": 10,
    }
    result = evaluate_gate(receipt)
    assert result.status == GateStatus.REJECTED
    assert result.real_model_qualified is False
    assert "placeholder" in result.reason.lower() or "binary" in result.reason.lower()


def test_fixed_timestamp_not_runtime():
    """Verify that receipts without dynamic execution evidence or with static dummy timestamps fail."""
    receipt = {
        "schema_version": "2.0",
        "candidate_id": "deepseek-v4.1-flash-ista-standard",
        "verification_method": "REAL_MODEL_EXECUTION",
        "execution_readiness": "READY_FOR_INFERENCE",
        "timestamp": "2026-10-25T02:45:00Z",  # fixed static timestamp from old run
        "duration_seconds": 0.0,
        "invocations": 1,
        "raw_log_sha256": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
        "binary_sha256": "ca37ff9f0120fccac47ddb8e8b1b7f3341889aab4c32e9b2ef9ccf63e1497a9a",
        "tokens_generated": 10,
    }
    result = evaluate_gate(receipt)
    assert result.status == GateStatus.REJECTED
    assert result.real_model_qualified is False
