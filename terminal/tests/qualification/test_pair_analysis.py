"""tests/qualification/test_pair_analysis.py - TDD tests for Task R17 Paired Benchmark Analysis.
Conforms strictly to MILSPEC-Codex standard: logging only, zero print, strict types.
"""
from __future__ import annotations

import pytest
from typing import List

from strata_control.pair_analysis import (
    RunSample,
    PairedComparisonConfig,
    analyze_paired_runs,
    PairedAdoptionStatus,
)


def test_mismatched_tokenizer_or_model_hash_rejected() -> None:
    """Comparing runs with different tokenizers or underlying model hashes must be rejected."""
    baseline = [
        RunSample(
            run_id="b1", config_name="baseline", model_hash="hash_A",
            tokenizer_hash="tok_A", context_length=4096, is_warmup=False,
            ttft_ms=100.0, tpot_ms=25.0, decode_tok_per_sec=40.0
        )
    ]
    candidate = [
        RunSample(
            run_id="c1", config_name="candidate", model_hash="hash_B",  # mismatched model hash
            tokenizer_hash="tok_A", context_length=4096, is_warmup=False,
            ttft_ms=90.0, tpot_ms=22.0, decode_tok_per_sec=45.0
        )
    ]
    result = analyze_paired_runs(baseline, candidate, PairedComparisonConfig())
    assert result.status == PairedAdoptionStatus.REJECTED
    assert any("hash mismatch" in r.lower() for r in result.rejection_reasons)


def test_insufficient_warmup_rejects_adoption() -> None:
    """Adoption cannot be granted without at least 3 verified warmup runs."""
    baseline = [
        RunSample(
            run_id=f"b_{i}", config_name="base", model_hash="hash_A",
            tokenizer_hash="tok_A", context_length=4096,
            is_warmup=(i < 1),  # Only 1 warmup instead of required 3
            ttft_ms=100.0, tpot_ms=25.0, decode_tok_per_sec=40.0
        )
        for i in range(12)
    ]
    candidate = [
        RunSample(
            run_id=f"c_{i}", config_name="cand", model_hash="hash_A",
            tokenizer_hash="tok_A", context_length=4096,
            is_warmup=(i < 1),
            ttft_ms=90.0, tpot_ms=22.0, decode_tok_per_sec=45.0
        )
        for i in range(12)
    ]
    result = analyze_paired_runs(baseline, candidate, PairedComparisonConfig(min_warmup_runs=3))
    assert result.status == PairedAdoptionStatus.REJECTED
    assert any("warmup" in r.lower() for r in result.rejection_reasons)


def test_unpaired_or_timedout_samples_filtered_out() -> None:
    """Timed-out or unpaired samples must never be silently included in adoption stats."""
    baseline = [
        RunSample(
            run_id=f"b_{i}", config_name="base", model_hash="hash_A",
            tokenizer_hash="tok_A", context_length=4096, is_warmup=(i < 3),
            ttft_ms=100.0, tpot_ms=25.0, decode_tok_per_sec=40.0,
            timed_out=(i == 5)
        )
        for i in range(13)
    ]
    candidate = [
        RunSample(
            run_id=f"c_{i}", config_name="cand", model_hash="hash_A",
            tokenizer_hash="tok_A", context_length=4096, is_warmup=(i < 3),
            ttft_ms=85.0, tpot_ms=21.0, decode_tok_per_sec=47.6,
            timed_out=False
        )
        for i in range(13)
    ]
    result = analyze_paired_runs(baseline, candidate, PairedComparisonConfig(min_warmup_runs=3, min_paired_runs=9))
    # Sample 5 was timed out in baseline, so pair 5 should be discarded
    assert result.valid_pairs_count == 9  # 13 total - 3 warmup - 1 timed out pair = 9


def test_performance_under_target_classified_as_experimental() -> None:
    """Implementations that function correctly but fail the 1.05x speedup threshold are classified as EXPERIMENTAL, not deleted."""
    baseline = [
        RunSample(
            run_id=f"b_{i}", config_name="base", model_hash="hash_A",
            tokenizer_hash="tok_A", context_length=4096, is_warmup=(i < 3),
            ttft_ms=100.0, tpot_ms=25.0, decode_tok_per_sec=40.0
        )
        for i in range(13)
    ]
    candidate = [
        RunSample(
            run_id=f"c_{i}", config_name="cand", model_hash="hash_A",
            tokenizer_hash="tok_A", context_length=4096, is_warmup=(i < 3),
            ttft_ms=100.0, tpot_ms=24.8, decode_tok_per_sec=40.3  # Only ~1.007x speedup (< 1.05x target)
        )
        for i in range(13)
    ]
    result = analyze_paired_runs(baseline, candidate, PairedComparisonConfig(min_warmup_runs=3, target_speedup=1.05))
    assert result.status == PairedAdoptionStatus.EXPERIMENTAL
    assert result.speedup_ratio < 1.05


def test_successful_paired_adoption() -> None:
    """Clear 1.1x speedup with nominal TTFT and PPL satisfies adoption criteria."""
    baseline = [
        RunSample(
            run_id=f"b_{i}", config_name="base", model_hash="hash_A",
            tokenizer_hash="tok_A", context_length=4096, is_warmup=(i < 3),
            ttft_ms=100.0, tpot_ms=25.0, decode_tok_per_sec=40.0, ppl=5.20
        )
        for i in range(13)
    ]
    candidate = [
        RunSample(
            run_id=f"c_{i}", config_name="cand", model_hash="hash_A",
            tokenizer_hash="tok_A", context_length=4096, is_warmup=(i < 3),
            ttft_ms=95.0, tpot_ms=21.0, decode_tok_per_sec=47.6, ppl=5.21
        )
        for i in range(13)
    ]
    result = analyze_paired_runs(baseline, candidate, PairedComparisonConfig(min_warmup_runs=3, target_speedup=1.05))
    assert result.status == PairedAdoptionStatus.ACCEPTED
    assert result.speedup_ratio >= 1.05
    assert result.confidence_interval_95[0] > 1.0
