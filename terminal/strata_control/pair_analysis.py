"""strata_control/pair_analysis.py - Rigorous statistical paired-analysis for model adoption.
Strictly conforms to MILSPEC-Codex standard: logging only, zero raw print, strict types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import math
import statistics
from typing import List, Optional, Tuple

logger = logging.getLogger("strata.pair_analysis")


class PairedAdoptionStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    EXPERIMENTAL = "EXPERIMENTAL"
    INCONCLUSIVE = "INCONCLUSIVE"
    REJECTED = "REJECTED"


@dataclass
class RunSample:
    run_id: str
    config_name: str
    model_hash: str
    tokenizer_hash: str
    context_length: int
    is_warmup: bool
    ttft_ms: float
    tpot_ms: float
    decode_tok_per_sec: float
    ppl: Optional[float] = None
    peak_vram_mib: float = 0.0
    peak_ram_mib: float = 0.0
    timed_out: bool = False


@dataclass(frozen=True)
class PairedComparisonConfig:
    min_warmup_runs: int = 3
    min_paired_runs: int = 9
    target_speedup: float = 1.05
    max_ttft_ratio: float = 1.05
    max_ppl_ratio: float = 1.01
    confidence_level: float = 0.95


@dataclass
class PairedAnalysisResult:
    status: PairedAdoptionStatus
    speedup_ratio: float
    ttft_ratio: float
    ppl_ratio: Optional[float]
    p50_candidate: float
    p50_baseline: float
    confidence_interval_95: Tuple[float, float]
    valid_pairs_count: int
    rejection_reasons: List[str] = field(default_factory=list)


def analyze_paired_runs(
    baseline: List[RunSample],
    candidate: List[RunSample],
    config: Optional[PairedComparisonConfig] = None
) -> PairedAnalysisResult:
    """Perform rigorous paired statistical evaluation between baseline and candidate configurations."""
    cfg = config or PairedComparisonConfig()
    reasons: List[str] = []

    # 1. Model & Tokenizer Hash Invariance Check
    b_model_hashes = {s.model_hash for s in baseline}
    c_model_hashes = {s.model_hash for s in candidate}
    if b_model_hashes != c_model_hashes:
        msg = f"Model hash mismatch between runs: baseline {b_model_hashes} vs candidate {c_model_hashes}"
        logger.error(msg)
        reasons.append(msg)

    b_tok_hashes = {s.tokenizer_hash for s in baseline}
    c_tok_hashes = {s.tokenizer_hash for s in candidate}
    if b_tok_hashes != c_tok_hashes:
        msg = f"Tokenizer hash mismatch between runs: baseline {b_tok_hashes} vs candidate {c_tok_hashes}"
        logger.error(msg)
        reasons.append(msg)

    b_ctx_lengths = {s.context_length for s in baseline}
    c_ctx_lengths = {s.context_length for s in candidate}
    if b_ctx_lengths != c_ctx_lengths:
        msg = f"Context length mismatch: baseline {b_ctx_lengths} vs candidate {c_ctx_lengths}"
        logger.error(msg)
        reasons.append(msg)

    if reasons:
        return PairedAnalysisResult(
            status=PairedAdoptionStatus.REJECTED,
            speedup_ratio=0.0,
            ttft_ratio=0.0,
            ppl_ratio=None,
            p50_candidate=0.0,
            p50_baseline=0.0,
            confidence_interval_95=(0.0, 0.0),
            valid_pairs_count=0,
            rejection_reasons=reasons,
        )

    # 2. Warmup Sufficiency Check
    b_warmups = sum(1 for s in baseline if s.is_warmup)
    c_warmups = sum(1 for s in candidate if s.is_warmup)
    if b_warmups < cfg.min_warmup_runs or c_warmups < cfg.min_warmup_runs:
        msg = f"Insufficient verified warmup runs: baseline had {b_warmups}, candidate had {c_warmups} (minimum required: {cfg.min_warmup_runs})"
        logger.warning(msg)
        reasons.append(msg)
        return PairedAnalysisResult(
            status=PairedAdoptionStatus.REJECTED,
            speedup_ratio=0.0,
            ttft_ratio=0.0,
            ppl_ratio=None,
            p50_candidate=0.0,
            p50_baseline=0.0,
            confidence_interval_95=(0.0, 0.0),
            valid_pairs_count=0,
            rejection_reasons=reasons,
        )

    # 3. Filter Warmup & Timed-Out Samples
    b_eval = [s for s in baseline if not s.is_warmup]
    c_eval = [s for s in candidate if not s.is_warmup]

    pairs: List[Tuple[RunSample, RunSample]] = []
    for b_s, c_s in zip(b_eval, c_eval):
        if not b_s.timed_out and not c_s.timed_out:
            pairs.append((b_s, c_s))

    valid_count = len(pairs)
    if valid_count < cfg.min_paired_runs:
        msg = f"Insufficient valid non-timeout pairs: found {valid_count} (minimum required: {cfg.min_paired_runs})"
        logger.warning(msg)
        reasons.append(msg)
        return PairedAnalysisResult(
            status=PairedAdoptionStatus.INCONCLUSIVE,
            speedup_ratio=0.0,
            ttft_ratio=0.0,
            ppl_ratio=None,
            p50_candidate=0.0,
            p50_baseline=0.0,
            confidence_interval_95=(0.0, 0.0),
            valid_pairs_count=valid_count,
            rejection_reasons=reasons,
        )

    # 4. Statistical Metrics Calculation
    pair_speedups = [c_s.decode_tok_per_sec / b_s.decode_tok_per_sec for b_s, c_s in pairs if b_s.decode_tok_per_sec > 0]
    b_tok_rates = [b_s.decode_tok_per_sec for b_s, _ in pairs]
    c_tok_rates = [c_s.decode_tok_per_sec for _, c_s in pairs]

    b_ttfts = [b_s.ttft_ms for b_s, _ in pairs]
    c_ttfts = [c_s.ttft_ms for _, c_s in pairs]

    p50_base = statistics.median(b_tok_rates)
    p50_cand = statistics.median(c_tok_rates)
    median_speedup = statistics.median(pair_speedups) if pair_speedups else 1.0

    p50_b_ttft = statistics.median(b_ttfts) if b_ttfts else 1.0
    p50_c_ttft = statistics.median(c_ttfts) if c_ttfts else 1.0
    ttft_ratio = p50_c_ttft / p50_b_ttft if p50_b_ttft > 0 else 1.0

    # PPL Ratio if available
    b_ppls = [b_s.ppl for b_s, _ in pairs if b_s.ppl is not None]
    c_ppls = [c_s.ppl for _, c_s in pairs if c_s.ppl is not None]
    ppl_ratio: Optional[float] = None
    if b_ppls and c_ppls and len(b_ppls) == len(c_ppls):
        ppl_ratio = statistics.median(c_ppls) / statistics.median(b_ppls)

    # 95% Confidence Interval Calculation (Student-t / Normal approximation on log speedup)
    if len(pair_speedups) >= 2:
        mean_speedup = statistics.mean(pair_speedups)
        stdev = statistics.stdev(pair_speedups)
        sem = stdev / math.sqrt(len(pair_speedups))
        t_crit = 2.262 if len(pair_speedups) <= 10 else 1.96
        ci_low = round(mean_speedup - t_crit * sem, 4)
        ci_high = round(mean_speedup + t_crit * sem, 4)
    else:
        ci_low = median_speedup
        ci_high = median_speedup

    ci_95 = (ci_low, ci_high)

    # 5. Adoption Gate Decision
    meets_speedup = median_speedup >= cfg.target_speedup and ci_low > 1.0
    meets_ttft = ttft_ratio <= cfg.max_ttft_ratio
    meets_ppl = (ppl_ratio is None) or (ppl_ratio <= cfg.max_ppl_ratio)

    if meets_speedup and meets_ttft and meets_ppl:
        status = PairedAdoptionStatus.ACCEPTED
        logger.info("Paired evaluation ACCEPTED: speedup=%0.3fx, ci=%s", median_speedup, ci_95)
    elif median_speedup < cfg.target_speedup:
        status = PairedAdoptionStatus.EXPERIMENTAL
        reasons.append(f"Performance did not reach {cfg.target_speedup}x speedup target (measured: {median_speedup:.3f}x). Classified as EXPERIMENTAL.")
        logger.info("Paired evaluation classified as EXPERIMENTAL: speedup=%0.3fx", median_speedup)
    else:
        status = PairedAdoptionStatus.INCONCLUSIVE
        reasons.append(f"Inconclusive paired results: TTFT ratio {ttft_ratio:.3f} or PPL ratio {ppl_ratio}")

    return PairedAnalysisResult(
        status=status,
        speedup_ratio=round(median_speedup, 4),
        ttft_ratio=round(ttft_ratio, 4),
        ppl_ratio=round(ppl_ratio, 4) if ppl_ratio is not None else None,
        p50_candidate=round(p50_cand, 2),
        p50_baseline=round(p50_base, 2),
        confidence_interval_95=ci_95,
        valid_pairs_count=valid_count,
        rejection_reasons=reasons,
    )
