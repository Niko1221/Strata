#!/usr/bin/env python3
"""Sakana AI ShinkaEvolve (arXiv:2509.19349) + robust-kbench (arXiv:2509.14279)
evolutionary optimizer for Riemann-Hypothesis / Fate / HybriMoE MoE cache parameters.

Uses adaptive parent sampling, novelty rejection, UCB1 bandit mutation operator selection,
and strict invariant verification gates (preventing reward hacking).
"""
from __future__ import annotations

import dataclasses
import logging
import math
import random
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("shinka_evolve_rh")


@dataclasses.dataclass(frozen=True)
class CacheGenome:
    s_shallow: float
    s_deep: float
    von_koch_c: float
    impact_alpha: float

    def as_vec(self) -> Tuple[float, float, float, float]:
        return (self.s_shallow, self.s_deep, self.von_koch_c, self.impact_alpha)


@dataclasses.dataclass
class EvalResult:
    genome: CacheGenome
    hit_rate: float
    weighted_cpu_saved: float
    verified: bool
    slots: List[int]


def truncated_zeta(n_experts: int, s: float) -> float:
    return sum((k ** (-s)) for k in range(1, n_experts + 1))


def zeta_lagrangian_allocate(
    costs: List[float],
    strides: List[int],
    exponents: List[float],
    budget: int,
    max_slots: int = 128,
) -> List[int]:
    n_layers = len(costs)
    slots = [0] * n_layers
    inv_z = [1.0 / truncated_zeta(max_slots, s) for s in exponents]
    used = 0
    while True:
        best_i = -1
        best_mu = -1.0
        for i in range(n_layers):
            if slots[i] >= max_slots or used + strides[i] > budget:
                continue
            rank = float(slots[i] + 1)
            mu = (costs[i] * (rank ** (-exponents[i])) * inv_z[i]) / float(strides[i])
            if mu > best_mu:
                best_mu = mu
                best_i = i
        if best_i < 0:
            break
        slots[best_i] += 1
        used += strides[best_i]
    return slots


def robust_kbench_verify(genome: CacheGenome, slots: List[int], strides: List[int], budget: int) -> bool:
    """Strict robust-kbench (arXiv:2509.14279) invariant gate: prevents invalid/reward-hacked genomes."""
    if not (0.80 <= genome.s_shallow <= 2.20 and 0.80 <= genome.s_deep <= 2.20):
        return False
    if not (0.0 <= genome.von_koch_c <= 1.0 and 0.0 <= genome.impact_alpha <= 2.0):
        return False
    used = sum(s * b for s, b in zip(slots, strides))
    if used > budget or any(s < 0 or s > 128 for s in slots):
        return False
    return True


def evaluate_genome(
    genome: CacheGenome,
    traces: List[List[Tuple[int, float]]],
    costs: List[float],
    strides: List[int],
    budget: int,
) -> EvalResult:
    n_layers = len(costs)
    exponents = [genome.s_shallow if i < n_layers // 2 else genome.s_deep for i in range(n_layers)]
    slots = zeta_lagrangian_allocate(costs, strides, exponents, budget)
    if not robust_kbench_verify(genome, slots, strides, budget):
        return EvalResult(genome, 0.0, 0.0, False, slots)

    # Simulate RH critical-line decay + von Koch hysteresis + HybriMoE impact scoring on traces
    total_lookups = 0
    total_hits = 0
    weighted_saved = 0.0
    for li in range(n_layers):
        cap = slots[li]
        resident = set(range(cap))
        scores: Dict[int, float] = {e: 0.0 for e in range(128)}
        rebal_step = 0
        for step_idx, (exp_id, ew) in enumerate(traces[li], start=1):
            total_lookups += 1
            if exp_id in resident:
                total_hits += 1
                weighted_saved += costs[li]
            scores[exp_id] += 1.0 + genome.impact_alpha * ew
            if step_idx % 32 == 0 and cap > 0:
                rebal_step += 1
                decay = min(0.96, max(0.50, 1.0 - 1.0 / math.sqrt(rebal_step + 4.0)))
                ranked = sorted(range(128), key=lambda e: scores[e], reverse=True)
                cold_res = sorted(resident, key=lambda e: scores[e])
                swaps = 0
                for hot in ranked[:cap]:
                    if hot in resident or swaps >= 3:
                        continue
                    cold = cold_res[swaps]
                    s_h, s_c = scores[hot], scores[cold]
                    thr = (
                        genome.von_koch_c * math.sqrt(max(0.0, s_h + s_c)) / max(1.0, math.log(rebal_step + math.e))
                        if s_c > 0.0
                        else 0.0
                    )
                    if s_h - s_c > thr:
                        resident.remove(cold)
                        resident.add(hot)
                        swaps += 1
                for e in range(128):
                    scores[e] *= decay
    hit_rate = total_hits / max(1, total_lookups)
    return EvalResult(genome, hit_rate, weighted_saved, True, slots)


def generate_synthetic_moe_traces(seed: int = 20261123, n_layers: int = 12, steps: int = 384) -> List[List[Tuple[int, float]]]:
    """Generates Fate-realistic Zipf/Zeta MoE routing traces (shallow layers steeper than deep)."""
    rng = random.Random(seed)
    traces: List[List[Tuple[int, float]]] = []
    for li in range(n_layers):
        true_s = 1.42 if li < n_layers // 2 else 1.04
        weights = [(k ** (-true_s)) for k in range(1, 129)]
        perm = list(range(128))
        rng.shuffle(perm)
        layer_trace: List[Tuple[int, float]] = []
        for _ in range(steps):
            chosen = rng.choices(perm, weights=weights, k=1)[0]
            ew = 0.05 + 0.35 * (1.0 / (1.0 + perm.index(chosen)))
            layer_trace.append((chosen, ew))
        traces.append(layer_trace)
    return traces


def run_shinka_evolve(generations: int = 24, seed: int = 42) -> EvalResult:
    rng = random.Random(seed)
    traces = generate_synthetic_moe_traces()
    # 12 outer layers matching Sakura-Inkling-2-Mini (6 shallow, 6 deep; heterogeneous Q8_0 / Q5_0 costs)
    costs = [1.30, 1.30, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.30, 1.30]
    strides = [4800000] * 12
    budget = 64 * 4800000

    baseline = CacheGenome(1.15, 1.15, 0.20, 0.0)
    archive: List[EvalResult] = [evaluate_genome(baseline, traces, costs, strides, budget)]
    logger.info(
        "Baseline genome=%s hit_rate=%.4f weighted_saved=%.1f slots=%s",
        baseline,
        archive[0].hit_rate,
        archive[0].weighted_cpu_saved,
        archive[0].slots,
    )

    # UCB1 bandit over 3 mutation operators (ShinkaEvolve)
    op_counts = [1, 1, 1]
    op_rewards = [0.0, 0.0, 0.0]

    for gen in range(1, generations + 1):
        # Adaptive parent sampling (fitness-proportionate over top-5 archive)
        top_k = sorted(archive, key=lambda r: r.weighted_cpu_saved, reverse=True)[:5]
        parent = rng.choice(top_k).genome

        total_pulls = sum(op_counts)
        ucb = [
            (op_rewards[i] / op_counts[i]) + 0.5 * math.sqrt(math.log(total_pulls + 1.0) / op_counts[i])
            for i in range(3)
        ]
        op = max(range(3), key=lambda i: ucb[i])

        if op == 0:  # Local Gaussian mutation
            cand = CacheGenome(
                min(2.2, max(0.8, parent.s_shallow + rng.gauss(0.0, 0.12))),
                min(2.2, max(0.8, parent.s_deep + rng.gauss(0.0, 0.12))),
                min(1.0, max(0.0, parent.von_koch_c + rng.gauss(0.0, 0.06))),
                min(2.0, max(0.0, parent.impact_alpha + rng.gauss(0.0, 0.20))),
            )
        elif op == 1:  # Heavy-tail exploration jump
            cand = CacheGenome(
                rng.uniform(1.0, 1.9),
                rng.uniform(0.85, 1.4),
                rng.uniform(0.0, 0.5),
                rng.uniform(0.0, 1.2),
            )
        else:  # Differential crossover between two top parents
            p2 = rng.choice(top_k).genome
            w = rng.uniform(0.2, 0.8)
            cand = CacheGenome(
                w * parent.s_shallow + (1.0 - w) * p2.s_shallow,
                w * parent.s_deep + (1.0 - w) * p2.s_deep,
                w * parent.von_koch_c + (1.0 - w) * p2.von_koch_c,
                w * parent.impact_alpha + (1.0 - w) * p2.impact_alpha,
            )

        # Novelty rejection filter
        v = cand.as_vec()
        if any(
            math.sqrt(sum((a - b) ** 2 for a, b in zip(v, r.genome.as_vec()))) < 0.02
            for r in archive
        ):
            continue

        res = evaluate_genome(cand, traces, costs, strides, budget)
        op_counts[op] += 1
        if res.verified:
            best_before = max(r.weighted_cpu_saved for r in archive)
            reward = max(0.0, res.weighted_cpu_saved - best_before) / max(1.0, best_before)
            op_rewards[op] += reward
            archive.append(res)

    best = max(archive, key=lambda r: r.weighted_cpu_saved)
    logger.info(
        "ShinkaEvolve best genome=%s hit_rate=%.4f weighted_saved=%.1f (+%.2f%%) slots=%s",
        best.genome,
        best.hit_rate,
        best.weighted_cpu_saved,
        100.0 * (best.weighted_cpu_saved / max(1e-9, archive[0].weighted_cpu_saved) - 1.0),
        best.slots,
    )
    return best


if __name__ == "__main__":
    run_shinka_evolve()
