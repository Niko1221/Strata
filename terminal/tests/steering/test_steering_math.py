"""tests/steering/test_steering_math.py - Mathematical invariants for project, reflect, and add steering."""
from __future__ import annotations

import math
import pytest

from strata_control.steering import SteeringProfile, apply_steering_transform


def test_project_sign_invariance() -> None:
    """Project operator I - v v^T is strictly invariant under direction negation v -> -v."""
    h = [1.0, 2.0, 3.0]
    v_pos = [0.0, 1.0, 0.0]
    v_neg = [0.0, -1.0, 0.0]

    p_pos = SteeringProfile(mode="project", vector=v_pos, strength=1.0)
    p_neg = SteeringProfile(mode="project", vector=v_neg, strength=1.0)

    res_pos = apply_steering_transform(h, p_pos)
    res_neg = apply_steering_transform(h, p_neg)

    assert pytest.approx(res_pos) == res_neg
    # Projected onto plane orthogonal to y-axis: [1.0, 0.0, 3.0]
    assert pytest.approx(res_pos) == [1.0, 0.0, 3.0]


def test_reflect_norm_preservation() -> None:
    """Reflection operator I - 2 v v^T preserves vector norm exactly."""
    h = [3.0, 4.0, 0.0]  # norm = 5.0
    v = [1.0, 0.0, 0.0]

    p_reflect = SteeringProfile(mode="reflect", vector=v, strength=1.0)
    res = apply_steering_transform(h, p_reflect)

    norm_orig = math.sqrt(sum(x * x for x in h))
    norm_res = math.sqrt(sum(x * x for x in res))

    assert pytest.approx(norm_orig) == norm_res
    assert pytest.approx(res) == [-3.0, 4.0, 0.0]


def test_identity_noop() -> None:
    """Mode off or strength 0 introduces strictly zero modification."""
    h = [1.5, -2.5, 3.5]
    p_off = SteeringProfile(mode="off", vector=[1.0, 0.0, 0.0])
    p_zero = SteeringProfile(mode="project", vector=[1.0, 0.0, 0.0], strength=0.0)

    assert apply_steering_transform(h, p_off) == h
    assert apply_steering_transform(h, p_zero) == h
