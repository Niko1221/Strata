import math
import pytest

from strata_control.steering import SteeringProfile, apply_steering_transform


def test_project_operator_orthogonality():
    """Verify that project operator at strength 1.0 removes all component along u."""
    h = [3.0, 4.0, 5.0, 0.0]
    v = [1.0, 0.0, 0.0, 0.0]  # along x
    profile = SteeringProfile(
        mode="project",
        vector=v,
        strength=1.0,
        hidden_dim=4,
        model_family="qwen",
    )

    h_prime = apply_steering_transform(h, profile, model_family="qwen")
    # x component should be completely removed (0.0)
    assert abs(h_prime[0]) < 1e-10
    assert h_prime[1] == 4.0
    assert h_prime[2] == 5.0
    assert h_prime[3] == 0.0

    # Dot product of h_prime and u is 0
    dot = sum(hp * vi for hp, vi in zip(h_prime, v))
    assert abs(dot) < 1e-10


def test_project_sign_invariance():
    """Verify that project operator is invariant to vector sign inversion (v vs -v)."""
    h = [2.5, -1.8, 4.2, 7.1]
    v = [0.6, -0.8, 0.0, 0.0]
    neg_v = [-0.6, 0.8, 0.0, 0.0]

    p1 = SteeringProfile(mode="project", vector=v, strength=1.0, hidden_dim=4, model_family="qwen")
    p2 = SteeringProfile(mode="project", vector=neg_v, strength=1.0, hidden_dim=4, model_family="qwen")

    res1 = apply_steering_transform(h, p1, model_family="qwen")
    res2 = apply_steering_transform(h, p2, model_family="qwen")

    for x1, x2 in zip(res1, res2):
        assert abs(x1 - x2) < 1e-12


def test_reflect_norm_preservation():
    """Verify that Householder reflection preserves L2 norm at strength 1.0."""
    h = [1.0, 2.0, 3.0, 4.0]
    v = [0.5, 0.5, 0.5, 0.5]
    profile = SteeringProfile(mode="reflect", vector=v, strength=1.0, hidden_dim=4, model_family="qwen")

    h_prime = apply_steering_transform(h, profile, model_family="qwen")

    norm_orig = math.sqrt(sum(x * x for x in h))
    norm_prime = math.sqrt(sum(x * x for x in h_prime))
    assert abs(norm_orig - norm_prime) < 1e-10


def test_family_isolation():
    """Verify that Qwen steering cannot be applied to DeepSeek MLA without explicit translation."""
    profile = SteeringProfile(mode="project", vector=[1.0, 0.0], hidden_dim=2, model_family="qwen")
    with pytest.raises(ValueError, match="cannot be applied to model family"):
        apply_steering_transform([1.0, 2.0], profile, model_family="deepseek41")
