from pathlib import Path
import pytest

from strata_control.steering import SteeringProfile, apply_steering_transform


def test_steering_runtime_dimension_mismatch():
    """Verify dimension mismatch between hidden state and profile is rejected."""
    profile = SteeringProfile(mode="project", vector=[1.0, 0.0, 0.0], hidden_dim=3, model_family="qwen")
    with pytest.raises(ValueError, match="dimension mismatch"):
        apply_steering_transform([1.0, 2.0], profile, model_family="qwen")


def test_steering_off_noop():
    """Verify steering mode 'off' returns input unaltered."""
    h = [1.2, 3.4, 5.6]
    profile = SteeringProfile(mode="off", vector=[1.0, 0.0, 0.0], hidden_dim=3, model_family="qwen")
    out = apply_steering_transform(h, profile, model_family="qwen")
    assert out == h
