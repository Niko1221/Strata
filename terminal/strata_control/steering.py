"""strata_control/steering.py - Research steering vector mathematics (project, reflect, add) and profile manager."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass(frozen=True)
class SteeringProfile:
    mode: str = "off"  # off, project, reflect, add
    vector: list[float] = field(default_factory=list)
    strength: float = 1.0
    layer_indices: list[int] = field(default_factory=list)
    hidden_dim: int = 4096
    model_family: str = "qwen"  # qwen, swift, deepseek41

    def __post_init__(self) -> None:
        if self.vector and self.hidden_dim == 4096 and len(self.vector) != 4096:
            object.__setattr__(self, "hidden_dim", len(self.vector))


def apply_steering_transform(
    hidden_state: list[float],
    profile: SteeringProfile,
    model_family: str | None = None,
) -> list[float]:
    """Apply linear algebraic transformation to hidden state with strict model family isolation."""
    if profile.mode == "off" or profile.strength == 0.0 or not profile.vector:
        return list(hidden_state)

    # Model family isolation check
    if model_family is not None and profile.model_family.lower() != model_family.lower():
        raise ValueError(
            f"Steering profile family '{profile.model_family}' cannot be applied to model family '{model_family}'. "
            f"Residual vector steering is isolated per architecture."
        )

    # Dimension verification
    if len(hidden_state) != profile.hidden_dim:
        raise ValueError(
            f"Hidden state dimension mismatch: expected {profile.hidden_dim}, got {len(hidden_state)}"
        )
    if len(profile.vector) != profile.hidden_dim:
        raise ValueError(
            f"Steering vector dimension mismatch: expected {profile.hidden_dim}, got {len(profile.vector)}"
        )

    v = profile.vector
    # Normalize v
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-12:
        return list(hidden_state)
    u = [x / norm for x in v]

    # Inner product (h . u)
    dot = sum(h * ui for h, ui in zip(hidden_state, u))

    if profile.mode == "project":
        # Projection: h - (h . u) u
        # Notice sign invariance: (-u) gives (-dot) * (-u) = dot * u, identically unchanged!
        return [h - profile.strength * dot * ui for h, ui in zip(hidden_state, u)]

    elif profile.mode == "reflect":
        # Reflection: h - 2 * (h . u) u (Householder reflection, norm preserving at strength=1.0)
        return [h - 2.0 * profile.strength * dot * ui for h, ui in zip(hidden_state, u)]

    elif profile.mode == "add":
        # Bias addition: h + strength * u
        return [h + profile.strength * ui for h, ui in zip(hidden_state, u)]

    else:
        raise ValueError(f"Unknown steering mode: {profile.mode}")
