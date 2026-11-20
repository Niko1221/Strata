"""Independent tiny dense oracles for existing matrix-free paper diagnostics.

These fixtures are offline checks, never a deployable compensation path.
"""

import math

import torch

from triality_poc.core import fisher_energy
from triality_poc.reference_absmax import (
    CODEC_ID,
    dequantize_reference_absmax,
    quantize_reference_absmax,
)


def test_matrix_free_fisher_matches_dense_and_shift_invariance() -> None:
    p = torch.tensor([0.05, 0.15, 0.3, 0.5], dtype=torch.float64)
    errors = torch.tensor([[1.0, -0.5, 2.0, 0.3], [3.0, 0.2, -1.0, 2.0]])
    errors = errors.to(torch.float64)
    dense = torch.diag(p) - torch.outer(p, p)
    expected = torch.einsum("ni,ij,nj->n", errors, dense, errors)
    torch.testing.assert_close(fisher_energy(p, errors), expected)
    torch.testing.assert_close(fisher_energy(p, errors + 17), expected)


def test_score_kl_exact_identity_and_second_order_limit() -> None:
    logits = torch.tensor([-0.3, 0.7, -1.2, 1.1], dtype=torch.float64)
    direction = torch.tensor([0.4, -0.2, 0.1, 0.5], dtype=torch.float64)
    p = logits.softmax(-1)
    delta = direction * 1e-4
    # Direct probabilities are independent of the log-softmax expression in validate.
    q = (logits + delta).softmax(-1)
    direct_kl = (p * (p / q).log()).sum()
    identity = torch.logsumexp(p.log() + delta, -1) - p.dot(delta)
    dense_fisher = torch.diag(p) - torch.outer(p, p)
    quadratic = delta @ dense_fisher @ delta / 2
    torch.testing.assert_close(direct_kl, identity, atol=3e-16, rtol=1e-7)
    torch.testing.assert_close(
        direct_kl, fisher_energy(p, delta) / 2, atol=2e-14, rtol=1e-4
    )
    torch.testing.assert_close(quadratic, fisher_energy(p, delta) / 2)


def test_shared_noise_shrinkage_uses_full_cross_key_covariance() -> None:
    p = torch.tensor([0.15, 0.35, 0.5], dtype=torch.float64)
    missing = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    # Correlated shared-noise draws, centred exactly and separate from eval draws.
    noise = torch.tensor(
        [[0.1, 0.3, -0.2], [-0.1, -0.3, 0.2], [0.2, 0.1, 0.4], [-0.2, -0.1, -0.4]],
        dtype=torch.float64,
    )
    fisher = torch.diag(p) - torch.outer(p, p)
    covariance = noise.T @ noise / len(noise)
    dense_n = torch.trace(fisher @ covariance)
    matrix_free_n = fisher_energy(p, noise).mean()
    torch.testing.assert_close(matrix_free_n, dense_n)
    # A diagonal-only calculation would incorrectly discard shared-G correlation.
    diagonal_only = torch.diag(fisher).dot(torch.diag(covariance))
    assert abs(float(dense_n - diagonal_only)) > 0.003
    d = missing @ fisher @ missing
    beta = d / (d + matrix_free_n)
    assert 0 < beta < 1
    # Analytic independent-noise expected risk, not an evaluation-set fit.
    risks = [(1 - b) ** 2 * d + b**2 * dense_n for b in (0.0, beta, 1.0)]
    assert risks[1] < risks[0] and risks[1] < risks[2]
    assert abs(float(-2 * (1 - beta) * d + 2 * beta * dense_n)) < 1e-14


def test_reference_codec_identity_and_independent_levels() -> None:
    assert CODEC_ID == "paper-reference-absmax-group8-v1"
    values = torch.tensor([[-3.0, -2.0, -1.0, 0.0, 0.4, 1.0, 2.0, 3.0]])
    for bits in (2, 3, 4, 6, 8):
        qmax = 2 ** (bits - 1) - 1
        scale = 3.0 / qmax
        expected = (values / scale).round().clamp(-qmax, qmax) * scale
        encoded = quantize_reference_absmax(values, bits)
        torch.testing.assert_close(dequantize_reference_absmax(encoded), expected)
        assert encoded.codes.numel() == math.ceil(8 * bits / 8)
