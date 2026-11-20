"""tests/security/test_gguf_admission_limits.py - Security tests for integer overflow and allocation guards (GHSA-vgg9-87g3-85w8)."""
from __future__ import annotations

import struct
from pathlib import Path
import pytest

from strata_control.gguf_preflight import check_tensor_dimensions


def test_tensor_size_overflow() -> None:
    """Verifies that integer overflow in tensor shape products is caught before allocation."""
    # A tensor claiming dimensions that multiply past 2^63 - 1 must raise ValueError
    overflow_dims = [2**32, 2**32]
    with pytest.raises(ValueError, match="Tensor dimensions overflow"):
        check_tensor_dimensions(overflow_dims, type_size=2)

    # Excessive non-overflow dimension exceeding max single tensor limit (e.g. 100 TB)
    huge_dims = [1024, 1024, 1024, 256]  # 256 GB for a single tensor
    with pytest.raises(ValueError, match="exceeds maximum allowed tensor size"):
        check_tensor_dimensions(huge_dims, type_size=2, max_tensor_bytes=64 * 1024 * 1024 * 1024)


def test_valid_tensor_dimensions() -> None:
    """Normal tensor dimensions pass within allowed limits."""
    valid_dims = [4096, 4096]  # 16M elements * 2 bytes = 32 MB
    n_bytes = check_tensor_dimensions(valid_dims, type_size=2)
    assert n_bytes == 32 * 1024 * 1024
