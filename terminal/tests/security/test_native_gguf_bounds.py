from pathlib import Path
import pytest

from strata_control.model_bundle import inspect_gguf
from tests.fixtures.gguf_builder import MinimalGgufBuilder


def test_native_dimension_rank_bounds(tmp_path: Path):
    """Verify tensors with rank > 8 or rank < 1 are rejected."""
    builder = MinimalGgufBuilder(version=3)
    builder.add_string("general.architecture", "llama")
    # Rank 9 tensor
    builder.add_tensor("rank9_tensor", [2] * 9, 0, 0)
    raw = builder.build()

    f = tmp_path / "rank9.gguf"
    f.write_bytes(raw)

    with pytest.raises(ValueError, match="invalid dimension count"):
        inspect_gguf(f)


def test_native_dimension_multiplication_overflow(tmp_path: Path):
    """Verify checked multiplication guards against integer wrapping on large product."""
    builder = MinimalGgufBuilder(version=3)
    builder.add_string("general.architecture", "llama")
    # 2^31 * 2^32 overflows 64-bit signed int
    builder.add_tensor("overflow_tensor", [1 << 31, 1 << 32], 0, 0)
    raw = builder.build()

    f = tmp_path / "overflow.gguf"
    f.write_bytes(raw)

    with pytest.raises(ValueError, match="overflow"):
        inspect_gguf(f)
