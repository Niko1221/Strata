import pytest

from strata_control.gguf_dialects import (
    resolve_dialect_type,
    compute_tensor_bytes,
    DialectType,
    QuantDialect,
)


def test_standard_types_resolution():
    d_f32 = resolve_dialect_type(0)
    assert d_f32.name == "F32"
    assert d_f32.block_size == 1
    assert d_f32.type_size == 4

    d_q4_0 = resolve_dialect_type(2)
    assert d_q4_0.name == "Q4_0"
    assert d_q4_0.block_size == 32
    assert d_q4_0.type_size == 18


def test_prism_pq2_0_resolution():
    # Standard GGML ID for PQ2_0 is 142 (block size 128, type size 34)
    d_pq2 = resolve_dialect_type(142, explicit_dialect="prism")
    assert d_pq2.name == "PQ2_0"
    assert d_pq2.block_size == 128
    assert d_pq2.type_size == 34
    assert d_pq2.dialect == DialectType.PRISM


def test_prism_ptq1_0_uses_the_pinned_native_layout():
    # Prism ggml-common.h: half + two high trits + 24 packed five-trit bytes.
    descriptor = resolve_dialect_type(143, explicit_dialect="prism")
    assert descriptor.block_size == 128
    assert descriptor.type_size == 28
    assert compute_tensor_bytes([256, 3], descriptor.block_size, descriptor.type_size) == 168


def test_compute_tensor_bytes_checked():
    # 4096 elements with F32 (block 1, type 4) -> 16384 bytes
    assert compute_tensor_bytes([4096], 1, 4) == 16384

    # 4096 elements with PQ2_0 (block 128, type 34) -> (4096 / 128) * 34 = 32 * 34 = 1088 bytes
    assert compute_tensor_bytes([4096], 128, 34) == 1088

    # Non-multiple of block size raises ValueError
    with pytest.raises(ValueError, match="not a multiple of block size"):
        compute_tensor_bytes([4095], 128, 34)

    # Overflowing dimension raises ValueError
    with pytest.raises(ValueError, match="overflow|Invalid dimension"):
        compute_tensor_bytes([2**32, 2**32], 1, 4)


def test_colliding_numeric_type_requires_dialect():
    # An ambiguous custom ID requires explicit dialect
    with pytest.raises(ValueError, match="explicit_dialect"):
        resolve_dialect_type(9999)
