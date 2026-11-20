"""tests/models/test_gguf_descriptor.py - Tests for bounded GGUF descriptor & inspection."""
from __future__ import annotations

import struct
from pathlib import Path
import pytest

from strata_control.model_bundle import inspect_gguf, InspectionLimits, GgufDescriptor
from strata_control.gguf_dialects import resolve_dialect_type, DialectType


def test_metadata_only_first_shard(tmp_path: Path) -> None:
    """A valid first shard with 0 tensors (metadata only) is inspectable."""
    fake_gguf = tmp_path / "shard_00000.gguf"
    # Construct minimal GGUF header: magic 'GGUF', version 3, n_tensors=0, n_metadata=0
    header = b"GGUF" + struct.pack("<IQQ", 3, 0, 0)
    fake_gguf.write_bytes(header)

    desc = inspect_gguf(fake_gguf)
    assert desc.tensor_count == 0
    assert desc.is_runnable is True  # Metadata shards are valid components


def test_unknown_arch_is_inspectable_not_runnable(tmp_path: Path) -> None:
    """Unknown architectures return metadata and is_runnable=False with a clear reason."""
    fake_gguf = tmp_path / "exotic_model.gguf"
    # 1 tensor, 0 metadata
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
    fake_gguf.write_bytes(header)

    desc = inspect_gguf(fake_gguf)
    assert desc.tensor_count == 1
    assert desc.is_runnable is False
    assert desc.unrunnable_reason is not None
    assert "no verified local backend" in desc.unrunnable_reason


def test_dtype_collision_requires_dialect() -> None:
    """Verifies that dialect resolution differentiates between standard and prism types."""
    # Standard F32
    d_f32 = resolve_dialect_type(0)
    assert d_f32.name == "F32"
    assert d_f32.dialect == DialectType.STANDARD

    # Prism PQ2_0
    d_pq2 = resolve_dialect_type(1001, explicit_dialect="prism")
    assert d_pq2.name == "PQ2_0"
    assert d_pq2.dialect == DialectType.PRISM

    # Unknown type
    with pytest.raises(ValueError, match="Unknown quantization type id"):
        resolve_dialect_type(99999)


def test_invalid_magic_rejected(tmp_path: Path) -> None:
    """Files without GGUF magic bytes are rejected."""
    bad_file = tmp_path / "corrupt.gguf"
    bad_file.write_bytes(b"NOT_A_GGUF_FILE")

    with pytest.raises(ValueError, match="Invalid magic bytes"):
        inspect_gguf(bad_file)
