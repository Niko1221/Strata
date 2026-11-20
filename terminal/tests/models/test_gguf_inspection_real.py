from pathlib import Path
import pytest

from strata_control.model_bundle import inspect_gguf, GgufDescriptor
from tests.fixtures.gguf_builder import MinimalGgufBuilder


def test_arbitrary_filename_qwen35(tmp_path: Path):
    """Verify that a model with arbitrary filename is identified by internal metadata bytes."""
    builder = MinimalGgufBuilder(version=3)
    builder.add_string("general.architecture", "qwen35")
    builder.add_tensor("blk.0.attn_q.weight", [4096, 4096], 0, 0, b"\x00" * 32)
    raw = builder.build()

    arbitrary_file = tmp_path / "custom_arbitrary_name_no_clues.bin"
    arbitrary_file.write_bytes(raw)

    desc = inspect_gguf(arbitrary_file)
    assert desc.architecture == "qwen35"
    assert desc.is_runnable is True
    assert desc.tensor_count == 1
    assert len(desc.tensors) == 1
    assert desc.tensors[0].name == "blk.0.attn_q.weight"
    assert desc.tensors[0].dimensions == [4096, 4096]


def test_metadata_only_not_runnable(tmp_path: Path):
    """Verify that a 0-tensor metadata GGUF shard is marked as is_metadata_only=True and is_runnable=False."""
    builder = MinimalGgufBuilder(version=3)
    builder.add_string("general.architecture", "qwen2")
    raw = builder.build()

    meta_file = tmp_path / "meta_shard.gguf"
    meta_file.write_bytes(raw)

    desc = inspect_gguf(meta_file)
    assert desc.architecture == "qwen2"
    assert desc.is_metadata_only is True
    assert desc.is_runnable is False
    assert desc.unrunnable_reason == "GGUF file contains metadata but no tensors (shard or metadata auxiliary)"


def test_header_truncation_raises(tmp_path: Path):
    """Verify that truncated GGUF header raises ValueError."""
    builder = MinimalGgufBuilder(version=3)
    builder.add_string("general.architecture", "llama")
    builder.add_tensor("tensor1", [1024], 0, 0)
    raw = builder.build()

    truncated_file = tmp_path / "truncated.gguf"
    # Cut in half
    truncated_file.write_bytes(raw[:len(raw) // 2])

    with pytest.raises(ValueError, match="truncated|limit"):
        inspect_gguf(truncated_file)


def test_unsupported_version_rejected(tmp_path: Path):
    """Verify that unsupported GGUF version raises ValueError."""
    builder = MinimalGgufBuilder(version=1)
    raw = builder.build()

    v1_file = tmp_path / "v1.gguf"
    v1_file.write_bytes(raw)

    with pytest.raises(ValueError, match="Unsupported GGUF container version 1"):
        inspect_gguf(v1_file)


def test_overflow_dimension_rejected(tmp_path: Path):
    """Verify integer overflow guard on excessive dimensions (GHSA-vgg9-87g3-85w8)."""
    builder = MinimalGgufBuilder(version=3)
    builder.add_string("general.architecture", "dsv41")
    # 2^32 * 2^32 overflows 64-bit int
    builder.add_tensor("overflow_tensor", [10_000_000, 10_000_000, 10_000_000], 0, 0)
    raw = builder.build()

    overflow_file = tmp_path / "overflow.gguf"
    overflow_file.write_bytes(raw)

    with pytest.raises(ValueError, match="overflow"):
        inspect_gguf(overflow_file)
