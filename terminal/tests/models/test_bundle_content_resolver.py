from pathlib import Path
import pytest

from strata_control.bundle_resolver import (
    ImportSelection,
    ReplacementRecipe,
    resolve_bundle,
)
from strata_control.models import _assess_projector, plan_import
from tests.fixtures.gguf_builder import MinimalGgufBuilder


def test_merged_lora_filename_is_base_model(tmp_path: Path):
    """Verify that a full model file with 'lora' in the filename is resolved as a base model."""
    builder = MinimalGgufBuilder(version=3)
    builder.add_string("general.architecture", "qwen35")
    builder.add_tensor("blk.0.attn_q.weight", [8, 1], 0, 0, b"\x00" * 32)
    raw = builder.build()

    lora_named_file = tmp_path / "Hikari-LoRA-fixed-scale-PQ2_0.gguf"
    lora_named_file.write_bytes(raw)

    selection = ImportSelection(primary_path=lora_named_file, architecture="qwen35")
    bundle = resolve_bundle(selection)
    assert bundle.split_total == 1
    assert len(bundle.files) == 1
    assert bundle.files[0].path == str(lora_named_file)


def test_negative_slot_rejected(tmp_path: Path):
    """Verify that negative slot indices are strictly rejected (no Python negative indexing)."""
    builder = MinimalGgufBuilder(version=3)
    builder.add_string("general.architecture", "dsv41")
    builder.add_tensor("blk.0.attn_q.weight", [8, 1], 0, 0, b"\x00" * 32)
    raw = builder.build()

    f = tmp_path / "model.gguf"
    f.write_bytes(raw)

    rep = tmp_path / "replacement.gguf"
    rep.write_bytes(raw)

    selection = ImportSelection(primary_path=f, architecture="dsv41")
    recipe = ReplacementRecipe(parent_model_set_id="bundle-123", slot=-1, replacement_path=str(rep))

    with pytest.raises((ValueError, IndexError), match="slot"):
        resolve_bundle(selection, recipe=recipe)


def test_shards_missing_detection(tmp_path: Path):
    """Verify that missing shard in a split set is detected and fails."""
    p1 = tmp_path / "model-00001-of-00003.gguf"
    p1.write_bytes(b"GGUF" + b"\x00" * 100)
    p3 = tmp_path / "model-00003-of-00003.gguf"
    p3.write_bytes(b"GGUF" + b"\x00" * 100)

    selection = ImportSelection(primary_path=p1)
    with pytest.raises(FileNotFoundError, match="Missing required shard"):
        resolve_bundle(selection)


def test_projector_size_tolerance(tmp_path: Path):
    """Verify that mmproj with minor alignment difference is marked pinned and ok."""
    builder = MinimalGgufBuilder(version=3)
    builder.add_string("general.architecture", "clip")
    builder.add_string("clip.projector_type", "qwen3vl_merger")
    raw = builder.build()

    # Target pinned size is 907543008. Simulate 64 bytes less (907542944)
    target_size = 907542944
    mmproj_file = tmp_path / "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
    with open(mmproj_file, "wb") as f:
        f.write(raw)
        if len(raw) < target_size:
            f.seek(target_size - 1)
            f.write(b"\x00")

    res = _assess_projector(mmproj_file)
    assert res["status"] == "ok"
    assert res["pinned"] is True


def test_candidate_fallback_qwen4exp_custom_name(tmp_path: Path):
    """Verify that custom-named / abliterated qwen4exp model resolves to swift/IQ2_XS with no issues."""
    builder = MinimalGgufBuilder(version=3)
    builder.add_string("general.architecture", "qwen4exp")
    builder.add_string("tokenizer.ggml.model", "gpt2")
    builder.add_array_string("tokenizer.ggml.tokens", ["hello", "world"])
    builder.add_array_string("tokenizer.ggml.merges", ["h ello"])
    builder.add_tensor("blk.0.attn_q.weight", [8, 1], 0, 0, b"\x00" * 32)
    raw = builder.build()

    shard1 = tmp_path / "Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS-00001-of-00002.gguf"
    shard2 = tmp_path / "Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS-00002-of-00002.gguf"
    shard1.write_bytes(raw)
    shard2.write_bytes(raw)

    plan = plan_import(str(tmp_path))
    assert plan["issues"] == []
    assert plan["file_set_status"] == "plausible"
    assert plan["candidate_selection"] is not None
    assert plan["candidate_selection"]["family"] == "swift"
    assert plan["candidate_selection"]["model"] == "IQ2_XS"
    assert plan["execution_compatibility"] == "unverified"


@pytest.mark.parametrize("name", ["Swift-custom-Q2_0.gguf", "Swift-custom.gguf", "Flash-custom-TQ1_0.gguf"])
def test_qwen4exp_name_does_not_invent_an_unsupported_swift_variant(tmp_path: Path, name: str):
    builder = MinimalGgufBuilder(version=3)
    builder.add_string("general.architecture", "qwen4exp")
    builder.add_string("tokenizer.ggml.model", "gpt2")
    builder.add_array_string("tokenizer.ggml.tokens", ["hello", "world"])
    builder.add_array_string("tokenizer.ggml.merges", ["h ello"])
    builder.add_tensor("blk.0.attn_q.weight", [8, 1], 0, 0, b"\x00" * 32)
    path = tmp_path / name
    path.write_bytes(builder.build())
    plan = plan_import(str(path))
    assert plan["candidate_selection"] is None
    assert "unrecognized_or_unsupported_model_variant" in plan["issues"]
    assert plan["would_write"] is False


def test_candidate_fallback_deepseek41(tmp_path: Path):
    """Verify that deepseek41 architecture resolves to family deepseek41 (not deepseek)."""
    builder = MinimalGgufBuilder(version=3)
    builder.add_string("general.architecture", "deepseek41")
    builder.add_string("tokenizer.ggml.model", "gpt2")
    builder.add_tensor("blk.0.attn_q.weight", [8, 1], 0, 0, b"\x00" * 32)
    raw = builder.build()

    shard = tmp_path / "DeepSeek-V4.1-custom.gguf"
    shard.write_bytes(raw)

    plan = plan_import(str(shard))
    assert plan["candidate_selection"] is not None
    assert plan["candidate_selection"]["family"] == "deepseek41"
    assert plan["candidate_selection"]["model"] == "IST-DAS-2.00bpw"
