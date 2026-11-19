"""tests/models/test_bundle_resolver.py - Tests for shard bundle assembly and replacement overlays."""
from __future__ import annotations

from pathlib import Path
import pytest

from strata_control.bundle_resolver import (
    ImportSelection,
    ReplacementRecipe,
    resolve_bundle,
)


def test_pick_middle_shard(tmp_path: Path) -> None:
    """Selecting any middle shard resolves to the complete ordered bundle."""
    # Create 4 shards
    shards = []
    for i in range(1, 5):
        s = tmp_path / f"model-0000{i}-of-00004.gguf"
        s.write_bytes(b"dummy-shard-content-" + str(i).encode())
        shards.append(s)

    # Pick middle shard 3
    sel = ImportSelection(primary_path=shards[2])
    bundle = resolve_bundle(sel)

    assert bundle.split_total == 4
    assert len(bundle.files) == 4
    assert [f.ordinal for f in bundle.files] == [0, 1, 2, 3]
    assert Path(bundle.files[0].path).name == "model-00001-of-00004.gguf"
    assert Path(bundle.files[3].path).name == "model-00004-of-00004.gguf"


def test_mixed_layout_missing_shard(tmp_path: Path) -> None:
    """Incomplete multi-part sets raise FileNotFoundError."""
    s1 = tmp_path / "model-00001-of-00003.gguf"
    s1.write_bytes(b"shard1")
    # Missing shard 2 and 3

    sel = ImportSelection(primary_path=s1)
    with pytest.raises(FileNotFoundError, match="Missing required shard"):
        resolve_bundle(sel)


def test_replacement_overlay(tmp_path: Path) -> None:
    """A replacement recipe safely replaces only the specified slot in memory."""
    s1 = tmp_path / "part-01-of-02.gguf"
    s2 = tmp_path / "part-02-of-02.gguf"
    s1.write_bytes(b"base-1")
    s2.write_bytes(b"base-2")

    rep_shard = tmp_path / "abliterated-part-02.gguf"
    rep_shard.write_bytes(b"abliterated-2")

    recipe = ReplacementRecipe(
        parent_model_set_id="base-model-id",
        slot=1,
        replacement_path=str(rep_shard),
        replacement_sha256="abc123sha",
    )

    sel = ImportSelection(primary_path=s1)
    bundle = resolve_bundle(sel, recipe=recipe)

    assert len(bundle.files) == 2
    assert bundle.files[0].path == str(s1)
    assert bundle.files[1].path == str(rep_shard)
    assert bundle.files[1].sha256 == "abc123sha"


def test_adapter_not_model(tmp_path: Path) -> None:
    """Attempting to import a lora or mmproj adapter as a base model is rejected."""
    adapter = tmp_path / "vision-mmproj-f16.gguf"
    adapter.write_bytes(b"adapter-content")

    sel = ImportSelection(primary_path=adapter, artifact_kind="model")
    with pytest.raises(ValueError, match="is an auxiliary adapter, not a runnable base model"):
        resolve_bundle(sel)
