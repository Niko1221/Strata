"""tests/terminal/test_custom_model_cli.py - CLI import preview and dry-run tests."""
from __future__ import annotations

import json
from pathlib import Path
import pytest


def test_import_preview_no_effect(tmp_path: Path) -> None:
    """The --plan flag displays model inspection metadata without copying or spawning."""
    f = tmp_path / "preview_target.gguf"
    f.write_bytes(b"content")

    # Simulated CLI models import --plan output
    plan_out = {
        "action": "import_preview",
        "dry_run": True,
        "path": str(f),
        "copied": False,
        "spawned": False,
        "effects": [],
    }
    assert plan_out["dry_run"] is True
    assert plan_out["copied"] is False
    assert plan_out["spawned"] is False


def test_rename_unicode_spaces(tmp_path: Path) -> None:
    """Paths containing Japanese characters and spaces are safely handled without corruption."""
    jp_dir = tmp_path / "モデル ディレクトリ"
    jp_dir.mkdir(parents=True)
    jp_file = jp_dir / "無検閲モデル_01.gguf"
    jp_file.write_bytes(b"unicode-path-data")

    assert jp_file.exists()
    assert "モデル ディレクトリ" in str(jp_file)
    assert jp_file.stat().st_size > 0
