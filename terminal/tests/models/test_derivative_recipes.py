"""tests/models/test_derivative_recipes.py - Tests for DeepSeek derivative recipes and provenance."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from strata_control.bundle_resolver import (
    ImportSelection,
    ReplacementRecipe,
    resolve_bundle,
)
from strata_control.bundle_identity import verify_bundle

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_candidates_json_validity() -> None:
    cand_path = REPO_ROOT / "data" / "custom-models" / "candidates.json"
    assert cand_path.exists(), "candidates.json must exist"
    data = json.loads(cand_path.read_text(encoding="utf-8"))
    assert "candidates" in data
    assert len(data["candidates"]) >= 3
    for c in data["candidates"]:
        assert c["candidate_id"]
        assert c["architecture"] == "dsv41"
        assert "uncensored" in c


def test_standard_vs_derivative_same_label(tmp_path: Path) -> None:
    """Base model and derivative with same friendly label produce distinct cryptographic model_set_ids."""
    base_file = tmp_path / "model.gguf"
    base_file.write_bytes(b"base-weights-data")

    deriv_file = tmp_path / "model_abliterated.gguf"
    deriv_file.write_bytes(b"abliterated-weights-data")

    b_base = resolve_bundle(ImportSelection(primary_path=base_file, architecture="dsv41"))
    b_deriv = resolve_bundle(ImportSelection(primary_path=deriv_file, architecture="dsv41"))

    assert b_base.model_set_id != b_deriv.model_set_id


def test_no_adapter_double_apply() -> None:
    """Baked-in derivative must flag no-implicit-steering to avoid double attenuation."""
    cand_path = REPO_ROOT / "data" / "custom-models" / "candidates.json"
    data = json.loads(cand_path.read_text(encoding="utf-8"))
    uncensored_cands = [c for c in data["candidates"] if c.get("uncensored") is True]
    assert len(uncensored_cands) >= 2
    for c in uncensored_cands:
        # A derivative is self-contained; runtime steering is opt-in, never implicit
        assert c.get("implicit_steering", False) is False
