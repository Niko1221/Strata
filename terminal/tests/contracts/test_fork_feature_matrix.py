"""tests/contracts/test_fork_feature_matrix.py - Tests for feature compatibility matrix across backends."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_INTAKE = REPO_ROOT / "_docs" / "upstream-intake"


def test_feature_matrix_integrity() -> None:
    f = DOCS_INTAKE / "feature-compatibility.json"
    assert f.exists(), "feature-compatibility.json must exist"
    data = json.loads(f.read_text(encoding="utf-8"))
    matrix = data.get("feature_matrix", [])
    assert len(matrix) >= 4
    backends = {item["backend_id"]: item for item in matrix}

    # strata-native keeps Spin8 and Triality
    assert backends["strata-native"]["supports_spin8"] is True
    assert backends["strata-native"]["supports_triality"] is True

    # strata-dsv41 uses native MLA, does not force Spin8
    assert backends["strata-dsv41"]["supports_spin8"] is False
    assert backends["strata-dsv41"]["supports_mla"] is True

    # prism supports PQ2_0
    assert backends["prism"]["supports_pq2_0"] is True


def test_dsv41_no_unproven_gqa_substitution() -> None:
    """DeepSeek compressed/latent attention is not silently substituted with Qwen GQA."""
    def resolve_attention_mode(backend_id: str, arch: str) -> str:
        if backend_id == "strata-dsv41" and arch == "dsv41":
            return "native_mla"
        if backend_id == "strata-native" and arch == "qwen2":
            return "gqa_spin8_triality"
        raise ValueError(f"Incompatible attention pairing: {backend_id} with {arch}")

    assert resolve_attention_mode("strata-dsv41", "dsv41") == "native_mla"
    assert resolve_attention_mode("strata-native", "qwen2") == "gqa_spin8_triality"
    with pytest.raises(ValueError, match="Incompatible attention pairing"):
        resolve_attention_mode("strata-native", "dsv41")
