"""tests/backends/test_external_backend_contracts.py - Tests for external backend adapters."""
from __future__ import annotations

from pathlib import Path
import pytest

from strata_control.bundle_resolver import ImportSelection, resolve_bundle
from strata_control.bundle_identity import verify_bundle
from strata_control.backends.llama_cpp import build_llama_cpp_plan
from strata_control.backends.prism import build_prism_plan


def test_prism_pq2_routes_only_to_prism(tmp_path: Path) -> None:
    """PQ2_0 dialect models build a valid Prism plan."""
    f = tmp_path / "bonsai_27b.gguf"
    f.write_bytes(b"bonsai-pq2-data")

    bundle = verify_bundle(resolve_bundle(ImportSelection(primary_path=f, dialect="pq2_0")))
    plan = build_prism_plan(bundle)
    assert plan.backend_id == "prism"
    assert "--dialect" in plan.args
    assert "pq2_0" in plan.args


def test_llama_cpp_no_false_deepseek_claim(tmp_path: Path) -> None:
    """llama-cpp rejects DeepSeek MoE bundles rather than falsely claiming compatibility."""
    f = tmp_path / "deepseek.gguf"
    f.write_bytes(b"data")

    bundle = verify_bundle(resolve_bundle(ImportSelection(primary_path=f, architecture="dsv41")))
    with pytest.raises(ValueError, match="does not support DeepSeek MoE"):
        build_llama_cpp_plan(bundle)


def test_supported_dense_gguf(tmp_path: Path) -> None:
    """A standard dense architecture builds a clean loopback launch plan."""
    f = tmp_path / "mistral.gguf"
    f.write_bytes(b"data")

    bundle = verify_bundle(resolve_bundle(ImportSelection(primary_path=f, architecture="mistral")))
    plan = build_llama_cpp_plan(bundle, port=18505)
    assert plan.backend_id == "llama-cpp"
    assert "--host" in plan.args
    assert "127.0.0.1" in plan.args
