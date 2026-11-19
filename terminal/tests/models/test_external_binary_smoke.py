from pathlib import Path
import pytest

from strata_control.backend_contracts import BackendCapabilities
from strata_control.backends.llama_cpp import build_llama_cpp_plan
from strata_control.backends.prism import build_prism_plan
from strata_control.bundle_identity import verify_bundle
from strata_control.bundle_resolver import ImportSelection, resolve_bundle


def test_prism_plan_for_hikari_pq2_0(tmp_path: Path):
    """Verify that Hikari with PQ2_0 dialect produces correct Prism plan."""
    f = tmp_path / "Hikari-27B-PQ2_0.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 32)
    bundle = resolve_bundle(ImportSelection(primary_path=f, architecture="qwen35", dialect="pq2_0"))
    verified = verify_bundle(bundle)

    plan = build_prism_plan(verified, port=18503)
    assert plan.backend_id == "prism"
    assert "--dialect" in plan.args
    assert "pq2_0" in plan.args
    assert "--port" in plan.args
    assert "18503" in plan.args


def test_prism_plan_rejects_unsupported_dialect(tmp_path: Path):
    """Verify that Prism plan rejects incompatible dialect."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 32)
    bundle = resolve_bundle(ImportSelection(primary_path=f, architecture="qwen2", dialect="tq4_1s"))
    verified = verify_bundle(bundle)

    with pytest.raises(ValueError, match="Prism backend does not support dialect"):
        build_prism_plan(verified)


def test_llama_cpp_plan_rejects_deepseek41(tmp_path: Path):
    """Verify that llama-cpp plan rejects deepseek41 architecture."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 32)
    bundle = resolve_bundle(ImportSelection(primary_path=f, architecture="deepseek41"))
    verified = verify_bundle(bundle)

    with pytest.raises(ValueError, match="does not support DeepSeek"):
        build_llama_cpp_plan(verified)


def test_binary_verification_with_caps(tmp_path: Path):
    """Verify that plan verifies binary presence when BackendCapabilities are supplied."""
    bin_p = tmp_path / "prism-server.exe"
    bin_p.write_bytes(b"PRISM_SERVER_BINARY")

    import hashlib
    bin_sha = hashlib.sha256(b"PRISM_SERVER_BINARY").hexdigest()

    caps = BackendCapabilities(
        schema_version="1.0",
        backend_id="prism",
        binary_path=str(bin_p),
        binary_sha256=bin_sha,
        supported_architectures=("qwen2", "qwen35"),
        supported_dialects=("pq2_0",),
    )

    f = tmp_path / "Hikari-27B-PQ2_0.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 32)
    bundle = resolve_bundle(ImportSelection(primary_path=f, architecture="qwen35", dialect="pq2_0"))
    verified = verify_bundle(bundle)

    plan = build_prism_plan(verified, caps=caps)
    assert plan.binary_path == str(bin_p)
    assert plan.verified_sha256 == bin_sha
