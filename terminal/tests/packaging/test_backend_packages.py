# tests/packaging/test_backend_packages.py
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_packaging_documentation_present():
    """Verify engine component and packaging update documentation exist."""
    engine_comp = REPO_ROOT / "packaging" / "ENGINE-COMPONENT.md"
    updates_doc = REPO_ROOT / "packaging" / "UPDATES.md"
    assert engine_comp.exists(), "packaging/ENGINE-COMPONENT.md must exist"
    assert updates_doc.exists(), "packaging/UPDATES.md must exist"


def test_windows_packaging_scripts_present():
    """Verify build scripts for Windows backends exist."""
    dsv41_win = REPO_ROOT / "packaging" / "build_dsv41_windows.bat"
    vision_win = REPO_ROOT / "packaging" / "build_vision_windows.bat"
    assert dsv41_win.exists(), "packaging/build_dsv41_windows.bat must exist"
    assert vision_win.exists(), "packaging/build_vision_windows.bat must exist"


def test_backend_registry_capabilities_match_packages():
    """Ensure backend capabilities schema and registry account for all packageable backends."""
    from strata_control.backend_registry import create_default_registry

    reg = create_default_registry()
    registered = reg.list_backends()
    expected_backends = ["strata-dsv41", "strata-native", "prism", "llama-cpp"]

    for b in expected_backends:
        assert b in registered, f"Backend {b} must be registered in BackendRegistry"
        caps = reg.get(b)
        assert caps is not None
        assert caps.backend_id == b
        assert len(caps.supported_architectures) > 0


def test_model_bundle_schema_valid():
    """Verify model-bundle and fork-extension schemas are valid JSON."""
    mb_schema = REPO_ROOT / "schemas" / "model-bundle.schema.json"
    fe_schema = REPO_ROOT / "schemas" / "fork-extension.schema.json"
    bc_schema = REPO_ROOT / "schemas" / "backend-capabilities.schema.json"

    assert mb_schema.exists()
    assert fe_schema.exists()
    assert bc_schema.exists()

    data_mb = json.loads(mb_schema.read_text(encoding="utf-8"))
    data_fe = json.loads(fe_schema.read_text(encoding="utf-8"))
    data_bc = json.loads(bc_schema.read_text(encoding="utf-8"))

    assert data_mb.get("type") == "object"
    assert data_fe.get("type") == "object"
    assert data_bc.get("type") == "object"


def test_upstream_cut_verifier_passes():
    """Ensure tools/check_upstream_cut.py passes programmatically."""
    from tools.check_upstream_cut import verify_upstream_cut

    report = verify_upstream_cut()
    assert report["all_passed"] is True, f"Upstream cut verification failed: {report}"
    assert report["status"] == "LOCKED"
