from pathlib import Path
import pytest

from strata_control.backend_contracts import BackendCapabilities, LoadRequest
from strata_control.backend_probe import probe_backend_binary
from strata_control.backend_registry import BackendRegistry, create_default_registry
from strata_control.bundle_identity import verify_bundle
from strata_control.bundle_resolver import ImportSelection, resolve_bundle


def test_probe_backend_binary_computes_real_hash(tmp_path: Path):
    """Verify that probing an existing binary computes its real SHA-256 hash."""
    fake_binary = tmp_path / "strata-engine.exe"
    fake_binary.write_bytes(b"\x7fELF\x02\x01\x01\x00" + b"\x90" * 1024)

    caps = probe_backend_binary(
        binary_path=fake_binary,
        backend_id="strata-native",
        supported_architectures=("qwen4exp", "qwen2"),
        supported_dialects=("standard",),
    )

    import hashlib
    expected_sha = hashlib.sha256(fake_binary.read_bytes()).hexdigest()
    assert caps.binary_sha256 == expected_sha
    assert caps.binary_sha256 != "native-sha"
    assert caps.caps_hash != ""


def test_probe_missing_binary_fails_strict():
    """Verify that strict probe of non-existent binary raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Backend binary not found"):
        probe_backend_binary(
            binary_path="non/existent/path/binary.exe",
            backend_id="strata-native",
            supported_architectures=("qwen2",),
            supported_dialects=("standard",),
            strict=True,
        )


def test_no_placeholder_hashes_in_probed_registry(tmp_path: Path):
    """Verify that create_default_registry with probed binaries does not produce placeholder -sha hashes."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "strata-native.exe").write_bytes(b"NATIVE_ENGINE_BIN")
    (bin_dir / "strata-dsv41.exe").write_bytes(b"DSV41_ENGINE_BIN")

    reg = create_default_registry(root_dir=tmp_path)
    for bid in reg.list_backends():
        caps = reg.get(bid)
        assert caps is not None
        assert not caps.binary_sha256.endswith("-sha")
