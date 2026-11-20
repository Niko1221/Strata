"""CPU checks for the exact fresh Windows package and installer contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packaging"))
import full_windows as full


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    native = f"{full.product_version()}-windows-x86_64-cuda-cuda13"
    generic = full.product_version()
    component = {"id": "strata-engine", "entrypoint": "strata.exe", "files": [
        {"path": "strata.exe", "sha256": _sha(b"native PE"), "bytes": 9},
    ]}
    content = {
        "programs/strata-desktop.exe": b"shell PE",
        "programs/host/server/index.mjs": b"server",
        "programs/host/dist/index.html": b"<html></html>",
        "programs/host/package.json": b"{}",
        "programs/host/hakua-chatbot.enabled": b"1\n",
        "programs/node/node.exe": b"node PE",
        "programs/control/BUNDLE.json": b"{}",
        "programs/control/python/python.exe": b"python PE",
        "programs/control/strata.py": b"print('ok')\n",
        "programs/control/metal/tools/metal_repack.py": b"# Metal repacker\n",
        "programs/control/metal/tools/stream_experts.py": b"# Metal expert streaming\n",
        "programs/harness/BUNDLE.json": b"{}",
        "programs/harness/lib/bin.js": b"harness",
        "programs/README.md": b"Fresh Windows test fixture.\n",
        "programs/licenses/TEST-NOTICE.txt": b"Test notice\n",
        f"components/engine/{native}/engine-component.json": json.dumps(component).encode(),
        f"components/engine/{native}/strata.exe": b"native PE",
        f"components/generic-gguf/{generic}/llama-server.exe": b"generic PE",
        full.WEBVIEW2: b"offline WebView2 PE",
        "README.md": b"Fresh Windows test fixture.\n",
    }
    sources = tmp_path / "sources"
    sources.mkdir()
    files = []
    for name, data in content.items():
        source = sources / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(data)
        files.append({"path": name, "source": str(source), "sha256": _sha(data), "bytes": len(data)})
    spec = {"public": {"source_commit": "a" * 40, "native_folder": native, "generic_folder": generic,
                       "programs_folder": full.programs_folder()}, "files": files}
    return spec, tmp_path / "stage", tmp_path / "package.zip"


@pytest.fixture
def cheap_disk():
    with patch.object(full.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)), \
         patch.object(full, "WEBVIEW2_SHA256", _sha(b"offline WebView2 PE")):
        yield


@pytest.fixture
def install_tmp(tmp_path: Path):
    h_root = Path("H:/Strata/releases") / full.product_version() / "full-package/test-temp"
    if sys.platform != "win32" or not h_root.parent.is_dir():
        yield tmp_path
        return
    h_root.mkdir(exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix="fresh-install-test-", dir=h_root))
    try:
        yield target
    finally:
        assert target.resolve().is_relative_to(h_root.resolve())
        shutil.rmtree(target)


def _build(spec, stage, archive):
    return full.build(spec, stage, archive, audit_native=lambda _: {}, audit_generic=lambda _: {}, audit_other=lambda _: {})


def test_build_verify_zip_and_strict_membership(tmp_path: Path, cheap_disk):
    spec, stage, archive = _fixture(tmp_path)
    receipt = _build(spec, stage, archive)
    assert receipt["files"] == len(spec["files"])
    assert full.verify(stage, audit_native=lambda _: {}, audit_generic=lambda _: {}, audit_other=lambda _: {})["schema"] == full.SCHEMA
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive) as z:
        z.extractall(extracted)
    assert full.verify(extracted, audit_native=lambda _: {}, audit_generic=lambda _: {}, audit_other=lambda _: {})["files"]
    assert receipt["manifest_sha256"] in (extracted / full.STARTER).read_text()
    (extracted / "unlisted.txt").write_text("drift")
    with pytest.raises(ValueError, match="membership"):
        full.verify(extracted, audit_native=lambda _: {}, audit_generic=lambda _: {}, audit_other=lambda _: {})


def test_refuse_source_drift_before_publication(tmp_path: Path, cheap_disk):
    spec, stage, archive = _fixture(tmp_path)
    Path(spec["files"][0]["source"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="source hash mismatch"):
        _build(spec, stage, archive)
    assert not archive.exists()


def test_auxiliary_pe_does_not_accept_unrelated_subdirectory_dll(tmp_path: Path):
    paths = ["programs/strata-desktop.exe", "programs/node/node.exe",
             "programs/control/python/python.exe", "programs/harness/one/addon.node",
             "programs/harness/two/unrelated.dll"]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake PE")
    with patch.object(full.binary_deps, "pe_imports", side_effect=lambda path: ["unrelated.dll"] if path.name == "addon.node" else []):
        with pytest.raises(ValueError, match="uncovered harness PE imports"):
            full.audit_auxiliary(tmp_path)


@pytest.mark.parametrize("name", [
    "programs/control/data/private.gguf", "programs/control/strata-stall.dmp",
    "programs/control/python/python.pdb", "programs/control/__pycache__/x.pyc",
    "programs/host/dist/brand/logo.svg", "programs/host/dist/fonts/GTWalsheimPro.woff2",
    "programs/harness/.env", "programs/harness/examples/prompt.txt",
    "programs/harness/node_modules/.npmrc", "programs/control/python/private.key",
    "programs/control/tools/update_install.ps1",
    "programs/control/metal/tools/other.py",
    "programs/control/metal/metal_repack.py",
    "programs/control/metal/tools/nested/stream_experts.py",
    "programs/control/data/experimental-speed-projection/meta.json",
])
def test_refuse_private_or_unlicensed_payload(name: str, tmp_path: Path):
    spec, _, _ = _fixture(tmp_path)
    spec["files"].append({"path": name, "source": "unused", "sha256": "0" * 64, "bytes": 1})
    with pytest.raises(ValueError, match="forbidden|unapproved|unlicensed|experimental"):
        full._manifest({"public": {**spec["public"], "installer_sha256": "a" * 64}, "files": spec["files"]})


@pytest.mark.skipif(sys.platform != "win32" or shutil.which("powershell.exe") is None, reason="Windows PowerShell required")
def test_stock_powershell_fresh_install_and_existing_target_refusal(install_tmp: Path, cheap_disk):
    tmp_path = install_tmp
    spec, stage, archive = _fixture(tmp_path)
    receipt = _build(spec, stage, archive)
    programs = tmp_path / "fresh" / "Programs" / "Strata"
    components = tmp_path / "fresh" / "components"
    shortcuts = tmp_path / "fresh" / "shortcuts"
    args = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(stage / full.INSTALLER),
            "-ExpectedManifestSha256", receipt["manifest_sha256"], "-ProgramsRoot", str(programs),
            "-ComponentsRoot", str(components), "-ShortcutRoot", str(shortcuts)]
    done = subprocess.run(args, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert (programs / "strata-desktop.exe").read_bytes() == b"shell PE"
    assert (components / "engine" / spec["public"]["native_folder"] / "strata.exe").read_bytes() == b"native PE"
    assert json.loads((components / "generic-gguf" / "active.json").read_text())["folder"] == spec["public"]["generic_folder"]
    receipts = list(components.glob("install-full-*.json"))
    assert len(receipts) == 1 and json.loads(receipts[0].read_text())["status"] == "success"
    assert (shortcuts / "Strata Preview 0.3.2 (SM120).lnk").is_file()
    second = subprocess.run(args, capture_output=True, text=True, timeout=30)
    assert second.returncode != 0 and "Target already exists" in second.stderr


@pytest.mark.skipif(sys.platform != "win32" or shutil.which("powershell.exe") is None, reason="Windows PowerShell required")
def test_concurrent_foreign_target_is_never_quarantined(install_tmp: Path, cheap_disk):
    tmp_path = install_tmp
    spec, stage, archive = _fixture(tmp_path)
    bulk = tmp_path / "bulk-source.bin"
    bulk.write_bytes(b"x" * (64 * 1024 * 1024))
    spec["files"].append({"path": "programs/host/dist/bulk.bin", "source": str(bulk),
                          "sha256": full.sha256(bulk), "bytes": bulk.stat().st_size})
    receipt = _build(spec, stage, archive)
    programs = tmp_path / "race" / "Programs" / "Strata"
    components = tmp_path / "race" / "components"
    args = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(stage / full.INSTALLER),
            "-ExpectedManifestSha256", receipt["manifest_sha256"], "-ProgramsRoot", str(programs),
            "-ComponentsRoot", str(components)]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 60
    marker = None
    while time.monotonic() < deadline and process.poll() is None:
        stages = list(programs.parent.glob("Strata.install-*/.strata-install-owner-*")) if programs.parent.exists() else []
        if stages:
            marker = stages[0]
            break
        time.sleep(0.005)
    if marker is None:
        out, err = process.communicate(timeout=30)
        pytest.fail(f"installer did not enter its owned staging phase: {out} {err}")
    programs.mkdir(parents=True)
    foreign = programs / "foreign-user-file.txt"
    foreign.write_text("keep this foreign target", encoding="utf-8")
    out, err = process.communicate(timeout=120)
    assert process.returncode != 0, out
    assert "Target already exists" in err
    assert foreign.read_text(encoding="utf-8") == "keep this foreign target"
    assert not list(programs.parent.glob("Strata.failed-*"))
