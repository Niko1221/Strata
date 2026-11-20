"""CPU-only transaction checks for the GUI companion overlay."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packaging"))
import gui_overlay as gui


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path: Path, *, existing: bool = False):
    stage = tmp_path / "stage"
    programs = tmp_path / "programs"
    components = tmp_path / "components"
    backup = tmp_path / "backup"
    for path in (stage, programs, components):
        path.mkdir()
    folder = "0.3.2-preview.20261027.1"
    payload = {
        "control/strata_control/runtime.py": b"new control runtime\n",
        "host/server/app.mjs": b"new server\n",
        "host/dist/index.html": b"new UI\n",
        f"generic-gguf/{folder}/llama-server.exe": b"fake PE server",
        f"generic-gguf/{folder}/vcruntime140.dll": b"matching immutable runtime",
    }
    files = []
    for name, data in payload.items():
        path = stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files.append({"path": name, "sha256": _sha(data), "bytes": len(data)})
    pre = {"control/strata_control/runtime.py": None,
           "host/server/app.mjs": None, "host/dist/index.html": None, gui.POINTER: None}
    if existing:
        old = {"control/strata_control/runtime.py": b"old control runtime\n",
               "host/server/app.mjs": b"old server\n", "host/dist/index.html": b"old UI\n",
               gui.POINTER: b'{"folder":"old"}\n'}
        for name, data in old.items():
            target = (components if name == gui.POINTER else programs) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            pre[name] = {"sha256": _sha(data), "bytes": len(data)}
    manifest = {"schema": "strata-gui-overlay-v1", "public": {
        "source_commit": "a" * 40, "generic_folder": folder, "preimages": pre},
        "files": files, "imports": {"llama-server.exe": [], "vcruntime140.dll": []}}
    (stage / gui.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return stage, programs, components, backup, folder, payload


@pytest.fixture
def cpu_guards():
    with patch.object(gui, "audit_generic", return_value={"llama-server.exe": [], "vcruntime140.dll": []}), \
         patch.object(gui.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)):
        yield


def test_apply_validated_overlay_and_hardlink_runtime(tmp_path: Path, cpu_guards):
    stage, programs, components, backup, folder, payload = _fixture(tmp_path, existing=True)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    source_runtime = runtime / "vcruntime140.dll"
    source_runtime.write_bytes(payload[f"generic-gguf/{folder}/vcruntime140.dll"])
    digest = gui.sha256(stage / gui.MANIFEST)
    receipt = gui.apply(stage, programs, components, backup, digest, runtime)
    assert receipt["inference_tested"] is False
    assert json.loads((backup / "plan.json").read_text())["manifest_sha256"] == digest
    assert json.loads((backup / "receipt.json").read_text())["status"] == "success"
    assert (components / gui.POINTER).read_text() == json.dumps({"folder": folder}, separators=(",", ":")) + "\n"
    assert (programs / "host/dist/index.html").read_bytes() == b"new UI\n"
    assert (programs / "control/strata_control/runtime.py").read_bytes() == b"new control runtime\n"
    assert (backup / "host/dist/index.html").read_bytes() == b"old UI\n"
    linked = components / "generic-gguf" / folder / "vcruntime140.dll"
    assert os.stat(linked).st_ino == os.stat(source_runtime).st_ino


def test_source_drift_rejected_before_mutation(tmp_path: Path, cpu_guards):
    stage, programs, components, backup, folder, _ = _fixture(tmp_path, existing=True)
    digest = gui.sha256(stage / gui.MANIFEST)
    (stage / "host/server/app.mjs").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source hash mismatch"):
        gui.apply(stage, programs, components, backup, digest)
    assert not backup.exists()
    assert not (components / "generic-gguf" / folder).exists()
    assert (programs / "host/server/app.mjs").read_bytes() == b"old server\n"
    assert (programs / "control/strata_control/runtime.py").read_bytes() == b"old control runtime\n"


def test_overlay_rejects_other_control_paths(tmp_path: Path, cpu_guards):
    stage, _, _, _, _, _ = _fixture(tmp_path)
    name = "control/strata_control/other.py"
    path = stage / name
    path.write_bytes(b"unexpected control file")
    manifest_path = stage / gui.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["files"].append({"path": name, "sha256": gui.sha256(path), "bytes": path.stat().st_size})
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unexpected overlay path"):
        gui.verify(stage)


def test_budgeted_runtime_link_failure_does_not_copy(tmp_path: Path, cpu_guards):
    stage, programs, components, backup, folder, payload = _fixture(tmp_path, existing=True)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "vcruntime140.dll").write_bytes(payload[f"generic-gguf/{folder}/vcruntime140.dll"])
    digest = gui.sha256(stage / gui.MANIFEST)
    with patch.object(gui.os, "link", side_effect=OSError("injected hardlink failure")):
        with pytest.raises(RuntimeError, match="injected hardlink failure"):
            gui.apply(stage, programs, components, backup, digest, runtime)
    assert (programs / "host/server/app.mjs").read_bytes() == b"old server\n"
    assert (programs / "control/strata_control/runtime.py").read_bytes() == b"old control runtime\n"
    assert (components / gui.POINTER).read_bytes() == b'{"folder":"old"}\n'
    failed = components / "generic-gguf" / f"{folder}.failed"
    assert failed.is_dir()
    assert not (failed / "vcruntime140.dll").exists()


def test_planned_runtime_link_drift_does_not_change_to_copy(tmp_path: Path, cpu_guards):
    stage, programs, components, backup, folder, payload = _fixture(tmp_path, existing=True)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    runtime_file = runtime / "vcruntime140.dll"
    runtime_file.write_bytes(payload[f"generic-gguf/{folder}/vcruntime140.dll"])
    digest = gui.sha256(stage / gui.MANIFEST)
    real_copy = gui._atomic_copy

    def mutate_after_first_generic(source, target, expected, link_source=None):
        result = real_copy(source, target, expected, link_source)
        if Path(target).name == "llama-server.exe":
            runtime_file.write_bytes(b"changed after disk planning")
        return result

    with patch.object(gui, "_atomic_copy", side_effect=mutate_after_first_generic):
        with pytest.raises(RuntimeError, match="runtime hardlink source changed"):
            gui.apply(stage, programs, components, backup, digest, runtime)
    assert (programs / "host/server/app.mjs").read_bytes() == b"old server\n"
    assert (components / gui.POINTER).read_bytes() == b'{"folder":"old"}\n'
    failed = components / "generic-gguf" / f"{folder}.failed"
    assert failed.is_dir()
    assert not (failed / "vcruntime140.dll").exists()


def test_late_pointer_failure_restores_preimages_and_quarantines_generic(tmp_path: Path, cpu_guards):
    stage, programs, components, backup, folder, _ = _fixture(tmp_path, existing=True)
    digest = gui.sha256(stage / gui.MANIFEST)
    real_replace = gui.os.replace
    failed = False

    def fail_pointer_once(source, target):
        nonlocal failed
        if not failed and Path(target) == components / gui.POINTER:
            failed = True
            raise OSError("injected pointer publication failure")
        return real_replace(source, target)

    with patch.object(gui.os, "replace", side_effect=fail_pointer_once):
        with pytest.raises(RuntimeError, match="injected pointer publication failure"):
            gui.apply(stage, programs, components, backup, digest)
    assert failed
    assert (programs / "host/server/app.mjs").read_bytes() == b"old server\n"
    assert (programs / "host/dist/index.html").read_bytes() == b"old UI\n"
    assert (components / gui.POINTER).read_bytes() == b'{"folder":"old"}\n'
    assert not (components / "generic-gguf" / folder).exists()
    assert (components / "generic-gguf" / f"{folder}.failed").is_dir()
    assert (backup / "host/server/app.mjs").read_bytes() == b"old server\n"
    failure_receipt = json.loads((backup / "receipt.json").read_text())
    assert failure_receipt["status"] == "failed"
    assert failure_receipt["rollback_errors"] == []
    assert failure_receipt["quarantined_generic"] == str(components / "generic-gguf" / f"{folder}.failed")


def test_uncertain_index_publish_restores_only_owned_bytes(tmp_path: Path, cpu_guards):
    stage, programs, components, backup, folder, _ = _fixture(tmp_path, existing=True)
    digest = gui.sha256(stage / gui.MANIFEST)
    real_replace = gui.os.replace
    failed = False

    def publish_then_fail(source, target):
        nonlocal failed
        if not failed and Path(target) == programs / "host/dist/index.html":
            failed = True
            real_replace(source, target)
            raise OSError("injected uncertain index result")
        return real_replace(source, target)

    with patch.object(gui.os, "replace", side_effect=publish_then_fail):
        with pytest.raises(RuntimeError, match="injected uncertain index result"):
            gui.apply(stage, programs, components, backup, digest)
    assert failed
    assert (programs / "host/server/app.mjs").read_bytes() == b"old server\n"
    assert (programs / "host/dist/index.html").read_bytes() == b"old UI\n"
    assert (components / gui.POINTER).read_bytes() == b'{"folder":"old"}\n'
    assert (components / "generic-gguf" / f"{folder}.failed").is_dir()
