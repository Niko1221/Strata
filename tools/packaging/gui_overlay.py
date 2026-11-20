"""Build and apply a pinned Windows GUI and generic GGUF backend overlay.

The private JSON spec has ``files`` entries with absolute ``source``, pinned
``sha256`` and package-relative ``path``. Its ``public`` object has a 40-hex
``source_commit``, one direct-child ``generic_folder`` name and ``preimages``:
each host file, the exact ``control/strata_control/runtime.py`` when shipped,
and ``generic-gguf/active.json`` maps to its expected old
{sha256, bytes}, or null if it must not exist. Never put private paths in public.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

import binary_deps
from native_preview import check_source, no_reparse, relative_name, sha256

MANIFEST = "gui-overlay.json"
RESERVE_BUILD = 10 * 1024**3
RESERVE_INSTALL = 6 * 1024**3
POINTER = "generic-gguf/active.json"


def _name(value: str) -> str:
    return relative_name(value)


def _generic_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_+.-]+", value) or value in (".", ".."):
        raise ValueError("unsafe generic folder")
    return value


def _install_name(name: str) -> bool:
    return (name.startswith("host/server/") or name.startswith("host/dist/")
            or name == "control/strata_control/runtime.py")


def _validate_public(public: dict, names: set[str]) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", public["source_commit"]):
        raise ValueError("source_commit must be a full commit SHA")
    folder = _generic_name(public["generic_folder"])
    if not f"generic-gguf/{folder}/llama-server.exe" in names:
        raise ValueError("generic backend executable missing")
    pre = public["preimages"]
    required = {n for n in names if _install_name(n)} | {POINTER}
    if set(pre) != required:
        raise ValueError("preimages must cover each host file and active pointer")
    for name, item in pre.items():
        _name(name)
        if item is not None and (not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) or not isinstance(item["bytes"], int) or item["bytes"] < 0):
            raise ValueError(f"invalid preimage: {name}")
    encoded = json.dumps(public)
    if re.search(r"[A-Za-z]:[/\\]|/Users/|/home/", encoded):
        raise ValueError("private path in public metadata")


def audit_generic(folder: Path) -> dict[str, list[str]]:
    """Audit every PE and its imported DLLs, without requiring strata.exe."""
    no_reparse(folder)
    graph = {}
    for path in sorted(folder.iterdir()):
        no_reparse(path)
        if path.is_dir() or path.suffix.lower() not in (".exe", ".dll"):
            continue
        if binary_deps._match(path.name, binary_deps.WINDOWS_OS + binary_deps.WINDOWS_DRIVER, True):
            raise ValueError("OS or GPU driver DLL must not be bundled")
        graph[path.name] = binary_deps.pe_imports(path)
    if "llama-server.exe" not in graph:
        raise ValueError("llama-server.exe is required")
    missing = binary_deps.uncovered(folder)
    if missing:
        raise ValueError(f"uncovered generic backend imports: {missing}")
    return graph


def verify(stage: Path) -> dict:
    """Require exact membership, metadata, PE graph and hashes."""
    no_reparse(stage)
    manifest_path = stage / MANIFEST
    no_reparse(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "strata-gui-overlay-v1":
        raise ValueError("unsupported GUI overlay schema")
    names: set[str] = set()
    for item in manifest["files"]:
        name = _name(item["path"])
        if name.lower() in {n.lower() for n in names} or name == MANIFEST:
            raise ValueError("duplicate or reserved package path")
        names.add(name)
        path = stage / name
        check_source(path, item["sha256"])
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"size mismatch: {name}")
    _validate_public(manifest["public"], names)
    generic = f"generic-gguf/{manifest['public']['generic_folder']}/"
    if any(not (n.startswith(generic) or _install_name(n) or n.startswith("licenses/") or n in ("README.md", "installer/gui_overlay.py", "installer/native_preview.py", "installer/binary_deps.py")) for n in names):
        raise ValueError("unexpected overlay path")
    actual = set()
    for path in stage.rglob("*"):
        no_reparse(path)
        if path.is_file() and path != manifest_path:
            actual.add(path.relative_to(stage).as_posix())
    if actual != names:
        raise ValueError("overlay membership differs from manifest")
    if audit_generic(stage / generic) != manifest["imports"]:
        raise ValueError("generic backend dependency graph differs")
    return manifest


def build(spec: dict, stage: Path, archive: Path, seven_zip: Path) -> dict:
    """Create a fresh stage and tar.7z, preserving failed outputs for inspection."""
    stage, archive, seven_zip = Path(stage), Path(archive), Path(seven_zip)
    tar_path = archive.with_suffix("")
    for path in (stage, archive, tar_path):
        no_reparse(path)
        if path.exists() or not path.parent.is_dir():
            raise ValueError(f"output must be absent under existing directory: {path}")
    if not archive.name.endswith(".tar.7z") or stage.parent.resolve() != archive.parent.resolve():
        raise ValueError("stage and .tar.7z archive must share one directory")
    items = list(spec["files"])
    helpers = Path(__file__).parent
    for helper in ("gui_overlay.py", "native_preview.py", "binary_deps.py"):
        source = helpers / helper
        items.append({"source": str(source), "sha256": sha256(source), "path": f"installer/{helper}"})
    names: set[str] = set()
    total = 0
    for item in items:
        name = _name(item["path"])
        if name.lower() in names or name == MANIFEST:
            raise ValueError("duplicate or reserved package path")
        names.add(name.lower())
        source = Path(item["source"])
        check_source(source, item["sha256"])
        total += source.stat().st_size
    _validate_public(spec["public"], {item["path"] for item in items})
    peak = total * 4 + 256 * 1024**2
    free = shutil.disk_usage(stage.parent).free
    if free < peak + RESERVE_BUILD:
        raise ValueError("insufficient build volume space plus 10 GiB reserve")
    stage.mkdir()
    files = []
    for item in items:
        source = Path(item["source"])
        target = stage / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, 1024**2)
        check_source(target, item["sha256"])
        files.append({"path": item["path"], "sha256": item["sha256"], "bytes": target.stat().st_size})
    folder = spec["public"]["generic_folder"]
    manifest = {"schema": "strata-gui-overlay-v1", "public": spec["public"], "files": files,
                "imports": audit_generic(stage / "generic-gguf" / folder)}
    (stage / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    verify(stage)
    with tarfile.open(tar_path, "x", format=tarfile.PAX_FORMAT) as output:
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            info = output.gettarinfo(str(path), arcname=f"{stage.name}/{path.relative_to(stage).as_posix()}")
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with path.open("rb") as stream:
                output.addfile(info, stream)
    subprocess.run([str(seven_zip), "a", "-t7z", "-mx=1", "-mmt=1", "-mtc=off", "-mta=off", "-mtm=off", str(archive), str(tar_path)], check=True, timeout=900)
    subprocess.run([str(seven_zip), "t", str(archive)], check=True, timeout=300)
    verify(stage)
    return {"archive": archive.name, "sha256": sha256(archive), "bytes": archive.stat().st_size,
            "tar_sha256": sha256(tar_path), "payload_bytes": total, "free_bytes_before": free}


def _check_preimage(path: Path, expected: dict | None) -> int:
    no_reparse(path)
    if expected is None:
        if path.exists():
            raise ValueError(f"unexpected existing target: {path}")
        return 0
    if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
        raise ValueError(f"installed preimage differs: {path}")
    return expected["bytes"]


def _atomic_copy(source: Path, target: Path, expected: str, link_source: Path | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    no_reparse(target)
    fd, temp_name = tempfile.mkstemp(prefix=".gui-overlay-", dir=target.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        if link_source is not None:
            if link_source.stat().st_dev != target.parent.stat().st_dev or sha256(link_source) != expected:
                raise ValueError("runtime hardlink source changed")
            temp.unlink()
            os.link(link_source, temp)
        else:
            shutil.copyfile(source, temp)
        if sha256(temp) != expected:
            raise ValueError("copy/link hash drift")
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def _runtime_link(runtime_root: Path | None, name: str, expected: str) -> Path | None:
    if runtime_root is None or not re.match(r"(?i)^(cudart|cublas|cublaslt|vcruntime|msvcp|concrt|vcomp).*\.dll$", name):
        return None
    candidate = runtime_root / name
    no_reparse(candidate)
    return candidate if candidate.is_file() and sha256(candidate) == expected else None


def apply(stage: Path, programs: Path, components: Path, backup: Path,
          expected_manifest_sha256: str, runtime_root: Path | None = None) -> dict:
    """Install an overlay; roll back owned host/pointer bytes after a late failure."""
    stage, programs, components, backup = map(Path, (stage, programs, components, backup))
    runtime_root = Path(runtime_root) if runtime_root else None
    for root in (stage, programs, components, backup, runtime_root):
        if root is not None:
            no_reparse(root)
    no_reparse(stage / MANIFEST)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) or sha256(stage / MANIFEST) != expected_manifest_sha256:
        raise ValueError("manifest SHA-256 differs")
    manifest = verify(stage)
    if backup.exists() or not backup.parent.is_dir() or not programs.is_dir() or not components.is_dir():
        raise ValueError("targets must exist and backup must be absent")
    folder = manifest["public"]["generic_folder"]
    generic_root = components / "generic-gguf"
    new_generic = generic_root / folder
    pointer = generic_root / "active.json"
    failed_generic = generic_root / f"{folder}.failed"
    for path in (new_generic, failed_generic, pointer):
        no_reparse(path)
    if new_generic.exists() or failed_generic.exists():
        raise ValueError("generic folder or failure quarantine already exists")
    files = {i["path"]: i for i in manifest["files"]}
    host = [n for n in files if _install_name(n)]
    preimages = manifest["public"]["preimages"]
    targets = {n: programs / n for n in host}
    targets[POINTER] = pointer
    backup_bytes = sum(_check_preimage(targets[n], preimages[n]) for n in targets)
    generic_items = [n for n in files if n.startswith(f"generic-gguf/{folder}/")]
    physical = sum(files[n]["bytes"] for n in host + generic_items) + len(json.dumps({"folder": folder}).encode())
    planned_links: dict[str, Path] = {}
    if runtime_root:
        no_reparse(runtime_root)
        for name in generic_items:
            link = _runtime_link(runtime_root, Path(name).name, files[name]["sha256"])
            if link and link.stat().st_dev == components.stat().st_dev:
                physical -= files[name]["bytes"]
                planned_links[name] = link
    if backup.parent.stat().st_dev == programs.stat().st_dev or backup.parent.stat().st_dev == components.stat().st_dev:
        physical += backup_bytes
    c_free = min(shutil.disk_usage(programs).free, shutil.disk_usage(components).free)
    h_free = shutil.disk_usage(backup.parent).free
    if c_free < physical + RESERVE_INSTALL or h_free < backup_bytes + RESERVE_BUILD:
        raise ValueError("insufficient install or rollback disk reserve")
    backup.mkdir()
    (backup / "plan.json").write_text(json.dumps({
        "manifest_sha256": expected_manifest_sha256,
        "stage": str(stage.resolve()), "programs": str(programs.resolve()),
        "components": str(components.resolve()), "backup": str(backup.resolve()),
        "generic_folder": folder, "preimages": preimages,
        "planned_runtime_links": {name: str(path.resolve()) for name, path in planned_links.items()},
        "admitted_install_bytes": physical, "admitted_backup_bytes": backup_bytes,
        "install_free_bytes": c_free, "backup_free_bytes": h_free,
    }, indent=2) + "\n", encoding="utf-8")
    written: dict[str, str] = {}
    rollback_errors: list[str] = []
    try:
        for name, expected in preimages.items():
            if expected is not None:
                target = backup / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(targets[name], target)
                check_source(target, expected["sha256"])
        for name in generic_items:
            item = files[name]
            target = components / name
            source = stage / name
            check_source(source, item["sha256"])
            _atomic_copy(source, target, item["sha256"], planned_links.get(name))
        for name in sorted(host, key=lambda n: (n.endswith("/index.html"), n)):
            item = files[name]
            check_source(stage / name, item["sha256"])
            _check_preimage(targets[name], preimages[name])
            written[name] = item["sha256"]
            _atomic_copy(stage / name, targets[name], item["sha256"])
        for name in generic_items + host:
            target = components / name if name in generic_items else targets[name]
            check_source(target, files[name]["sha256"])
        pointer_bytes = (json.dumps({"folder": folder}, separators=(",", ":")) + "\n").encode()
        _check_preimage(pointer, preimages[POINTER])
        with tempfile.NamedTemporaryFile(dir=generic_root, prefix=".gui-pointer-", delete=False) as stream:
            temp_pointer = Path(stream.name)
            stream.write(pointer_bytes)
        try:
            written[POINTER] = hashlib.sha256(pointer_bytes).hexdigest()
            os.replace(temp_pointer, pointer)
        finally:
            if temp_pointer.exists():
                temp_pointer.unlink()
        check_source(pointer, written[POINTER])
        receipt = {"status": "success", "manifest_sha256": expected_manifest_sha256,
                "source_commit": manifest["public"]["source_commit"],
                "generic_folder": folder,
                "installed_files": len(host) + len(generic_items), "runtime_links_possible": runtime_root is not None,
                "inference_tested": False, "backup": str(backup)}
        (backup / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt
    except Exception as error:
        for name in reversed(list(written)):
            target = targets[name]
            try:
                old = preimages[name]
                if old is None and not target.exists():
                    continue
                if old is not None and target.is_file() and target.stat().st_size == old["bytes"] and sha256(target) == old["sha256"]:
                    continue
                if not target.is_file() or sha256(target) != written[name]:
                    raise ValueError("installed file changed after publication")
                if old is not None:
                    _atomic_copy(backup / name, target, old["sha256"])
                elif name == POINTER:
                    target.unlink()
                # Keep new inactive host helpers and assets for inspection.
            except Exception as rollback_error:
                rollback_errors.append(f"{name}: {rollback_error}")
        try:
            if new_generic.exists():
                for path in (generic_root, new_generic, failed_generic):
                    no_reparse(path)
                if (generic_root.resolve().parent != components.resolve()
                        or new_generic.resolve().parent != generic_root.resolve()
                        or failed_generic.resolve().parent != generic_root.resolve()):
                    raise ValueError("generic quarantine leaves component root")
                os.replace(new_generic, failed_generic)
        except Exception as rollback_error:
            rollback_errors.append(f"generic quarantine: {rollback_error}")
        retained = []
        for name in host:
            if preimages[name] is None:
                try:
                    if targets[name].is_file() and sha256(targets[name]) == files[name]["sha256"]:
                        retained.append(name)
                except OSError as inspect_error:
                    rollback_errors.append(f"retained file inspection {name}: {inspect_error}")
        failure_receipt = {"status": "failed", "manifest_sha256": expected_manifest_sha256,
                           "error": str(error), "rollback_errors": rollback_errors,
                           "quarantined_generic": str(failed_generic) if failed_generic.is_dir() else None,
                           "retained_new_host_files": retained, "backup": str(backup),
                           "inference_tested": False}
        try:
            (backup / "receipt.json").write_text(json.dumps(failure_receipt, indent=2) + "\n", encoding="utf-8")
        except Exception as receipt_error:
            rollback_errors.append(f"receipt: {receipt_error}")
        raise RuntimeError(f"GUI overlay apply failed: {error}; rollback errors: {rollback_errors}; backup: {backup}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    for flag in ("spec", "stage", "archive", "seven-zip"):
        b.add_argument(f"--{flag}", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--stage", required=True)
    a = sub.add_parser("apply")
    for flag in ("stage", "programs", "components", "backup", "manifest-sha256"):
        a.add_argument(f"--{flag}", required=True)
    a.add_argument("--runtime-root")
    args = parser.parse_args()
    if args.command == "build":
        result = build(json.loads(Path(args.spec).read_text(encoding="utf-8-sig")), Path(args.stage), Path(args.archive), Path(args.seven_zip))
    elif args.command == "verify":
        result = {"files": len(verify(Path(args.stage))["files"])}
    else:
        result = apply(Path(args.stage), Path(args.programs), Path(args.components), Path(args.backup), args.manifest_sha256, Path(args.runtime_root) if args.runtime_root else None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
