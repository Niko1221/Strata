"""Small fail-closed filesystem and production-verifier adapters; no import side effects."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
from typing import Any, Callable
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = Path("H:/Strata/releases/local-053-20261109-opencode")
NAME = "0.5.3-windows-x86_64-cuda-cuda13"
NATIVE_COMMIT = "ded724a4029cbfcf48d58821f01f88fe246c7460"
NATIVE_ROOT = Path("H:/Strata/worktrees/native-053-source-20261109")
NATIVE = BASE / "native-component" / NAME
REFERENCE = Path("H:/Strata/releases/personal-052-local2-20261107/app/signed-component-r4/0.5.2-windows-x86_64-cuda-cuda13")
RESOURCES = ROOT / "apps/desktop/src-tauri/resources"
WORK = BASE / "app/component-053"
STAGING = ROOT / "_docs/local-archive/2026-11-09-pre-main/component-staging"
RECEIPT = BASE / "app/staged-native.json"
PYTHON = Path("C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe")
NODE = Path("C:/Users/downl/AppData/Local/Programs/Strata/node/node.exe")
PRISM = "9c729c73eb050dc466203fe149a27acd5be48763a33ea2c5d2b89f78e4ddcbaa"
PRISM_EXE = "5ca5d8bfdfcb6fe46a96a06b71eaf45aedfbd1c6673bda2a10eb6412613be777"
TURBO = "85a6fea57bf766f7ebbb98ff62ee0ff8c83330f99b770e68031f662dae93ae14"
MANIFEST = "engine-component.json"


class Blocked(RuntimeError):
    """An intentionally public, non-secret failure description."""


def require(ok: Any, message: str) -> None:
    if not ok:
        raise Blocked(message)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load(path: Path) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON field")
            result[key] = value
        return result
    require(path.stat().st_size <= 16 * 1024**2, "JSON exceeds receipt budget")
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique)


def write_new(path: Path, value: Any) -> None:
    require(path.parent.is_dir(), "Output parent absent")
    no_links(path)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def no_links(path: Path) -> None:
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            require(not part.is_symlink() and not part.is_junction()
                    and not (part.lstat().st_file_attributes & 0x400 if os.name == "nt" else False),
                    "Reparse/linked paths refused")


def relative(name: str) -> str:
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name
            and not PureWindowsPath(name).drive and not name.startswith("/")
            and all(part not in ("", ".", "..") and not part.endswith((" ", "."))
                    for part in name.split("/")), "Unsafe package member")
    return name


def inventory(root: Path, tick: Callable[[], Any] = lambda: None) -> dict[str, str]:
    no_links(root)
    require(root.is_dir(), "Tree absent")
    result: dict[str, str] = {}
    folded: set[str] = set()
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in sorted(dirs + names):
            path = Path(directory) / name
            no_links(path)
            if name in names:
                tick()
                require(path.is_file(), "Non-regular tree member")
                rel = relative(path.relative_to(root).as_posix())
                require(rel.casefold() not in folded, "Case-colliding package member")
                folded.add(rel.casefold())
                result[rel] = sha(path)
    return dict(sorted(result.items()))


def copy_tree(source: Path, target: Path, tick: Callable[[], Any]) -> dict[str, str]:
    require(target.parent.is_dir() and not target.exists(), "Fresh copy destination required")
    before = inventory(source, tick)
    target.mkdir()
    for name in before:
        tick()
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination)
    require(inventory(target, tick) == before == inventory(source, tick), "Copy/source bytes changed")
    return before


def preserve_bundle_metadata(bundle: Path, pending: Path, tick: Callable[[], Any]) -> dict[str, str]:
    """Retain the reviewed root scaffolding so an atomic root swap cannot dirty app source."""
    before = inventory(bundle, tick)
    root_files = {name for name in before if "/" not in name}
    require(root_files == {".gitignore", "README.txt"}, "Unknown managed bundle root metadata; parent review required")
    for name in sorted(root_files):
        destination = pending / name
        require(not destination.exists(), "Pending bundle metadata must be fresh")
        with (bundle / name).open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
        require(sha(destination) == before[name], "Managed root metadata changed")
    require(inventory(bundle, tick) == before, "Prior managed root changed during preparation")
    return before


def verify_payload(root: Path, ec: Any, signing: Any, *, signed: bool) -> tuple[dict, dict]:
    """Use built-in trust only, full production payload verification, plus exact membership."""
    tree = inventory(root)
    component = ec.read_manifest(root)
    manifest = load(root / MANIFEST)
    if signed:
        signing.verify_bytes(signing.load_trusted_keys(environ={}),
                             ec.signing_payload(manifest), manifest.get("signature"))
    else:
        require(manifest.get("signature") is None, "Unsigned native/composite input required")
    ec.verify(component, full=True)
    expected = {relative(row["path"]): row["sha256"] for row in manifest["files"]}
    require(len(expected) == len(manifest["files"]) and
            {k: v for k, v in tree.items() if k != MANIFEST} == expected,
            "Unexpected package membership/payload")
    require(inventory(root) == tree, "Payload changed while verified")
    return manifest, tree


def retain(name: str) -> bool:
    """Only independently versioned backends and complete notices come from the old native package."""
    return name.startswith(("local-backends/prism/", "local-backends/turboquant/", "notices/"))


def compose(native: Path, old: Path, target: Path, tick: Callable[[], Any]) -> dict:
    """Caller must authenticate both inputs first. Start with the new native tree."""
    fresh = copy_tree(native, target, tick)
    require(not any(k.startswith("local-backends/") for k in fresh), "Native-only input required")
    for name, checksum in inventory(old, tick).items():
        if not retain(name):
            continue
        tick()
        if name in fresh:
            require(fresh[name] == checksum, "Conflicting native/retained support file")
            continue
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old / name, destination)
        require(sha(destination) == checksum, "Retained copy changed")
    tree = inventory(target, tick)
    # Control's licensed vector belongs in Control. No model payload belongs here.
    require(not any(Path(k).suffix.lower() in (".gguf", ".q36epk", ".pack", ".safetensors", ".pt", ".pth", ".bin", ".onnx")
                    for k in tree), "Model data in native component")
    manifest = load(target / MANIFEST)
    manifest.update(signature=None, privateCandidate=True, releaseQualified=False,
                    composition={"nativeManifestSHA256": fresh[MANIFEST],
                                 "retainedManifestSHA256": sha(old / MANIFEST),
                                 "runtimeManifestSHA256": {"prism": PRISM, "turboquant": TURBO}})
    manifest["files"] = [{"path": k, "sha256": v, "bytes": (target / k).stat().st_size}
                         for k, v in tree.items() if k != MANIFEST]
    # This is the newly owned unsigned manifest only; fresh native-build.json is never rewritten.
    (target / MANIFEST).unlink()
    write_new(target / MANIFEST, manifest)
    return manifest


def check_archive(archive: Path, entry: dict, tree: dict[str, str]) -> None:
    require(entry["url"] == archive.name and entry["format"] == "zip"
            and entry["size"] == archive.stat().st_size and entry["sha256"] == sha(archive)
            and entry["manifest_sha256"] == tree[MANIFEST], "Archive entry differs")
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        require(len(infos) == len(tree) and len({i.filename.casefold() for i in infos}) == len(infos),
                "ZIP duplicate/membership mismatch")
        require({relative(i.filename) for i in infos} == set(tree), "ZIP must have payload at root")
        require(sum(i.file_size for i in infos) == entry["installed_bytes"], "ZIP installed byte count differs")
        for info in infos:
            require(not info.is_dir() and not info.flag_bits & 1
                    and (info.external_attr >> 16) & 0o170000 != 0o120000, "ZIP link/encryption refused")
            with bundle.open(info) as stream:
                require(hashlib.file_digest(stream, "sha256").hexdigest() == tree[info.filename],
                        "ZIP payload differs")


def production_modules() -> tuple[Any, Any, Any]:
    """Only the reviewed app's public verification/dependency code is loaded."""
    import sys
    sys.path.insert(0, str(ROOT))
    from terminal.strata_control import engine_component, signing
    spec = importlib.util.spec_from_file_location("component_binary_deps", ROOT / "packaging/binary_deps.py")
    require(spec and spec.loader, "Production dependency reader absent")
    deps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deps)
    return engine_component, signing, deps


def check_dependencies(folder: Path, deps: Any) -> None:
    """Native and each backend have independent Windows DLL search roots."""
    for root in (folder, folder / "local-backends/prism", folder / "local-backends/turboquant"):
        names = [p.name for p in root.iterdir() if p.is_file()]
        binaries = [p for p in root.iterdir() if p.suffix.lower() in (".exe", ".dll")]
        require(binaries, "Dependency scope has no binaries")
        for binary in binaries:
            kind, _imports = deps.imports(binary)
            require(kind == "pe", "Malformed/non-PE runtime binary")
        require(not deps.uncovered(root, names), "Uncovered production binary dependencies")
