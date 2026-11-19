"""Freeze an explicitly inventoried native Windows preview; never remove evidence.

The private spec contains source paths, SHA-256 pins and public build metadata.
Only relative names, hashes, sizes and the public metadata enter the package.
No dependency discovery downloads files or copies from PATH/System32.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tarfile

import binary_deps

RESERVE = 10 * 1024**3
MANIFEST = "preview-manifest.json"


def sha256(path: Path) -> str:
    """Hash large inputs with bounded memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def relative_name(value: str) -> str:
    """Require an unambiguous Windows-safe relative package name."""
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or any(c in value for c in '\\:<>"|?*\r\n')
        or any(p in (".", "..") or p.endswith((" ", ".")) for p in path.parts)
        or any(
            re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", p.split(".")[0], re.I)
            for p in path.parts
        )
    ):
        raise ValueError(f"unsafe relative path: {value!r}")
    return value


def no_reparse(path: Path) -> None:
    """Reject symlinks/junctions in both existing and prospective paths."""
    path = path.absolute()
    for part in [path, *path.parents]:
        try:
            details = part.lstat()
        except FileNotFoundError:
            continue
        # Python 3.10 has no Path.is_junction(). Windows exposes all reparse
        # points through lstat, including junctions and prospective ancestors.
        reparse = getattr(details, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if stat.S_ISLNK(details.st_mode) or reparse:
            raise ValueError(f"reparse path: {part}")


def check_source(path: Path, expected: str) -> None:
    """Reject missing or changed pinned inputs before publication."""
    no_reparse(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256(path) != expected:
        raise ValueError(f"source hash mismatch: {path.name}")


def audit(folder: Path) -> dict[str, list[str]]:
    """Parse every PE strictly and audit recursive imports including delay loads."""
    graph = {}
    names = {p.name.lower() for p in folder.iterdir() if p.is_file()}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in (".exe", ".dll"):
            continue
        if binary_deps._match(
            path.name, binary_deps.WINDOWS_DRIVER + binary_deps.WINDOWS_OS, True
        ):
            raise ValueError("OS or GPU driver DLL must not be bundled")
        graph[path.name] = binary_deps.pe_imports(path)
    if not graph or "strata.exe" not in graph:
        raise ValueError("native strata.exe is required")
    # cuBLAS may open cuBLASLt dynamically; an import table alone is insufficient.
    if "cublas64_13.dll" in names and "cublaslt64_13.dll" not in names:
        raise ValueError("cuBLAS requires the bundled cuBLASLt runtime")
    missing = binary_deps.uncovered(folder)
    if missing:
        raise ValueError(f"uncovered dependencies: {missing}")
    return graph


def verify(folder: Path) -> dict:
    """Check the complete extracted file set and every payload hash."""
    no_reparse(folder)
    manifest = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    expected = set()
    for item in manifest["files"]:
        name = relative_name(item["path"])
        if name.lower() in expected or name == MANIFEST:
            raise ValueError("duplicate or self-referencing manifest entry")
        expected.add(name.lower())
        path = folder / name
        check_source(path, item["sha256"])
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"size mismatch: {name}")
    actual = set()
    for path in folder.rglob("*"):
        no_reparse(path)
        if path.is_file() and path.relative_to(folder).as_posix() != MANIFEST:
            actual.add(path.relative_to(folder).as_posix().lower())
    if actual != expected:
        raise ValueError("package membership differs from manifest")
    if audit(folder) != manifest["imports"]:
        raise ValueError("dependency graph differs from manifest")
    return manifest


def build(spec: dict, stage: Path, archive: Path, seven_zip: Path) -> dict:
    """Create a fresh stage, real tar then 7z, preserving every failed output."""
    tar_path = archive.with_suffix("")
    for path in (stage, archive, tar_path):
        no_reparse(path)
        if path.exists():
            raise ValueError(f"destination already exists: {path}")
        if not path.parent.is_dir():
            raise ValueError(f"create the output parent first: {path.parent}")
    if not archive.name.endswith(".tar.7z"):
        raise ValueError("archive must end in .tar.7z")
    if stage.parent.resolve() != archive.parent.resolve():
        raise ValueError("stage and archive must share one admitted volume/directory")
    items = spec["files"]
    names = set()
    total = 0
    for item in items:
        name = relative_name(item["path"])
        if name.lower() in names or name.lower() == MANIFEST:
            raise ValueError("duplicate/reserved destination")
        names.add(name.lower())
        source = Path(item["source"])
        check_source(source, item["sha256"])
        total += source.stat().st_size
    public = spec["public"]
    for key in ("source_commit", "package_commit"):
        if not re.fullmatch(r"[0-9a-f]{40}", public[key]):
            raise ValueError(f"invalid {key}")
    encoded = json.dumps(public)
    if re.search(r"[A-Za-z]:[/\\]|/Users/|/home/", encoded):
        raise ValueError("private absolute path in public metadata")
    peak = total * 5 + 256 * 1024**2  # stage, tar, archive, extraction + margin
    free = shutil.disk_usage(stage.parent).free
    if free < peak + RESERVE:
        raise ValueError("insufficient disk budget plus 10 GiB reserve")
    stage.mkdir()
    files = []
    for item in items:
        source, target = Path(item["source"]), stage / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, 1024**2)
        check_source(target, item["sha256"])
        files.append(
            {
                "path": item["path"],
                "sha256": item["sha256"],
                "bytes": target.stat().st_size,
            }
        )
    manifest = {
        "schema": "strata-native-preview-v1",
        **public,
        "files": files,
        "imports": audit(stage),
    }
    (stage / MANIFEST).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    verify(stage)
    # Normalise tar metadata. The compressed bytes also depend on the recorded 7z version.
    with tarfile.open(tar_path, "x", format=tarfile.USTAR_FORMAT) as output:
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            info = output.gettarinfo(
                str(path), arcname=f"{stage.name}/{path.relative_to(stage).as_posix()}"
            )
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with path.open("rb") as stream:
                output.addfile(info, stream)
    subprocess.run(
        [
            str(seven_zip),
            "a",
            "-t7z",
            "-mx=5",
            "-mmt=1",
            "-mtc=off",
            "-mta=off",
            "-mtm=off",
            str(archive),
            str(tar_path),
        ],
        check=True,
        timeout=900,
    )
    subprocess.run([str(seven_zip), "t", str(archive)], check=True, timeout=300)
    # Detect source-stage drift during archiving, not just before it.
    verify(stage)
    return {
        "archive": archive.name,
        "sha256": sha256(archive),
        "bytes": archive.stat().st_size,
        "tar_sha256": sha256(tar_path),
        "payload_bytes": total,
        "admitted_peak_bytes": peak,
        "free_bytes_before": free,
        "reserve_bytes": RESERVE,
    }


def main() -> None:
    """Expose explicit build and independent extracted-package verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--seven-zip", type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify)
        logging.info("verified %d payload files", len(result["files"]))
    elif all((args.spec, args.stage, args.archive, args.seven_zip)):
        result = build(
            json.loads(args.spec.read_text(encoding="utf-8-sig")),
            args.stage,
            args.archive,
            args.seven_zip,
        )
        receipt = args.archive.with_suffix(args.archive.suffix + ".json")
        with receipt.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
        logging.info("archive SHA-256 %s", result["sha256"])
    else:
        parser.error("supply --verify or all build arguments")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
