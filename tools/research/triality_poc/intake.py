"""Verify the supplied paper archive and extract a read-only, offline reference.

This tool never executes an archive member, imports its Python, or deserialises
its checkpoints. Existing extraction directories are deliberately rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
from zipfile import ZipFile, ZipInfo

ARCHIVE_SHA256 = "f9358b4eec96aa302df926a3a7dbf1b8e5401263cea0ada25612f0c0bae1d009"
ARCHIVE_ROOT = "triality_llm_paper_poc_20261023"
PAYLOAD_FILES = 61
MAX_EXPANDED_BYTES = 16 * 1024 * 1024


def member_path(info: ZipInfo) -> PurePosixPath:
    """Reject unsafe names, special files, and ambiguous Windows destinations."""
    # ZipInfo normalises backslashes on Windows; inspect the unmodified name.
    name = info.orig_filename
    path = PurePosixPath(name)
    mode = info.external_attr >> 16
    if (
        "\\" in name
        or ":" in name
        or path.is_absolute()
        or ".." in path.parts
        or len(path.parts) < 2
        or path.parts[0] != ARCHIVE_ROOT
        or str(path) != name
        or any(
            part.endswith((".", " ")) or PureWindowsPath(part).is_reserved()
            for part in path.parts
        )
        or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
        or info.is_dir()
    ):
        raise ValueError(f"Unsafe archive member: {name!r}")
    return path


def verify_archive(archive: Path, manuscript: Path) -> dict:
    """Check archive identity, complete manifest coverage and manuscript bytes."""
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError("Archive SHA-256 does not match the frozen intake")
    with ZipFile(archive) as zipped:
        entries = zipped.infolist()
        paths = [member_path(info) for info in entries]
        if len({str(path).casefold() for path in paths}) != len(paths):
            raise ValueError("Duplicate or case-colliding archive members")
        expanded = sum(info.file_size for info in entries)
        if expanded > MAX_EXPANDED_BYTES:
            raise ValueError("Archive exceeds the small-reference extraction budget")
        manifest_name = f"{ARCHIVE_ROOT}/MANIFEST.sha256"
        manifest = {}
        for line in zipped.read(manifest_name).decode("utf-8").splitlines():
            expected, separator, relative = line.partition("  ")
            if (
                not separator
                or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected)
                or relative in manifest
            ):
                raise ValueError("Malformed or duplicate manifest record")
            full_name = f"{ARCHIVE_ROOT}/{relative}"
            member_path(ZipInfo(full_name))
            manifest[relative] = expected
        actual_names = {str(path.relative_to(ARCHIVE_ROOT)) for path in paths}
        if len(manifest) != PAYLOAD_FILES or set(manifest) != actual_names - {
            "MANIFEST.sha256"
        }:
            raise ValueError("Manifest must cover exactly the 61 payload files")
        for relative, expected in manifest.items():
            data = zipped.read(f"{ARCHIVE_ROOT}/{relative}")
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Payload digest mismatch: {relative}")
        paper = zipped.read(f"{ARCHIVE_ROOT}/paper/manuscript_ja.md")
        if paper != manuscript.read_bytes():
            raise ValueError("Adjacent manuscript differs from archived manuscript")
    return {
        "archive_sha256": digest,
        "manuscript_sha256": hashlib.sha256(paper).hexdigest(),
        "payload_files_verified": len(manifest),
        "archive_files": len(entries),
        "expanded_bytes": expanded,
        "manuscript_byte_equal": True,
        "manifest": manifest,
    }


def extract_reference(archive: Path, destination: Path) -> None:
    """Create an exclusive tree, check containment, and mark each file read-only.

    This is accidental-write protection, not a tamper-proof filesystem boundary.
    A failed partial extraction is retained for inspection, never overwritten.
    """
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    for ancestor in destination.parents:
        if ancestor.is_symlink() or (
            hasattr(ancestor, "is_junction") and ancestor.is_junction()
        ):
            raise ValueError("Extraction ancestors must not be symlinks")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    root = destination.resolve()
    with ZipFile(archive) as zipped:
        entries = [(info, member_path(info)) for info in zipped.infolist()]
        if len({str(path).casefold() for _, path in entries}) != len(entries):
            raise ValueError("Duplicate or case-colliding archive members")
        for info, path in entries:
            target = root.joinpath(*path.relative_to(ARCHIVE_ROOT).parts)
            if not target.resolve().is_relative_to(root):
                raise ValueError("Extraction escaped its destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(zipped.read(info))
            target.chmod(stat.S_IREAD)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("manuscript", type=Path)
    parser.add_argument("--extract", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify_archive(args.archive, args.manuscript)
    if args.extract is not None:
        extract_reference(args.archive, args.extract)
        receipt["reference_directory"] = str(args.extract.resolve())
        receipt["protection"] = "exclusive creation; read-only files; not tamper-proof"
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
