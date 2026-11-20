"""Small external-input boundary fixtures; never load the supplied checkpoints."""

import hashlib
import stat
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

import intake


@pytest.mark.parametrize(
    "name",
    [
        "../outside",
        "/absolute",
        f"{intake.ARCHIVE_ROOT}/../outside",
        f"{intake.ARCHIVE_ROOT}/folder\\outside",
        f"{intake.ARCHIVE_ROOT}/file:stream",
        f"{intake.ARCHIVE_ROOT}/AUX.txt",
        f"{intake.ARCHIVE_ROOT}/file.",
        f"{intake.ARCHIVE_ROOT}/space ",
        f"{intake.ARCHIVE_ROOT}//normalised",
    ],
)
def test_reject_unsafe_member_names(name: str) -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        intake.member_path(ZipInfo(name))


def test_reject_symlink_member() -> None:
    entry = ZipInfo(f"{intake.ARCHIVE_ROOT}/link")
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ValueError, match="Unsafe"):
        intake.member_path(entry)


def fixture_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    archive = tmp_path / "fixture.zip"
    manuscript = tmp_path / "manuscript.md"
    manuscript.write_bytes(b"fixture manuscript\n")
    data = {f"small/{i}.txt": str(i).encode() for i in range(60)}
    data["paper/manuscript_ja.md"] = manuscript.read_bytes()
    manifest = "\n".join(
        f"{hashlib.sha256(content).hexdigest()}  {name}"
        for name, content in data.items()
    )
    with ZipFile(archive, "w") as zipped:
        for name, content in data.items():
            zipped.writestr(f"{intake.ARCHIVE_ROOT}/{name}", content)
        zipped.writestr(f"{intake.ARCHIVE_ROOT}/MANIFEST.sha256", manifest)
    monkeypatch.setattr(
        intake, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    return archive, manuscript


def test_full_manifest_coverage_and_manuscript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, manuscript = fixture_archive(tmp_path, monkeypatch)
    result = intake.verify_archive(archive, manuscript)
    assert result["payload_files_verified"] == 61
    assert result["archive_files"] == 62
    manuscript.write_bytes(b"different")
    with pytest.raises(ValueError, match="manuscript differs"):
        intake.verify_archive(archive, manuscript)


def test_payload_corruption_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, manuscript = fixture_archive(tmp_path, monkeypatch)
    with ZipFile(archive) as zipped:
        records = [(entry.filename, zipped.read(entry)) for entry in zipped.infolist()]
    with ZipFile(archive, "w") as zipped:
        for name, data in records:
            zipped.writestr(name, b"changed" if name.endswith("small/0.txt") else data)
    monkeypatch.setattr(
        intake, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="Payload digest mismatch"):
        intake.verify_archive(archive, manuscript)


def test_extract_exclusive_and_case_collisions(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr(f"{intake.ARCHIVE_ROOT}/one.txt", b"one")
        zipped.writestr(f"{intake.ARCHIVE_ROOT}/ONE.txt", b"two")
    destination = tmp_path / "reference"
    with pytest.raises(ValueError, match="case-colliding"):
        intake.extract_reference(archive, destination)
    assert list(destination.iterdir()) == []
    with pytest.raises(FileExistsError):
        intake.extract_reference(archive, destination)
