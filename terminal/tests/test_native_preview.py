"""CPU checks of the preview's fail-closed inventory and archive boundaries."""

from pathlib import Path
import struct
import stat
import sys
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packaging"))
import native_preview as preview  # noqa: E402


def pe() -> bytes:
    """Minimal PE32+ with empty imports; no executable instructions."""
    data = bytearray(512)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<H", data, 0x98, 0x20B)
    return bytes(data)


class PreviewTests(unittest.TestCase):
    def test_reparse_attribute_rejects_prospective_ancestor_without_path_junction_api(self):
        base = Path(tempfile.gettempdir()).absolute() / "junction-contract-only"
        original = Path.lstat

        def details(path, *args, **kwargs):
            if path == base:
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return original(path, *args, **kwargs)

        with patch.object(Path, "lstat", details):
            with self.assertRaisesRegex(ValueError, "reparse path"):
                preview.no_reparse(base / "not-created" / "payload.exe")

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "original.exe"
        self.source.write_bytes(pe())
        self.stage = self.root / "Strata preview"
        self.archive = self.root / "preview.tar.7z"
        self.spec = {
            "public": {"source_commit": "a" * 40, "package_commit": "b" * 40},
            "files": [
                {
                    "source": str(self.source),
                    "path": "strata.exe",
                    "sha256": preview.sha256(self.source),
                }
            ],
        }

    def fake_archive(self, command: list, **kwargs: object) -> None:
        if command[1] == "a":
            self.archive.write_bytes(b"test double; not a 7z integration test")

    def build(self) -> dict:
        with patch.object(
            preview.shutil,
            "disk_usage",
            return_value=type("Disk", (), {"free": 100 * 1024**3})(),
        ), patch.object(preview.subprocess, "run", side_effect=self.fake_archive):
            return preview.build(self.spec, self.stage, self.archive, Path("7z"))

    def test_real_tar_and_complete_payload_verification(self) -> None:
        self.build()
        with tarfile.open(self.archive.with_suffix("")) as archive:
            self.assertEqual(
                archive.getnames(),
                ["Strata preview/preview-manifest.json", "Strata preview/strata.exe"],
            )
            self.assertEqual(
                archive.extractfile("Strata preview/strata.exe").read(), pe()
            )
        self.assertEqual(len(preview.verify(self.stage)["files"]), 1)
        (self.stage / "unlisted.txt").write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "membership"):
            preview.verify(self.stage)

    def test_modified_source_rejected_without_stage(self) -> None:
        self.source.write_bytes(pe() + b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.build()
        self.assertFalse(self.stage.exists())

    def test_modified_extracted_binary_rejected(self) -> None:
        self.build()
        (self.stage / "strata.exe").write_bytes(pe() + b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            preview.verify(self.stage)

    def test_existing_evidence_preserved(self) -> None:
        self.stage.mkdir()
        sentinel = self.stage / "failure.txt"
        sentinel.write_text("retain")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.build()
        self.assertEqual(sentinel.read_text(), "retain")

    def test_archive_destination_preserved(self) -> None:
        self.archive.write_bytes(b"retain")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.build()
        self.assertEqual(self.archive.read_bytes(), b"retain")

    def test_unsafe_and_case_duplicate_destinations(self) -> None:
        for name in (
            "../escape",
            "C:/escape",
            "foo\\bar",
            "/escape",
            "aux.txt",
            "file.",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                preview.relative_name(name)
        self.spec["files"].append(dict(self.spec["files"][0], path="STRATA.EXE"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.build()

    def test_low_disk_rejects_before_stage(self) -> None:
        with patch.object(
            preview.shutil,
            "disk_usage",
            return_value=type("Disk", (), {"free": preview.RESERVE})(),
        ):
            with self.assertRaisesRegex(ValueError, "disk budget"):
                preview.build(self.spec, self.stage, self.archive, Path("7z"))
        self.assertFalse(self.stage.exists())

    def test_malformed_pe_is_not_silently_skipped(self) -> None:
        self.source.write_bytes(b"not PE")
        self.spec["files"][0]["sha256"] = preview.sha256(self.source)
        with self.assertRaisesRegex(ValueError, "not a PE"):
            self.build()
        self.assertTrue(self.stage.exists())  # failed evidence is retained

    def test_missing_recursive_dependency_fails(self) -> None:
        with patch.object(
            preview.binary_deps,
            "uncovered",
            return_value=[("strata.exe", "missing.dll")],
        ):
            with self.assertRaisesRegex(ValueError, "uncovered"):
                self.build()
        self.assertFalse(self.archive.exists())

    def test_dynamic_cublaslt_requirement(self) -> None:
        self.spec["files"].append(dict(self.spec["files"][0], path="cublas64_13.dll"))
        with self.assertRaisesRegex(ValueError, "cuBLASLt"):
            self.build()

    def test_private_public_metadata_rejected(self) -> None:
        self.spec["public"]["description"] = "C:/Users/example/private"
        with self.assertRaisesRegex(ValueError, "private absolute"):
            self.build()
        self.assertFalse(self.stage.exists())

    def test_system_dlls_are_never_redistributed(self) -> None:
        self.spec["files"].append(dict(self.spec["files"][0], path="nvcuda.dll"))
        with self.assertRaisesRegex(ValueError, "driver DLL"):
            self.build()


if __name__ == "__main__":
    unittest.main()
