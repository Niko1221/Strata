"""tools/packaging/keep_symbols.py: the file parsers (ELF sections and build-id, PE sections and CodeView record) and the check command, on files the
machine already has; the objcopy / dsymutil steps are exercised by the release builds."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "packaging"))
import keep_symbols as ks  # noqa: E402


def first_existing(*names):
    for n in names:
        if os.path.isfile(n):
            return Path(n)
    return None


class ParserTests(unittest.TestCase):
    def test_elf_sections_and_check(self):
        exe = first_existing("/bin/ls", "/usr/bin/ls", "/bin/sh")
        if exe is None or ks.elf_sections(exe) is None:
            self.skipTest("no 64-bit ELF here")
        names = [n for n, *_ in ks.elf_sections(exe)]
        self.assertIn(".text", names)
        size, digest = ks.elf_section_bytes(exe, ".text")
        self.assertGreater(size, 1000)
        self.assertEqual(len(digest), 64)
        args = type("A", (), {"a": str(exe), "b": str(exe)})()
        self.assertEqual(ks.do_check(args), 0)

    def test_pe_codeview_and_check(self):
        exe = first_existing(r"C:\Windows\System32\notepad.exe", r"C:\Windows\System32\cmd.exe", r"C:\Windows\explorer.exe")
        if exe is None:
            self.skipTest("no PE file here")
        info = ks.pe_info(exe)
        self.assertIsNotNone(info)
        self.assertIn(".text", info[0])
        if info[1] is not None:
            guid, age, pdb = info[1]
            self.assertEqual(len(guid), 32)
            self.assertTrue(pdb.lower().endswith(".pdb"))
        args = type("A", (), {"a": str(exe), "b": str(exe)})()
        self.assertEqual(ks.do_check(args), 0)

    def test_check_detects_different_code(self):
        with tempfile.TemporaryDirectory() as d:
            exe = first_existing("/bin/ls", "/usr/bin/ls", "/bin/sh")
            other = first_existing("/bin/cat", "/usr/bin/cat", "/bin/echo")
            if exe is None or other is None or ks.elf_sections(exe) is None or ks.elf_sections(other) is None:
                self.skipTest("no two ELF files here")
            args = type("A", (), {"a": str(exe), "b": str(other)})()
            self.assertEqual(ks.do_check(args), 1)

    def test_record_replaces_the_entry_of_a_file(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            ks.record(out, {"file": "strata", "id": "a"})
            ks.record(out, {"file": "strata-vision", "id": "b"})
            ks.record(out, {"file": "strata", "id": "c"})
            items = json.loads((out / "symbols.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted((e["file"], e["id"]) for e in items), [("strata", "c"), ("strata-vision", "b")])


if __name__ == "__main__":
    unittest.main()
