"""Every distribution library an engine package bundles on purpose (collect_libs.py --bundle-system, and its built-in list) has its notice in
tools/packaging/notices/<name>.txt, and a script that bundles one also packs the notice.

    python -m unittest tools.test_bundle_system_notices -v

The 0.2.16 Intel Arc package bundles libz.so.1 (libur_loader.so.0 needs it, binary_deps.py no longer lists it as a base library); without the
notice file the copy carried no licence text, and without `--include 'licenses/*'` the notice never reached the package (package_engine.py without
--tree packs top-level files only). Nothing is built here.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]            # terminal/
PACK = ROOT.parent / "tools" / "packaging"
NOTICES = PACK / "notices"
sys.path.insert(0, str(PACK))
import collect_libs  # noqa: E402

BUNDLE_SYSTEM = re.compile(r"""--bundle-system\s+(?:'([^']+)'|"([^"]+)"|([^\s'"\\]+))""")


def notice_of(name: str) -> Path:
    return NOTICES / (name.split(".so", 1)[0] + ".txt")


def scripts():
    for p in sorted(PACK.iterdir()):
        if p.suffix in (".sh", ".bat", ".ps1") or (p.suffix == ".py" and p.name != "collect_libs.py"):
            yield p


class BundledSystemLibraries(unittest.TestCase):
    def test_the_built_in_list_has_its_notices(self):
        self.assertTrue(collect_libs.BUNDLE_FROM_SYSTEM)
        for name in collect_libs.BUNDLE_FROM_SYSTEM:
            with self.subTest(name=name):
                self.assertTrue(notice_of(name).is_file(), name)

    def test_every_bundle_system_name_in_a_package_script_has_a_notice(self):
        seen = 0
        for p in scripts():
            for m in BUNDLE_SYSTEM.finditer(p.read_text(encoding="utf-8", errors="replace")):
                name = next(g for g in m.groups() if g)
                seen += 1
                with self.subTest(script=p.name, library=name):
                    self.assertTrue(notice_of(name).is_file(), "%s bundles %s: tools/packaging/notices/%s is missing" % (p.name, name, notice_of(name).name))
        self.assertGreaterEqual(seen, 1)           # build_sycl_linux.sh names libz.so.1

    def test_a_script_that_bundles_one_packs_the_notices_too(self):
        for p in scripts():
            text = p.read_text(encoding="utf-8", errors="replace")
            if BUNDLE_SYSTEM.search(text) and "package_engine.py" in text and "--tree" not in text:
                with self.subTest(script=p.name):
                    self.assertIn("--include 'licenses/*'", text)

    def test_a_script_with_collect_libs_packs_the_notices_of_the_built_in_list(self):
        """libatomic.so.1 and libgomp.so.1 are bundled by collect_libs.py without any flag (BUNDLE_FROM_SYSTEM): whatever packs its output
        has to pass --include 'licenses/*', or the copy ships without its notice (#7189: strata-inkling needs libgomp.so.1)."""
        seen = 0
        for p in scripts():
            text = p.read_text(encoding="utf-8", errors="replace")
            if "collect_libs.py" in text and "package_engine.py --src" in text and "--tree" not in text:
                seen += 1
                with self.subTest(script=p.name):
                    self.assertIn("--include 'licenses/*'", text)
        self.assertGreaterEqual(seen, 2)           # build_cuda_linux.sh, build_cuda_linux_aarch64.sh, build_sycl_linux.sh

    def test_every_package_in_the_source_dockerfile_packs_the_notices(self):
        text = (PACK.parent / "docker" / "Dockerfile.src").read_text(encoding="utf-8")
        calls = [s for s in re.split(r";\s*\\?\n|;", text.replace("\\\n", " ")) if "package_engine.py --src" in s]
        self.assertEqual(len(calls), 3, "the engine, Hy4 and Inkling packages")
        for call in calls:
            with self.subTest(call=call.strip()[:90]):
                self.assertIn("--include 'licenses/*'", call)

    def test_the_notices_are_text_that_names_their_library(self):
        for n in sorted(NOTICES.glob("lib*.txt")):
            with self.subTest(notice=n.name):
                text = n.read_text(encoding="utf-8")
                self.assertTrue(text.startswith(n.stem + ".so"), n.name)
                self.assertIn("License", text)

    def test_the_vendor_notices_name_their_libraries(self):
        """intel-oneapi.txt (the oneAPI runtime of the SYCL packages: libiomp5, libsycl, oneMKL ...) and microsoft-vc-runtime.txt (VCOMP140.DLL and the
        other Visual C++ runtime DLLs of a Windows package): not a distribution library, so no libX.so name; they say what they cover and under what terms."""
        for name, first, words in (("intel-oneapi.txt", "Intel oneAPI runtime libraries", ("libiomp5", "License", "third-party")),
                                   ("microsoft-vc-runtime.txt", "Microsoft Visual C++ runtime libraries", ("VCOMP140.DLL", "License", "Microsoft"))):
            with self.subTest(notice=name):
                text = (NOTICES / name).read_text(encoding="utf-8")
                self.assertTrue(text.startswith(first), name)
                for w in words:
                    self.assertIn(w, text)


class VendorNotices(unittest.TestCase):
    """A package that carries Intel's oneAPI runtime or Microsoft's Visual C++ runtime carries their notice (the 0.2.22 packages carried both without)."""

    def test_oneapi_libraries_get_the_intel_notice_and_the_installations_license_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "opt" / "intel" / "oneapi"
            (root / "compiler" / "latest" / "lib").mkdir(parents=True)
            (root / "licensing" / "2026.1").mkdir(parents=True)
            (root / "licensing" / "2026.1" / "license.txt").write_text("INTEL LICENSE TEXT\n", encoding="utf-8")
            (root / "licensing" / "2026.1" / "third-party-programs.txt").write_text("THIRD PARTY TEXT\n", encoding="utf-8")
            iomp = root / "compiler" / "latest" / "lib" / "libiomp5.so"
            iomp.write_bytes(b"x")
            out = Path(tmp) / "out"
            out.mkdir()
            covered = collect_libs.copy_vendor_notices({"libiomp5.so": str(iomp), "libstdc++.so.6": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"}, out)
            self.assertEqual(covered, ["libiomp5.so"])
            notice = (out / "licenses" / "intel-oneapi.txt").read_text(encoding="utf-8")
            self.assertTrue(notice.startswith("Intel oneAPI runtime libraries"))
            self.assertIn("libiomp5.so <- " + str(iomp), notice)
            self.assertNotIn("libstdc++", notice)
            self.assertEqual((out / "licenses" / "intel-oneapi-license.txt").read_text(encoding="utf-8"), "INTEL LICENSE TEXT\n")
            self.assertEqual((out / "licenses" / "intel-oneapi-third-party-programs.txt").read_text(encoding="utf-8"), "THIRD PARTY TEXT\n")

    def test_add_notices_writes_licenses_for_a_staging_folder(self):
        import tempfile
        import add_notices
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "stage"
            stage.mkdir()
            for n in ("strata-hy4", "libiomp5.so", "libsycl.so.8", "libz.so.1", "VCOMP140.DLL"):
                (stage / n).write_bytes(b"x")
            root = Path(tmp) / "oneapi"
            (root / "licensing" / "2026.1").mkdir(parents=True)
            (root / "licensing" / "2026.1" / "license.txt").write_text("L\n", encoding="utf-8")
            wrote = add_notices.add(stage, [str(root)])
            self.assertIn("licenses/intel-oneapi.txt", wrote)
            self.assertIn("licenses/intel-oneapi-license.txt", wrote)
            self.assertIn("licenses/microsoft-vc-runtime.txt", wrote)
            text = (stage / "licenses" / "intel-oneapi.txt").read_text(encoding="utf-8")
            self.assertIn("libiomp5.so", text)
            self.assertIn("libsycl.so.8", text)
            self.assertNotIn("libz.so.1", text)
            plain = Path(tmp) / "plain"
            plain.mkdir()
            (plain / "strata").write_bytes(b"x")
            self.assertEqual(add_notices.add(plain, []), [])
            self.assertFalse((plain / "licenses").exists())

    def test_no_oneapi_library_no_intel_notice(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertEqual(collect_libs.copy_vendor_notices({"libgomp.so.1": "/usr/lib/x86_64-linux-gnu/libgomp.so.1"}, out), [])
            self.assertFalse((out / "licenses").exists())

    def test_a_windows_package_with_the_vc_runtime_gets_its_notice(self):
        import tempfile
        import package_engine
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            files = [{"path": "strata-inkling.exe"}, {"path": "VCOMP140.DLL"}, {"path": "MSVCP140.dll"}, {"path": "cudart64_13.dll"}]
            added = package_engine._vc_runtime_notice(staging, "windows-x86_64", files)
            self.assertEqual([a["path"] for a in added], ["licenses/microsoft-vc-runtime.txt"])
            text = (staging / "licenses" / "microsoft-vc-runtime.txt").read_text(encoding="utf-8")
            self.assertIn("  VCOMP140.DLL\n", text)
            self.assertIn("  MSVCP140.dll\n", text)
            self.assertNotIn("cudart", text)
            self.assertEqual(package_engine._vc_runtime_notice(staging / "x", "linux-x86_64", files), [])           # other platforms: nothing
            self.assertEqual(package_engine._vc_runtime_notice(staging / "y", "windows-x86_64", [{"path": "strata.exe"}]), [])   # no VC DLL: nothing


if __name__ == "__main__":
    unittest.main()
