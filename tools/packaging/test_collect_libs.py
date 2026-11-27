"""CPU-only real-ELF coverage for system libraries bundled in Linux packages."""
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

import binary_deps
import collect_libs


RUNTIME_FIXTURES = {
    "libatomic.so.1": (
        "static unsigned __int128 counter;\n"
        "int main(void) {\n"
        "    return __atomic_fetch_add(&counter, 1, __ATOMIC_SEQ_CST) != 0;\n"
        "}\n",
        ("-latomic",),
    ),
    "libgomp.so.1": (
        "#include <omp.h>\n"
        "int main(void) { return omp_get_max_threads() < 1; }\n",
        ("-fopenmp",),
    ),
}


@unittest.skipUnless(sys.platform.startswith("linux") and all(shutil.which(x) for x in ("gcc", "ldd", "readelf")),
                     "needs Linux and the native ELF toolchain")
class CollectSystemLibraries(unittest.TestCase):
    def test_every_system_bundle_has_a_real_elf_fixture(self):
        self.assertFalse(set(collect_libs.BUNDLE_FROM_SYSTEM) - RUNTIME_FIXTURES.keys(),
                         "add a real ELF fixture when adding a system-bundled runtime")

    def test_system_libraries_are_bundled_with_their_notices(self):
        for soname, (source, flags) in RUNTIME_FIXTURES.items():
            with self.subTest(soname=soname):
                self.check_runtime(soname, source, flags)

    def check_runtime(self, soname, code, flags):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "fixture.c"
            source.write_text(code)
            binary = root / "strata-fixture"
            subprocess.run(["gcc", str(source), "-o", str(binary), *flags],
                           check=True, capture_output=True, timeout=30)
            self.assertIn(soname, binary_deps.elf_needed(binary))
            dependencies, missing = collect_libs.dependencies(binary, [])
            self.assertEqual(missing, [])
            self.assertTrue(collect_libs.is_system(dependencies[soname]))
            subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)
            stage = root / "stage"
            stage.mkdir()
            shutil.copy2(binary, stage / binary.name)
            self.assertEqual(collect_libs.main(["--binary", str(binary), "--out", str(stage)]), 0)
            self.assertEqual(binary_deps.uncovered(stage), [])
            self.assertTrue((stage / soname).is_file())
            self.assertFalse((stage / soname).is_symlink())
            notice = soname.split(".so", 1)[0] + ".txt"
            self.assertTrue((stage / "licenses" / notice).is_file())
            self.assertNotIn(soname, (stage / "SYSTEM-LIBS.txt").read_text().splitlines())
            archive = root / "engine.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for p in stage.rglob("*"):
                    if p.is_file():
                        tar.add(p, arcname=str(p.relative_to(stage)))
            self.assertEqual(binary_deps.archive_uncovered(archive), [])


if __name__ == "__main__":
    unittest.main()
