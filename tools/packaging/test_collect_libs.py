"""CPU-only regression for distribution OpenMP dependencies in Linux packages."""
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

import binary_deps
import collect_libs


@unittest.skipUnless(sys.platform.startswith("linux") and all(shutil.which(x) for x in ("gcc", "ldd", "readelf")),
                     "needs Linux and the native ELF toolchain")
class CollectOpenMP(unittest.TestCase):
    def test_system_openmp_is_bundled_with_its_notice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "omp.c"
            source.write_text("#include <omp.h>\nint main(void) { return omp_get_max_threads() < 1; }\n")
            binary = root / "strata-inkling"
            subprocess.run(["gcc", "-fopenmp", str(source), "-o", str(binary)], check=True, capture_output=True)
            self.assertIn("libgomp.so.1", binary_deps.elf_needed(binary))
            stage = root / "stage"
            stage.mkdir()
            shutil.copy2(binary, stage / binary.name)
            self.assertEqual(collect_libs.main(["--binary", str(binary), "--out", str(stage)]), 0)
            self.assertEqual(binary_deps.uncovered(stage), [])
            self.assertTrue((stage / "libgomp.so.1").is_file())
            self.assertFalse((stage / "libgomp.so.1").is_symlink())
            self.assertTrue((stage / "licenses" / "libgomp.txt").is_file())
            self.assertNotIn("libgomp.so.1", (stage / "SYSTEM-LIBS.txt").read_text().splitlines())
            archive = root / "engine.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for p in stage.rglob("*"):
                    if p.is_file():
                        tar.add(p, arcname=str(p.relative_to(stage)))
            self.assertEqual(binary_deps.archive_uncovered(archive), [])


if __name__ == "__main__":
    unittest.main()
