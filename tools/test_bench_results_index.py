"""docs/bench-results/README.md lists every result folder (tools/bench_results_index.py)."""
import os
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_results_index as bri  # noqa: E402


class BenchResultsIndex(unittest.TestCase):
    def test_the_index_is_current_and_lists_every_folder(self):
        have = open(bri.OUT, encoding="utf-8", newline="").read().replace("\r\n", "\n")
        self.assertEqual(have, bri.build(), "run python tools/bench_results_index.py")
        for name in {p.split("/", 1)[0] for p in bri.published_files() if "/" in p}:
            self.assertIn("[%s](%s/)" % (name, name), have)

    def test_unpublished_files_do_not_change_counts_or_titles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "public").mkdir()
            (root / "public/report.md").write_text("# Published result\n", encoding="utf-8")
            with patch.object(bri, "RESULTS", str(root)), patch.object(bri, "published_files", return_value={"public/report.md"}):
                before = bri.build()
                (root / "private").mkdir()
                (root / "private/README.md").write_text("# Unpublished\n", encoding="utf-8")
                (root / "public/README.md").write_text("# Private working title\n", encoding="utf-8")
                self.assertEqual(bri.build(), before)
                self.assertIn("Published result | 1 |", before)


if __name__ == "__main__":
    unittest.main()
