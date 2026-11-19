import tempfile
import unittest
from pathlib import Path
from strata_control.pack_space import remaining_pack_gb


class PackSpace(unittest.TestCase):
    def test_existing_allocation_reduces_only_additional_disk_need(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(remaining_pack_gb(root, 36), 36)
            file = root / "experts.bin.tmp"
            content = b"x" * (1024 * 1024)
            file.write_bytes(content)
            need = remaining_pack_gb(root, 36)
            self.assertLess(need, 36)
            self.assertGreaterEqual(need, 36 - len(content) / 1e9)
            self.assertEqual(file.read_bytes(), content)
            self.assertEqual(remaining_pack_gb(root, 0), 0)
