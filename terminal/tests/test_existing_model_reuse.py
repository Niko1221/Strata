"""Existing local weights must be selected explicitly, never replaced by a download."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from strata_control import models
from strata_control.profiles import ProfileError


class ExistingModelReuse(unittest.TestCase):
    def test_matching_set_is_reused_and_ambiguous_sets_refuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "SC117" / "Swift"
            first.mkdir(parents=True)
            (first / "Swift-IQ2_XS.gguf").write_bytes(b"fixture")
            def inspect(path):
                return {"candidate_selection": {"family": "swift", "model": "IQ2_XS"},
                        "file_set_status": "plausible", "model_dir": str(Path(path).parent)}
            with patch.object(models, "plan_import", side_effect=inspect):
                self.assertEqual(models.existing_model_folder(root, "swift", "IQ2_XS"), first)
                self.assertIsNone(models.existing_model_folder(root, "qwen", "IQ2_XS"))
                self.assertIsNone(models.existing_model_folder(root, "swift", "IQ2_XS", managed=first))
                second = root / "other"
                second.mkdir()
                (second / "Swift-IQ2_XS.gguf").write_bytes(b"fixture")
                with self.assertRaisesRegex(ProfileError, "More than one"):
                    models.existing_model_folder(root, "swift", "IQ2_XS")

    def test_incomplete_matching_set_blocks_automatic_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "Swift-IQ2_XS.gguf").write_bytes(b"fixture")
            with patch.object(models, "plan_import", return_value={
                "candidate_selection": {"family": "swift", "model": "IQ2_XS"},
                "file_set_status": "incomplete",
            }):
                with self.assertRaisesRegex(ProfileError, "need attention"):
                    models.existing_model_folder(root, "swift", "IQ2_XS")


if __name__ == "__main__":
    unittest.main()
