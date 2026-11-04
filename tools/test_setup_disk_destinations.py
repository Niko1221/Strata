"""Disk preflight follows the files setup will write. No engines, downloads or GPU.

The drive tests emulate volume identity; real Windows volumes are a separate gate.
Run: python -m unittest tools.test_setup_disk_destinations
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import setup
from tools import test_setup_dsv41 as ds
from tools import test_setup_qwen36 as qw
from tools import test_setup_unsloth as flash


class DiskPools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.c, self.h = self.root / "C", self.root / "H"
        self.c.mkdir()
        self.h.mkdir()

    def check(self, writes, free, separate=False):
        def same(a, b):
            return (self.c in Path(a).parents or Path(a) == self.c) == (self.c in Path(b).parents or Path(b) == self.c)
        out = io.StringIO()
        with mock.patch.object(setup, "same_drive", side_effect=same if separate else lambda a, b: True), \
             mock.patch.object(setup, "free_gb", side_effect=free), contextlib.redirect_stdout(out):
            try:
                setup.check_setup_space(writes, 0)
                return 0, out.getvalue()
            except SystemExit as exc:
                return exc.code, out.getvalue()

    def test_folders_on_one_volume_share_one_budget(self):
        code, out = self.check([(self.c / "models", 6e9), (self.c / "packs", 6e9)], lambda p: 10)
        self.assertEqual(code, 1, out)
        self.assertIn("need ~12 GB", out)

    def test_separate_volumes_are_not_charged_the_combined_total(self):
        code, out = self.check([(self.c / "models", 6e9), (self.h / "packs", 8e9)], lambda p: 10, separate=True)
        self.assertEqual(code, 0, out)

    def test_each_destination_is_checked_even_when_its_share_is_smaller(self):
        code, out = self.check([(self.c / "models", 8e9), (self.h / "packs", 3e9)],
                               lambda p: 2 if p == self.h else 100, separate=True)
        self.assertEqual(code, 1, out)
        self.assertIn(str(self.h / "packs"), out)

    def test_zero_write_source_is_never_probed_or_created(self):
        source = self.c / "source-does-not-exist"
        code, out = self.check([(source, 0), (self.h / "data", 3e9)], lambda p: 100, separate=True)
        self.assertEqual(code, 0, out)
        self.assertFalse(source.exists())

    def test_nested_new_destination_probes_its_nearest_existing_ancestor(self):
        probed = []
        def free(path):
            probed.append(path)
            return 10
        code, out = self.check([(self.h / "new" / "nested" / "models", 2e9)], free, separate=True)
        self.assertEqual(code, 0, out)
        self.assertEqual(probed, [self.h])

    def test_same_volume_uses_filesystem_identity_not_path_text(self):
        self.assertTrue(setup.same_drive(self.c, self.h))  # two real directories on this test filesystem
        with mock.patch.object(setup, "free_gb", return_value=10), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                setup.check_setup_space([(self.c, 6e9), (self.h, 6e9)], 0)


class PartialFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dest = Path(self.tmp.name) / "model.gguf"
        self.part = self.dest.with_name(self.dest.name + ".part")

    def test_sequential_partial_receives_only_its_existing_bytes(self):
        self.part.write_bytes(b"abcdef")
        self.assertEqual(setup.setup_partial_bytes(self.dest, 10), 6)

    def test_oversized_partial_is_not_credited(self):
        self.part.write_bytes(b"x" * 11)
        self.assertEqual(setup.setup_partial_bytes(self.dest, 10), 0)
        self.assertEqual(self.part.stat().st_size, 11)  # preflight never deletes input

    def test_parallel_sparse_length_is_not_downloaded_bytes(self):
        with self.part.open("wb") as f:
            f.seek(11)
            f.write(b"x")
        self.part.with_name(self.part.name + ".chunks").write_text(json.dumps({"v": 1, "total": 12, "chunk": 4, "done": [[2, 3]]}))
        with mock.patch.object(setup, "DL_CHUNK", 4):
            self.assertEqual(setup.setup_partial_bytes(self.dest, 12), 4)

    def test_overlapping_chunk_ranges_are_counted_once(self):
        self.part.write_bytes(b"x" * 12)
        self.part.with_name(self.part.name + ".chunks").write_text(json.dumps({"v": 1, "total": 12, "chunk": 4, "done": [[0, 2], [1, 3]]}))
        with mock.patch.object(setup, "DL_CHUNK", 4):
            self.assertEqual(setup.setup_partial_bytes(self.dest, 12), 12)

    def test_invalid_chunk_map_is_not_credited(self):
        self.part.write_bytes(b"x" * 12)
        self.part.with_name(self.part.name + ".chunks").write_text("not json")
        self.assertEqual(setup.setup_partial_bytes(self.dest, 12), 0)

    def test_wrong_chunk_total_is_not_credited(self):
        self.part.write_bytes(b"x" * 12)
        self.part.with_name(self.part.name + ".chunks").write_text(json.dumps({"v": 1, "total": 12, "chunk": 4, "done": [[0, 3]]}))
        with mock.patch.object(setup, "DL_CHUNK", 4):
            self.assertEqual(setup.setup_partial_bytes(self.dest, 16), 0)

    def test_shared_file_plan_and_install_count_parallel_pieces_the_same_way(self):
        models = self.dest.parent / "models"
        folder = models / "deepseek-v4.1-flash-shared"
        folder.mkdir(parents=True)
        name = setup.DSV41_ISTA_ENGRAM
        part = folder / (name + ".part")
        part.write_bytes(b"x" * 12)
        part.with_name(part.name + ".chunks").write_text(json.dumps({"v": 1, "total": 12, "chunk": 4, "done": [[2, 3]]}))
        fam = setup.MULTI_FILE_FAMILIES["deepseek41"]
        with mock.patch.dict(fam["sha256"], {name: (12, "0" * 64)}), mock.patch.object(setup, "DL_CHUNK", 4):
            left = dict(setup.dsv41_shared_missing(ds.ISTA, models, vision=False))
        self.assertEqual(left[name], 8)


class QwenFiles(unittest.TestCase):
    def setUp(self):
        self.case = qw.InstallOwnPath()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)

    def test_missing_own_model_is_reported_before_disk_or_engine(self):
        own = self.case.root / "own"
        own.mkdir()
        with mock.patch.object(setup, "free_gb", return_value=0.001):
            code, out = self.case.install(model="UD-Q4_K_XL", vision="no", gguf_dir=str(own))
        self.assertEqual(code, 1, out)
        self.assertIn("is missing", out)
        self.assertNotIn("not enough free disk space", out)
        self.assertEqual(self.case.calls, [])

    def test_partial_own_file_is_not_treated_as_an_owned_complete_model(self):
        own = self.case.root / "own"
        own.mkdir()
        (own / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf.part").write_bytes(b"GGUF")
        code, out = self.case.install(model="UD-Q4_K_XL", vision="no", gguf_dir=str(own))
        self.assertEqual(code, 1, out)
        self.assertIn("is missing", out)
        self.assertEqual(self.case.calls, [])

    def test_model_and_picture_file_share_one_download_budget(self):
        fam = setup.OWN_PATH_FAMILIES["qwen36"]
        model_name = fam["file"].format(q="UD-Q4_K_XL")
        model_gb = fam["sha256"][model_name][0] / 1e9
        with mock.patch.object(setup, "free_gb", return_value=model_gb + 0.1):
            code, out = self.case.install(model="UD-Q4_K_XL", vision="cpu")
        self.assertEqual(code, 1, out)
        self.assertIn("not enough free disk space", out)
        self.assertFalse([c for c in self.case.calls if c[0] == "get"])

    def test_engine_disabling_pictures_does_not_require_own_picture_file(self):
        fam = setup.OWN_PATH_FAMILIES["qwen36"]
        own = self.case.root / "own"
        own.mkdir()
        (own / fam["file"].format(q="UD-Q4_K_XL")).write_bytes(b"GGUF")
        eng = self.case.root / "text-only-engine"
        eng.mkdir()
        def engine(a, gpu, hip, amd, cuda_tk, vision):
            return self.case.root / "llama", eng, {"version": "0.1.44"}, [], gpu, vision
        with mock.patch.object(setup, "engine_step", side_effect=engine):
            code, out = self.case.install(model="UD-Q4_K_XL", vision="cpu", gguf_dir=str(own))
        self.assertEqual(code, 0, out)
        self.assertNotIn("vision", self.case.cfg)
        self.assertFalse([c for c in self.case.calls if c[0] == "get"])

    def test_sequential_partial_needs_only_the_remaining_model_bytes(self):
        fam = setup.OWN_PATH_FAMILIES["qwen36"]
        name = fam["file"].format(q="UD-Q4_K_XL")
        folder = self.case.root / "models" / "qwen36-UD-Q4_K_XL"
        folder.mkdir(parents=True)
        (folder / (name + ".part")).write_bytes(b"x" * 800)
        with mock.patch.dict(fam["sha256"], {name: (1000, "0" * 64)}), \
             mock.patch.object(setup, "free_gb", return_value=250 / 1e9):
            code, out = self.case.install(model="UD-Q4_K_XL", vision="no")
        self.assertEqual(code, 0, out)
        self.assertTrue([c for c in self.case.calls if c[0] == "get"])

    def test_same_volume_picture_reuse_does_not_require_another_copy(self):
        fam = setup.OWN_PATH_FAMILIES["qwen36"]
        name = fam["file"].format(q="UD-Q4_K_XL")
        folder = self.case.root / "models" / "qwen36-UD-Q4_K_XL"
        folder.mkdir(parents=True)
        (folder / name).write_bytes(b"GGUF")
        setup.mark(folder / name)
        sibling = folder.parent / "qwen36-Q8_0"
        sibling.mkdir()
        (sibling / fam["mmproj"]).write_bytes(b"picture")
        with mock.patch.dict(fam["sha256"], {fam["mmproj"]: (7, "0" * 64)}), \
             mock.patch.object(setup, "free_gb", return_value=0):
            code, out = self.case.install(model="UD-Q4_K_XL", vision="cpu")
        self.assertEqual(code, 0, out)
        self.assertTrue((folder / fam["mmproj"]).samefile(sibling / fam["mmproj"]))

    def test_picture_link_fallback_checks_copy_destination_before_writing(self):
        fam = setup.OWN_PATH_FAMILIES["qwen36"]
        name = fam["file"].format(q="UD-Q4_K_XL")
        folder = self.case.root / "models" / "qwen36-UD-Q4_K_XL"
        folder.mkdir(parents=True)
        (folder / name).write_bytes(b"GGUF")
        setup.mark(folder / name)
        sibling = folder.parent / "qwen36-Q8_0"
        sibling.mkdir()
        source = sibling / fam["mmproj"]
        source.write_bytes(b"picture")
        with mock.patch.dict(fam["sha256"], {fam["mmproj"]: (7, "0" * 64)}), \
             mock.patch.object(setup, "free_gb", return_value=0), \
             mock.patch.object(setup.os, "link", side_effect=OSError("hard links unavailable")):
            code, out = self.case.install(model="UD-Q4_K_XL", vision="cpu")
        self.assertEqual(code, 1, out)
        self.assertIn("not enough free disk space", out)
        self.assertFalse((folder / fam["mmproj"]).exists())
        self.assertFalse((folder / (fam["mmproj"] + ".copying")).exists())
        self.assertEqual(source.read_bytes(), b"picture")

    def test_later_linkable_picture_is_tried_before_a_copy_space_failure(self):
        fam = setup.OWN_PATH_FAMILIES["qwen36"]
        name = fam["file"].format(q="UD-Q4_K_XL")
        folder = self.case.root / "models" / "qwen36-UD-Q4_K_XL"
        folder.mkdir(parents=True)
        (folder / name).write_bytes(b"GGUF")
        setup.mark(folder / name)
        for sibling_name in ("a-denied", "z-linkable"):
            sibling = folder.parent / sibling_name
            sibling.mkdir()
            (sibling / fam["mmproj"]).write_bytes(b"picture")
        real_link = setup.os.link
        tried = []
        def link(src, dst):
            tried.append(Path(src).parent.name)
            if Path(src).parent.name == "a-denied":
                raise OSError("hard link denied")
            return real_link(src, dst)
        with mock.patch.dict(fam["sha256"], {fam["mmproj"]: (7, "0" * 64)}), \
             mock.patch.object(setup, "free_gb", return_value=0), mock.patch.object(setup.os, "link", side_effect=link):
            code, out = self.case.install(model="UD-Q4_K_XL", vision="cpu")
        self.assertEqual(code, 0, out)
        self.assertEqual(tried, ["a-denied", "z-linkable"])
        self.assertTrue((folder / fam["mmproj"]).samefile(folder.parent / "z-linkable" / fam["mmproj"]))

    def test_oversized_partial_cannot_subtract_another_download(self):
        fam = setup.OWN_PATH_FAMILIES["qwen36"]
        name = fam["file"].format(q="UD-Q4_K_XL")
        folder = self.case.root / "models" / "qwen36-UD-Q4_K_XL"
        folder.mkdir(parents=True)
        part = folder / (name + ".part")
        part.write_bytes(b"x" * 2000)
        with mock.patch.dict(fam["sha256"], {name: (1000, "0" * 64)}), \
             mock.patch.object(setup, "free_gb", return_value=250 / 1e9):
            code, out = self.case.install(model="UD-Q4_K_XL", vision="no")
        self.assertEqual(code, 1, out)
        self.assertIn("not enough free disk space", out)
        self.assertEqual(self.case.calls, [])
        self.assertEqual(part.stat().st_size, 2000)


class DeepSeekFiles(unittest.TestCase):
    def setUp(self):
        self.case = ds.InstallOfficial()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.own = self.case.root / "own"
        self.own.mkdir()
        self.models = self.case.root / "models"
        self.models.mkdir()
        self.shared = self.models / "deepseek-v4.1-flash-shared"
        self.shared.mkdir()
        for n in setup.MULTI_FILE_FAMILIES["deepseek41"]["models"][ds.ISTA]["shards"]:
            (self.own / n).write_bytes(b"GGUF")

    def free_on(self, own_gb, data_gb):
        return lambda p: own_gb if p == self.own or self.own in Path(p).parents else data_gb

    def test_shared_download_checks_data_drive_not_owned_gguf_drive(self):
        with mock.patch.object(setup, "free_gb", side_effect=self.free_on(0.01, 900)):
            code, out = self.case.install_ista(gguf_dir=str(self.own))
        self.assertEqual(code, 0, out)
        self.assertTrue([c for c in self.case.calls if c[0] == "get"])
        self.assertTrue(all(c[2].parent == self.shared for c in self.case.calls if c[0] == "get"))

    def test_full_data_drive_stops_before_shared_download(self):
        with mock.patch.object(setup, "free_gb", side_effect=self.free_on(900, 0.01)):
            code, out = self.case.install_ista(gguf_dir=str(self.own))
        self.assertEqual(code, 1, out)
        self.assertIn("not enough free disk space", out)
        self.assertIn(str(self.shared), out)
        self.assertFalse([c for c in self.case.calls if c[0] == "get"])

    def test_missing_own_shard_wins_over_a_space_error(self):
        missing = next(self.own.glob("*.gguf"))
        missing.unlink()
        with mock.patch.object(setup, "free_gb", return_value=0.01):
            code, out = self.case.install_ista(gguf_dir=str(self.own))
        self.assertEqual(code, 1, out)
        self.assertIn(f"{missing} is missing", out)
        self.assertNotIn("not enough free disk space", out)
        self.assertEqual(self.case.calls, [])


class FlashNextFiles(flash.Base):
    def existing_q2(self):
        own = self.t / "own"
        own.mkdir()
        fam = setup.FAMILIES["qwen"]
        for i in range(1, setup.model_shards(fam, "Q2_0") + 1):
            file = own / setup.model_file(fam, "Q2_0", i)
            file.write_bytes(b"GGUF")
            setup.mark(file)
        return own

    def test_missing_own_flash_next_shard_wins_over_pack_space_error(self):
        own = self.existing_q2()
        missing = sorted(own.glob("*.gguf"))[-1]
        missing.unlink()
        code, out, _ = self.main(["--family", "qwen", "--gguf-dir", str(own), "--vision", "no", "--context", "8192"],
                                 family=False, m="Q2_0", ram=128, free_at=lambda p: 0)
        self.assertEqual(code, 1, out)
        self.assertIn(f"{missing} is missing", out)
        self.assertNotIn("not enough free disk space", out)
        self.assertEqual(self.downloads, [])
        self.assertEqual(self.runs, [])

    def test_own_gguf_drive_does_not_pay_for_new_pack(self):
        own = self.existing_q2()
        free_at = lambda p: 1 if p == own or own in Path(p).parents else 100
        code, out, _ = self.main(["--family", "qwen", "--gguf-dir", str(own), "--vision", "no", "--context", "8192"],
                                 family=False, m="Q2_0", ram=128, free_at=free_at)
        self.assertEqual(code, 0, out)
        self.assertTrue([r for r in self.runs if "strata_pack.py" in r[1]])

    def test_full_pack_destination_is_not_hidden_by_free_gguf_drive(self):
        own = self.existing_q2()
        free_at = lambda p: 100 if p == own or own in Path(p).parents else 1
        code, out, _ = self.main(["--family", "qwen", "--gguf-dir", str(own), "--vision", "no", "--context", "8192"],
                                 family=False, m="Q2_0", ram=128, free_at=free_at)
        self.assertEqual(code, 1, out)
        self.assertIn("not enough free disk space", out)
        self.assertIn(str(self.t / "data" / "packs"), out)
        self.assertEqual(self.runs, [])

    def test_explicit_models_dir_still_pays_for_picture_download(self):
        own = self.t / "own"
        own.mkdir()
        for name in setup.UNSLOTH_SHARDS:
            (own / name).write_bytes(b"GGUF")
            setup.mark(own / name)
        (self.t / "models").mkdir()
        free_at = lambda p: 0.01 if p == self.t / "models" else 1000
        code, out, _ = self.main(["--gguf-dir", str(own), "--vision", "cpu", "--context", "8192"], free_at=free_at)
        self.assertEqual(code, 1, out)
        self.assertIn("not enough free disk space", out)
        self.assertIn(str(self.t / "models"), out)
        self.assertEqual(self.downloads, [])

    def test_download_and_pack_on_separate_volumes_each_fit(self):
        models, data = self.t / "models", self.t / "data"
        models.mkdir()
        def same(a, b):
            return (models == a or models in Path(a).parents) == (models == b or models in Path(b).parents)
        free_at = lambda p: 75 if p == models or models in Path(p).parents else 45
        with mock.patch.object(setup, "same_drive", side_effect=same):
            code, out, _ = self.main(["--family", "qwen", "--vision", "no", "--context", "8192"],
                                     family=False, m="Q2_0", ram=128, free_at=free_at)
        self.assertEqual(code, 0, out)  # 66.4 GB on the model volume, 40 GB on the pack volume, not 106.4 on either

    def test_completed_pack_and_draft_are_not_charged_again(self):
        own = self.existing_q2()
        pack = self.t / "data" / "packs" / "q2_0"
        (pack / "tokenizer").mkdir(parents=True)
        for f in (pack / "experts.bin", pack / "index.txt", pack / "tokenizer" / "vocab.json"):
            f.write_text("x")
        code, out, _ = self.main(["--family", "qwen", "--gguf-dir", str(own), "--vision", "no", "--context", "8192"],
                                 family=False, m="Q2_0", ram=128, free_at=lambda p: 0)
        self.assertEqual(code, 0, out)
        self.assertFalse([r for r in self.runs if "strata_pack.py" in r[1]])

    def test_missing_draft_layer_is_charged_to_data(self):
        own = self.existing_q2()
        pack = self.t / "data" / "packs" / "q2_0"
        (pack / "tokenizer").mkdir(parents=True)
        for f in (pack / "experts.bin", pack / "index.txt", pack / "tokenizer" / "vocab.json"):
            f.write_text("x")
        (self.t / "data" / "mtp" / "rt" / "experts.bin").unlink()
        free_at = lambda p: 100 if p == own or own in Path(p).parents else 3
        code, out, _ = self.main(["--family", "qwen", "--gguf-dir", str(own), "--vision", "no", "--context", "8192"],
                                 family=False, m="Q2_0", ram=128, free_at=free_at)
        self.assertEqual(code, 1, out)
        self.assertIn(str(self.t / "data" / "mtp"), out)
        self.assertEqual(self.runs, [])


if __name__ == "__main__":
    unittest.main()
