"""Existing model files need explicit replacement: no GPU or real downloads.

    python -m unittest tools.test_setup_downloads
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_golden import PROFILES, install  # noqa: E402
from test_setup_pins import Response  # noqa: E402


def gguf_bytes(template=None, tensors=2):
    data = bytearray(struct.pack("<IIQQ", 0x46554747, 3, tensors, int(template is not None)))
    if template is not None:
        key, value = b"tokenizer.chat_template", template.encode()
        data += struct.pack("<Q", len(key)) + key + struct.pack("<IQ", 8, len(value)) + value
    for i, name in enumerate(("blk.0.attn_q.weight", "blk.0.attn_k.weight")[:tensors]):
        data += struct.pack("<Q", len(name)) + name.encode() + struct.pack("<IQIQ", 1, 8, 0, 32 * i)
    return bytes(data) + bytes(-len(data) % 32 + 32 * tensors)


class Downloads(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dst = self.root / "model.gguf"
        self.original = gguf_bytes()
        self.dst.write_bytes(self.original)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def test_complete_unmarked_gguf_is_reused_without_network_or_copy(self):
        for url in ("https://example.com/model.gguf", str(self.root / "mirror.gguf")):
            self.dst.with_name("model.gguf.done").unlink(missing_ok=True)
            with mock.patch.object(setup.urllib.request, "urlopen") as network, \
                    mock.patch.object(setup.shutil, "copyfile") as copy:
                setup.download(url, self.dst)
            network.assert_not_called()
            copy.assert_not_called()
            self.assertEqual(self.dst.read_bytes(), self.original)
            self.assertTrue(setup.done(self.dst))

    def test_complete_file_is_reused_when_a_finish_mark_cannot_be_written(self):
        with mock.patch.object(Path, "write_text", side_effect=PermissionError("read only")), \
                mock.patch.object(setup.urllib.request, "urlopen") as network:
            setup.download("https://example.com/model.gguf", self.dst)
        network.assert_not_called()
        self.assertFalse(setup.done(self.dst))
        self.assertEqual(self.dst.read_bytes(), self.original)

    def test_an_existing_short_or_malformed_gguf_is_not_replaced(self):
        invalid_type = struct.pack("<IIQQQ", 0x46554747, 3, 0, 1, 1) + b"x" + struct.pack("<I", 99)
        for data in (self.original[:-40], b"not GGUF", b"", invalid_type):
            self.dst.write_bytes(data)
            with mock.patch.object(setup.urllib.request, "urlopen") as network, self.assertRaises(SystemExit):
                setup.download("https://example.com/model.gguf", self.dst)
            network.assert_not_called()
            self.assertEqual(self.dst.read_bytes(), data)

    def test_force_fetches_even_same_size_and_discards_old_partial_and_hash(self):
        new = self.original[:-1] + b"x"
        setup.mark(self.dst, "sha256 " + hashlib.sha256(self.original).hexdigest())
        self.dst.with_name("model.gguf.part").write_bytes(b"obsolete partial")
        calls = []

        def urlopen(req, timeout=None):
            self.assertEqual(self.dst.read_bytes(), self.original)  # old file stays until the transfer completes
            calls.append((req.get_method(), req.headers.get("Range")))
            return Response(new)

        with mock.patch.object(setup.urllib.request, "urlopen", urlopen):
            setup.download("https://example.com/model.gguf", self.dst, force=True)
        self.assertEqual(calls, [("HEAD", None), ("GET", "bytes=0-")])
        self.assertEqual(self.dst.read_bytes(), new)
        self.assertNotIn("sha256", self.dst.with_name("model.gguf.done").read_text())
        self.assertFalse(self.dst.with_name("model.gguf.part").exists())

    def test_failed_force_download_keeps_the_existing_file_and_mark(self):
        setup.mark(self.dst, "old marker")
        with mock.patch.object(setup.urllib.request, "urlopen", side_effect=OSError("offline")), \
                mock.patch.object(setup.time, "sleep"), self.assertRaises(SystemExit):
            setup.download("https://example.com/model.gguf", self.dst, force=True)
        self.assertEqual(self.dst.read_bytes(), self.original)
        self.assertEqual(self.dst.with_name("model.gguf.done").read_text(), "old marker")

    def test_invalid_forced_replacements_preserve_the_file_and_mark(self):
        expected = (len(self.original), hashlib.sha256(self.original).hexdigest())
        invalid_type = struct.pack("<IIQQQ", 0x46554747, 3, 0, 1, 1) + b"x" + struct.pack("<I", 99)
        cases = ((self.original[:-1] + b"x", expected), (self.original, (len(self.original) + 1, expected[1])),
                 (b"invalid GGUF".ljust(len(self.original), b"x"), None), (invalid_type, None))
        for data, pinned in cases:
            for local in (False, True):
                with self.subTest(pinned=pinned, local=local):
                    setup.mark(self.dst, "old marker")
                    src = self.root / "mirror.gguf"
                    src.write_bytes(data)
                    # A hash left by an earlier staged transfer must never skip checking the new bytes.
                    self.dst.with_name("model.gguf.part.done").write_text("sha256 " + expected[1])
                    with mock.patch.object(setup.urllib.request, "urlopen", side_effect=lambda *a, **k: Response(data)), \
                            self.assertRaises(SystemExit):
                        setup.download(str(src) if local else "https://example.com/model.gguf", self.dst,
                                       force=True, expected=pinned)
                    self.assertEqual(self.dst.read_bytes(), self.original)
                    self.assertEqual(self.dst.with_name("model.gguf.done").read_text(), "old marker")

    def test_overlong_single_tensor_replacement_preserves_the_original(self):
        data = gguf_bytes(tensors=1) + bytes(64)
        src = self.root / "mirror.gguf"
        src.write_bytes(data)
        self.assertTrue(setup.whole_shard(src))        # minimum length alone misses the PLE trailing-data rule
        for local in (False, True):
            with self.subTest(local=local):
                self.dst.write_bytes(self.original)
                setup.mark(self.dst, "original verified marker")
                with mock.patch.object(setup.urllib.request, "urlopen", side_effect=lambda *a, **k: Response(data)), \
                        self.assertRaises(SystemExit):
                    setup.download(str(src) if local else "https://example.com/model.gguf", self.dst, force=True)
                self.assertEqual(self.dst.read_bytes(), self.original)
                self.assertEqual(self.dst.with_name("model.gguf.done").read_text(), "original verified marker")

    def test_modelscope_hash_is_checked_before_a_forced_replacement(self):
        new = self.original[:-1] + b"x"
        sha = hashlib.sha256(self.original).hexdigest()
        setup.mark(self.dst, "old marker")
        with mock.patch.object(setup, "model_source", return_value="modelscope"), \
                mock.patch.object(setup, "ms_file", return_value=("repo", "file")), \
                mock.patch.object(setup, "ms_url", return_value="https://example.com/model.gguf"), \
                mock.patch.object(setup, "reachable", return_value=True), \
                mock.patch.object(setup, "ms_meta", return_value=(len(new), sha)), \
                mock.patch.object(setup.urllib.request, "urlopen", side_effect=lambda *a, **k: Response(new)), \
                self.assertRaises(SystemExit):
            setup.download("https://example.com/model.gguf", self.dst, force=True)
        self.assertEqual(self.dst.read_bytes(), self.original)
        self.assertEqual(self.dst.with_name("model.gguf.done").read_text(), "old marker")

    def test_a_valid_pinned_replacement_is_verified_then_published(self):
        new = self.original[:-1] + b"x"
        sha = hashlib.sha256(new).hexdigest()
        verify = setup.verify_sha256

        def check(path, *args, **kwargs):
            self.assertEqual(path.name, "model.gguf.part")
            self.assertEqual(self.dst.read_bytes(), self.original)
            return verify(path, *args, **kwargs)

        with mock.patch.object(setup.urllib.request, "urlopen", side_effect=lambda *a, **k: Response(new)), \
                mock.patch.object(setup, "verify_sha256", side_effect=check):
            setup.download("https://example.com/model.gguf", self.dst, force=True, expected=(len(new), sha))
        self.assertEqual(self.dst.read_bytes(), new)
        self.assertEqual(self.dst.with_name("model.gguf.done").read_text(), "sha256 " + sha)
        self.assertFalse(self.dst.with_name("model.gguf.part.done").exists())

    def test_local_force_copy_replaces_only_after_copy_finishes(self):
        src = self.root / "new.gguf"
        src.write_bytes(self.original[:-1] + b"x")
        setup.download(str(src), self.dst, force=True)
        self.assertEqual(self.dst.read_bytes(), src.read_bytes())
        with mock.patch.object(setup.shutil, "copyfile", side_effect=OSError("disk full")), \
                self.assertRaises(OSError):
            setup.download(str(src), self.dst, force=True)
        self.assertEqual(self.dst.read_bytes(), src.read_bytes())

    def test_missing_destination_still_resumes_an_existing_part(self):
        self.dst.unlink()
        self.dst.with_name("model.gguf.part").write_bytes(self.original[:32])
        calls = []

        def urlopen(req, timeout=None):
            calls.append((req.get_method(), req.headers.get("Range")))
            return Response(self.original) if req.get_method() == "HEAD" else Response(self.original[32:], status=206)

        with mock.patch.object(setup.urllib.request, "urlopen", urlopen):
            setup.download("https://example.com/model.gguf", self.dst)
        self.assertEqual(calls, [("HEAD", None), ("GET", "bytes=32-")])
        self.assertEqual(self.dst.read_bytes(), self.original)

    def test_forced_download_reuses_its_disk_planning_head_information(self):
        new = self.original[:-1] + b"x"
        calls = []

        def urlopen(req, timeout=None):
            calls.append(req.get_method())
            return Response(new)

        with mock.patch.object(setup.urllib.request, "urlopen", urlopen):
            info = setup.download_info("https://example.com/model.gguf")
            setup.download("https://example.com/model.gguf", self.dst, force=True, info=info)
        self.assertEqual(calls, ["HEAD", "GET"])
        self.assertEqual(self.dst.read_bytes(), new)

    def test_a_local_file_with_the_wrong_pinned_hash_can_be_preserved(self):
        with self.assertRaises(SystemExit):
            setup.verify_sha256(self.dst, len(self.original), "0" * 64, remove_bad=False)
        self.assertEqual(self.dst.read_bytes(), self.original)


class ReplacementSpace(unittest.TestCase):
    def test_peak_follows_shard_order_and_released_files(self):
        cases = (("missing last", (10, 50, None), (10, 50, 44), 50),
                 ("missing middle", (10, None, 44), (10, 50, 44), 94),
                 ("all present", (10, 50, 44), (10, 50, 44), 50),
                 ("all missing", (None, None, None), (10, 50, 44), 104),
                 ("growing", (10, 20, 30), (20, 40, 60), 90),
                 ("shrinking", (40, 40, 40), (20, 20, 20), 20))
        for name, old, new, peak in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                shards = [Path(tmp) / f"shard-{i}.gguf" for i in range(3)]
                for shard, size in zip(shards, old):
                    if size is not None:
                        shard.write_bytes(bytes(size))
                self.assertEqual(setup.replacement_space(shards, list(new)), peak)

    def test_discarded_partial_space_is_released_in_transfer_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            shards = [Path(tmp) / f"shard-{i}.gguf" for i in range(3)]
            shards[0].write_bytes(bytes(10))
            shards[2].write_bytes(bytes(44))
            shards[1].with_name(shards[1].name + ".part").write_bytes(bytes(30))
            self.assertEqual(setup.replacement_space(shards, [10, 50, 44]), 64)

    def test_linked_shards_do_not_release_their_targets_space(self):
        for symlink in (False, True):
            with self.subTest(symlink=symlink), tempfile.TemporaryDirectory() as tmp:
                shards = [Path(tmp) / f"shard-{i}.gguf" for i in range(3)]
                shards[0].write_bytes(bytes(10))
                source = Path(tmp) / "shared.gguf"
                source.write_bytes(bytes(50))
                if symlink:
                    try:
                        shards[1].symlink_to(source)
                    except OSError:
                        continue                       # Windows may require permission to create symlinks
                else:
                    setup.os.link(source, shards[1])
                self.assertEqual(setup.replacement_space(shards, [10, 50, 44]), 94)


class Setup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.models = Path(self.tmp.name) / "models"
        self.fam = setup.FAMILIES["qwen"]
        self.shards = [self.models / "Q2_0" / setup.model_file(self.fam, "Q2_0", i) for i in (1, 2)]
        for shard in self.shards:
            shard.parent.mkdir(parents=True, exist_ok=True)
            shard.write_bytes(gguf_bytes())
        self.argv = ["--models-dir", str(self.models), "--family", "qwen", "--model", "Q2_0", "--no-start"]
        self.ram, self.cards = PROFILES["64GB-1x32GB"]
        info = mock.patch.object(setup, "download_info", side_effect=lambda url, what=None: (url, len(gguf_bytes()), None))
        info.start()
        self.addCleanup(info.stop)

    def old_pack(self):
        data = Path(self.tmp.name) / "data"
        pack = data / "packs/q2_0"
        (pack / "tokenizer").mkdir(parents=True, exist_ok=True)
        for name in ("index.txt", "native_experts.txt", "experts.bin", "tokenizer/vocab.json", "keep.txt"):
            (pack / name).write_bytes(b"old cache")
        (data / "mtp/rt").mkdir(parents=True)
        (data / "mtp/rt/experts.bin").touch()
        return data, pack

    def test_whole_unmarked_shards_are_reused_even_without_writable_marks(self):
        # setup's harness mocks GGUF parsing; the download tests above exercise the real header reader.
        download = mock.Mock(side_effect=AssertionError("downloaded an existing model"))
        code, out, _, _ = install(self.ram, self.cards, self.argv, extra=[
            mock.patch.object(setup, "whole_shard", return_value=True),
            mock.patch.object(setup, "mark", return_value=None),
            mock.patch.object(setup, "download", download),
        ])
        self.assertEqual(code, 0, out)
        download.assert_not_called()

    def test_force_replaces_marked_shards_and_vision_and_rebuilds_the_pack(self):
        for shard in self.shards:
            setup.mark(shard)
        encoder = self.models / self.fam["mmproj"]
        encoder.write_bytes(gguf_bytes())
        setup.mark(encoder)
        calls, builds = [], []

        def free_gb(path):
            pack = setup.ROOT / "data/packs/q2_0"
            pack.mkdir(parents=True, exist_ok=True)
            for name in ("index.txt", "native_experts.txt", "experts.bin", "keep.txt"):
                (pack / name).write_bytes(b"old cache")
            return 900

        def download(url, dst, what=None, **kwargs):
            calls.append((Path(dst), kwargs))
            self.assertTrue(kwargs.get("force"))
            dst.write_bytes(gguf_bytes())
            setup.mark(dst)

        def build(cmd, **kwargs):
            if "iq_pack.py" in cmd[1]:
                pack = Path(cmd[cmd.index("--out") + 1])
                self.assertFalse((pack / "experts.bin").exists())
                self.assertFalse((pack / "native_experts.txt").exists())
                self.assertTrue((pack / "keep.txt").exists())
                builds.append(cmd)

        code, out, _, _ = install(self.ram, self.cards, [*self.argv, "--force-download", "--vision", "cpu"], extra=[
            mock.patch.object(setup, "free_gb", free_gb),
            mock.patch.object(setup, "download", download),
            mock.patch.object(setup, "run", build),
        ])
        self.assertEqual(code, 0, out)
        self.assertEqual([path for path, _ in calls], [*self.shards, encoder])
        self.assertEqual(len(builds), 1)

    def test_forced_rebuild_uses_only_the_replacement_chat_template(self):
        import strata_tokenizer
        from gguf_reader import GGUFFile
        from types import SimpleNamespace

        data, pack = self.old_pack()
        template_path = pack / "tokenizer/chat_template.jinja"
        tokenizer = SimpleNamespace(pre="qwen35", tokens=["x"], ids={"x": 0}, ranks={},
                                    token_types=[1], special_ids={})
        for avx512 in (False, True):
            for template in (None, "replacement model template"):
                with self.subTest(avx512=avx512, template=template):
                    template_path.write_text("previous model template", encoding="utf-8")
                    builds = []

                    def download(url, dst, what=None, **kwargs):
                        dst.write_bytes(gguf_bytes(template))
                        setup.mark(dst)

                    def build(cmd, **kwargs):
                        if Path(cmd[1]).name in ("iq_pack.py", "strata_tokenizer.py"):
                            # Run the real exporter with a small vocabulary and real replacement GGUF metadata.
                            with mock.patch.object(strata_tokenizer.Tokenizer, "from_gguf", return_value=tokenizer), \
                                    mock.patch.dict(sys.modules, {"gguf_reader": SimpleNamespace(GGUFFile=GGUFFile)}):
                                strata_tokenizer.extract(self.shards[0], pack)
                            builds.append(cmd)

                    code, out, _, _ = install(self.ram, self.cards,
                        [*self.argv, "--force-download", "--vision", "none"], avx512=avx512, extra=[
                            mock.patch.object(setup, "data_folder", return_value=(data, [])),
                            mock.patch.object(setup, "download", download), mock.patch.object(setup, "run", build)])
                    self.assertEqual(code, 0, out)
                    self.assertEqual(len(builds), 1)
                    if template is None:
                        self.assertFalse(template_path.exists())
                    else:
                        self.assertEqual(template_path.read_text(encoding="utf-8"), template)
                    self.assertTrue((pack / "tokenizer/vocab.json").exists())
                    self.assertEqual((pack / "keep.txt").read_bytes(), b"old cache")

    def test_failed_replacement_blocks_repacking_until_a_full_forced_retry(self):
        data, pack = self.old_pack()
        pending = self.shards[0].parent / setup.REPLACEMENT_INCOMPLETE
        for shard in self.shards:
            setup.mark(shard)
        replacement = gguf_bytes()[:-1] + b"x"
        calls = []

        def download(url, dst, what=None, **kwargs):
            self.assertTrue(pending.exists())
            self.assertFalse((pack / "index.txt").exists())
            self.assertFalse((pack / "native_experts.txt").exists())
            self.assertFalse((pack / "tokenizer/vocab.json").exists())
            calls.append(dst)
            if len(calls) == 2:
                raise SystemExit(1)
            dst.write_bytes(replacement)
            setup.mark(dst)

        folder = mock.patch.object(setup, "data_folder", return_value=(data, []))
        code, out, _, _ = install(self.ram, self.cards, [*self.argv, "--force-download", "--vision", "none"], extra=[
            folder, mock.patch.object(setup, "download", download)])
        self.assertEqual(code, 1, out)
        self.assertEqual(self.shards[0].read_bytes(), replacement)
        self.assertEqual(self.shards[1].read_bytes(), gguf_bytes())
        self.assertFalse((pack / "experts.bin").exists())
        self.assertTrue(pending.exists())
        builds = []

        def build(cmd, **kwargs):
            if "iq_pack.py" in cmd[1]:
                self.assertFalse(pending.exists())
                self.assertEqual(Path(cmd[cmd.index("--out") + 1]), pack)
                builds.append(cmd)
                (pack / "native_experts.txt").write_bytes(b"rebuilt")
                (pack / "tokenizer/vocab.json").write_bytes(b"rebuilt")

        code, out, _, _ = install(self.ram, self.cards, [*self.argv, "--vision", "none"], extra=[
            mock.patch.object(setup, "data_folder", return_value=(data, [])),
            mock.patch.object(setup, "download", side_effect=AssertionError("downloaded complete shards")),
            mock.patch.object(setup, "run", build)])
        self.assertEqual(code, 1, out)
        self.assertIn("replacement is incomplete", out)
        self.assertIn("--force-download", out)
        self.assertEqual(builds, [])
        self.assertTrue(pending.exists())

        restored = gguf_bytes()[:-1] + b"y"
        calls.clear()

        def download_all(url, dst, what=None, **kwargs):
            self.assertTrue(pending.exists())
            self.assertTrue(kwargs["force"])
            calls.append(dst)
            dst.write_bytes(restored)
            setup.mark(dst)

        code, out, _, _ = install(self.ram, self.cards, [*self.argv, "--force-download", "--vision", "none"], extra=[
            mock.patch.object(setup, "data_folder", return_value=(data, [])),
            mock.patch.object(setup, "download", download_all), mock.patch.object(setup, "run", build)])
        self.assertEqual(code, 0, out)
        self.assertEqual(calls, self.shards)
        self.assertTrue(all(shard.read_bytes() == restored for shard in self.shards))
        self.assertEqual(len(builds), 1)
        self.assertFalse(pending.exists())
        self.assertEqual((pack / "keep.txt").read_bytes(), b"old cache")

    def test_encoder_failure_keeps_the_replacement_marker(self):
        data, pack = self.old_pack()
        pending = self.shards[0].parent / setup.REPLACEMENT_INCOMPLETE

        def download(url, dst, what=None, **kwargs):
            self.assertTrue(pending.exists())
            if what == "vision encoder":
                raise SystemExit(1)
            dst.write_bytes(gguf_bytes())
            setup.mark(dst)

        code, out, _, _ = install(self.ram, self.cards, [*self.argv, "--force-download", "--vision", "cpu"], extra=[
            mock.patch.object(setup, "data_folder", return_value=(data, [])),
            mock.patch.object(setup, "download", download),
            mock.patch.object(setup, "run", side_effect=AssertionError("packed incomplete replacement"))])
        self.assertEqual(code, 1, out)
        self.assertTrue(pending.exists())
        self.assertFalse((pack / "native_experts.txt").exists())

    def test_pack_failure_after_full_replacement_allows_a_normal_retry(self):
        data, pack = self.old_pack()
        pending = self.shards[0].parent / setup.REPLACEMENT_INCOMPLETE

        def download(url, dst, what=None, **kwargs):
            dst.write_bytes(gguf_bytes())
            setup.mark(dst)

        code, out, _, _ = install(self.ram, self.cards, [*self.argv, "--force-download", "--vision", "none"], extra=[
            mock.patch.object(setup, "data_folder", return_value=(data, [])),
            mock.patch.object(setup, "download", download),
            mock.patch.object(setup, "run", side_effect=SystemExit(1))])
        self.assertEqual(code, 1, out)
        self.assertFalse(pending.exists())
        builds = mock.Mock()
        code, out, _, _ = install(self.ram, self.cards, [*self.argv, "--vision", "none"], extra=[
            mock.patch.object(setup, "data_folder", return_value=(data, [])),
            mock.patch.object(setup, "download", side_effect=AssertionError("downloaded complete replacement")),
            mock.patch.object(setup, "run", builds)])
        self.assertEqual(code, 0, out)
        self.assertTrue(any("iq_pack.py" in call.args[0][1] for call in builds.call_args_list))

    def test_shared_shards_from_an_incomplete_replacement_are_not_reused(self):
        self.shards[1].unlink()
        source = self.models / "IQ2_XS" / setup.model_file(self.fam, "IQ2_XS", 2)
        source.parent.mkdir()
        source.write_bytes(b"incomplete replacement's shared shard")
        setup.mark(source)
        (source.parent / setup.REPLACEMENT_INCOMPLETE).touch()
        download = mock.Mock()
        link = mock.Mock()
        code, out, _, _ = install(self.ram, self.cards, [*self.argv, "--vision", "none"], extra=[
            mock.patch.object(setup, "whole_shard", return_value=True),
            mock.patch.object(setup, "download", download), mock.patch.object(setup.os, "link", link)
        ])
        self.assertEqual(code, 0, out)
        self.assertEqual(download.call_args.args[1], self.shards[1])
        link.assert_not_called()

    def test_force_passes_pinned_hashes_to_staged_validation(self):
        for shard in self.shards:
            setup.mark(shard, "old marker")
        pinned = {shard.name: (len(gguf_bytes()), hashlib.sha256(gguf_bytes()).hexdigest()) for shard in self.shards}
        family = {**self.fam, "sha256": pinned}
        real_download, real_verify = setup.download, setup.verify_sha256
        new = gguf_bytes()[:-1] + b"x"
        code, out, _, _ = install(self.ram, self.cards, [*self.argv, "--force-download", "--vision", "none"], extra=[
            mock.patch.dict(setup.FAMILIES, {"qwen": family}),
            mock.patch.object(setup, "download", real_download),
            mock.patch.object(setup, "verify_sha256", real_verify),
            mock.patch.object(setup, "whole_shard", return_value=True),
            mock.patch.object(setup.urllib.request, "urlopen", side_effect=lambda *a, **k: Response(new))])
        self.assertEqual(code, 1, out)
        self.assertIn("wrong SHA-256", out)
        for shard in self.shards:
            self.assertEqual(shard.read_bytes(), gguf_bytes())
            self.assertEqual(shard.with_name(shard.name + ".done").read_text(), "old marker")

    def test_disk_check_counts_missing_shards_and_replacement_space(self):
        fam = setup.FAMILIES["unsloth"]
        for i in (1, 3):
            shard = self.models / "unsloth-UD-IQ4_XS" / setup.model_file(fam, "UD-IQ4_XS", i)
            shard.parent.mkdir(parents=True, exist_ok=True)
            with shard.open("wb") as f:
                f.truncate(fam["sha256"][shard.name][0])  # sparse files: real sizes without allocating 44 GB
            setup.mark(shard)
        download = mock.Mock(side_effect=AssertionError("passed an insufficient disk check"))
        code, out, _, _ = install(self.ram, self.cards, ["--models-dir", str(self.models), "--family", "unsloth",
            "--model", "UD-IQ4_XS", "--force-download", "--vision", "none", "--no-start"], extra=[
                mock.patch.object(setup, "free_gb", return_value=60),
                mock.patch.object(setup, "download", download)])
        self.assertEqual(code, 1, out)
        self.assertIn("not enough free disk space", out)
        self.assertIn("need ~96 GB", out)
        download.assert_not_called()

    def test_disk_check_allows_a_missing_last_shard_when_the_peak_fits(self):
        fam = setup.FAMILIES["unsloth"]
        for i in (1, 2):
            shard = self.models / "unsloth-UD-IQ4_XS" / setup.model_file(fam, "UD-IQ4_XS", i)
            shard.parent.mkdir(parents=True, exist_ok=True)
            with shard.open("wb") as f:
                f.truncate(fam["sha256"][shard.name][0])
            setup.mark(shard)

        def prebuilt(*args, **kwargs):
            engine = setup.ROOT / "engine"
            (engine / "BUILD.json").write_text('{"version": "0.1.40", "source": "local"}')
            return engine

        download = mock.Mock()
        code, out, _, _ = install(self.ram, self.cards, ["--models-dir", str(self.models), "--family", "unsloth",
            "--model", "UD-IQ4_XS", "--force-download", "--vision", "none", "--no-start"], extra=[
                mock.patch.object(setup, "free_gb", return_value=60),
                mock.patch.object(setup, "get_prebuilt", prebuilt),
                mock.patch.object(setup, "download", download)])
        self.assertEqual(code, 0, out)
        self.assertEqual(download.call_count, 3)

    def test_disk_check_reserves_space_to_rebuild_a_forced_pack(self):
        data, pack = self.old_pack()
        for shard in self.shards:
            setup.mark(shard)
        download = mock.Mock(side_effect=AssertionError("passed an insufficient disk check"))
        code, out, _, _ = install(self.ram, self.cards, [*self.argv, "--force-download", "--vision", "none"],
            avx512=True, extra=[mock.patch.object(setup, "data_folder", return_value=(data, [])),
                mock.patch.object(setup, "free_gb", return_value=20),
                mock.patch.object(setup, "download", download)])
        self.assertEqual(code, 1, out)
        self.assertIn("need ~42 GB", out)
        self.assertTrue((pack / "experts.bin").exists())
        download.assert_not_called()

    def test_force_conflicts_are_rejected_before_installation(self):
        for flags in (["--gguf-dir", str(self.models)], ["--check"], ["--update"], ["--rollback-engine"]):
            with self.subTest(flags=flags), mock.patch.object(sys, "argv", ["setup.py", "--force-download", *flags]), \
                    mock.patch.object(setup, "data_folder") as install_data, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                setup.main()
            self.assertEqual(error.exception.code, 2)
            install_data.assert_not_called()


if __name__ == "__main__":
    unittest.main()
