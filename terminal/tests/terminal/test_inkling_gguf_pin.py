"""The Unsloth Inkling GGUF downloads are pinned to a commit (0221-DSPARK, release blocker found 2026-11-23).

setup verifies every downloaded GGUF against the byte size and SHA-256 table in strata_control/inkling_family.py. On 2026-11-23 14:54 UTC
(commit 19bf97a66719d0165f6f8082a9d59581912a3863 "Support mainline llama.cpp") unsloth/Inkling-2-Mini-GGUF replaced every main GGUF (+1,280 bytes
each) and the drafter (402,278,560 -> 503,889,952 bytes, another tensor layout) under the same file names, so resolve/main no longer matches the
table and every install of a GGUF would fail its check. The shipped source is the commit whose bytes the table describes."""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strata_control import inkling_family as F  # noqa: E402

PINNED = "800c946d7b9c830c86290df49afbbe91db8455f5"


class GgufPin(unittest.TestCase):
    def test_the_shipped_gguf_source_is_a_commit_not_main(self):
        src = F.gguf_sources(env={})                      # no override: the shipped file
        self.assertTrue(src, "no shipped GGUF source")
        base = src[0]["base"]
        self.assertTrue(re.search(r"^https://huggingface\.co/unsloth/Inkling-2-Mini-GGUF/resolve/[0-9a-f]{40}$", base), base)
        self.assertEqual(base.rsplit("/", 1)[1], PINNED)

    def test_every_gguf_url_goes_through_the_pin(self):
        src = F.sources_for(F.DSPARK_GGUF_FILE, env={})[0]
        for name in list(F.UNSLOTH_GGUF_FILES):
            urls = F.urls_for(name, src)
            self.assertEqual(urls, [f"https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/{PINNED}/{name}"])

    def test_the_table_still_describes_the_pinned_bytes(self):
        # the pin is only right while the table is the one measured against it (check_pin.py compared all 13 files to the repo at that commit)
        self.assertEqual(F.UNSLOTH_GGUF_FILES["Inkling-2-Mini-UD-Q4_K_XL.gguf"][0], 17_629_759_456)
        self.assertEqual(F.UNSLOTH_GGUF_FILES[F.DSPARK_GGUF_FILE], (402_278_560, "2ec78510da837e8fc3be54a14757a0c5fbd2d8d9e85a0d92853fafdf674f3773"))
        for name, (size, sha) in F.UNSLOTH_GGUF_FILES.items():
            self.assertGreater(size, 100_000_000, name)
            self.assertEqual(len(sha), 64, name)

    def test_both_layouts_of_a_main_gguf_are_known_the_drafter_has_one(self):
        # the re-uploaded main files (KL 0.000000, top-1 100 percent against the pinned ones on the same engine, UD-Q4_K_XL, 2000 tokens, 2026-11-23)
        for name, (size, sha) in F.ALT_GGUF_FILES.items():
            self.assertIn(name, F.UNSLOTH_GGUF_FILES)
            self.assertEqual(F.pins_for(name), [F.UNSLOTH_GGUF_FILES[name], (size, sha)])
            self.assertEqual(size - F.UNSLOTH_GGUF_FILES[name][0], 1280, name)
            self.assertNotEqual(sha, F.UNSLOTH_GGUF_FILES[name][1])
        self.assertEqual(F.pins_for(F.DSPARK_GGUF_FILE), [F.UNSLOTH_GGUF_FILES[F.DSPARK_GGUF_FILE]])   # the new drafter layout is not readable
        self.assertEqual(F.pins_for("Inkling-2-Mini-SQ-10.strata"), [F.FILES["Inkling-2-Mini-SQ-10.strata"]])
        self.assertEqual(F.pins_for("nothing.gguf"), [])

    def test_setup_accepts_a_file_in_either_layout(self):
        import hashlib
        import tempfile
        from unittest import mock
        import setup
        old, new = b"hello", b"hello2"
        files = {"X.gguf": (len(old), hashlib.sha256(old).hexdigest())}
        alt = {"X.gguf": (len(new), hashlib.sha256(new).hexdigest())}
        with tempfile.TemporaryDirectory() as td, mock.patch.object(F, "FILES", files), mock.patch.object(F, "ALT_GGUF_FILES", alt):
            td = Path(td)
            for body in (old, new):
                p = td / ("f-%d" % len(body)) / "X.gguf"
                p.parent.mkdir()
                p.write_bytes(body)
                a = mock.Mock(gguf_dir=None)
                self.assertTrue(setup.inkling_have(a, p), body)
                setup.inkling_verify(p, "X.gguf", False)          # passes, no exit
            bad = td / "bad" / "X.gguf"
            bad.parent.mkdir()
            bad.write_bytes(b"hello22")                              # neither size
            self.assertFalse(setup.inkling_have(mock.Mock(gguf_dir=None), bad))
            with self.assertRaises(SystemExit):
                setup.inkling_verify(bad, "X.gguf", False)

    def test_the_drafter_is_the_original_checkpoint_pinned_by_commit(self):
        # thinkingmachines/Inkling-2-Mini mtp.safetensors: Apache-2.0, not gated; the upload commit (later commits only touch README / config)
        self.assertEqual(F.DRAFTER_FILES[F.MTP_FILE], (635_801_416, "7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2"))
        self.assertEqual(F.pins_for(F.MTP_FILE), [F.DRAFTER_FILES[F.MTP_FILE]])
        src = F.sources_for(F.MTP_FILE, env={})
        self.assertEqual(F.urls_for(F.MTP_FILE, src[0]),
                         ["https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/1fb4134b9a74f9681054f4ef82e4bf1f458a2337/mtp.safetensors"])
        env = {F.DRAFTER_SOURCE_ENV: "https://example.org/d"}
        self.assertEqual(F.sources_for(F.MTP_FILE, env=env)[0]["base"], "https://example.org/d")
        self.assertEqual(F.sources_for(F.MTP_FILE, env=env)[-1]["base"], F.MTP_BASE)

    def _drafter(self, have=(), failing=(), reads_mtp=True):
        import tempfile
        from unittest import mock
        import setup
        calls = []

        def fetch(a, name, folder, own):
            calls.append(name)
            if name in failing:
                raise SystemExit(1)
            return folder / name

        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(setup, "inkling_fetch", side_effect=fetch), \
                mock.patch.object(setup, "inkling_have", side_effect=lambda a, p: p.name in have), \
                mock.patch.object(setup, "inkling_verify"):
            got = setup.inkling_fetch_drafter(mock.Mock(), Path(td), False, reads_mtp)
        return got.name, calls

    def test_setup_fetches_the_original_drafter_first(self):
        self.assertEqual(self._drafter(), (F.MTP_FILE, [F.MTP_FILE]))

    def test_setup_falls_back_to_unsloths_copy_when_the_original_cannot_be_fetched(self):
        self.assertEqual(self._drafter(failing=(F.MTP_FILE,)), (F.DSPARK_GGUF_FILE, [F.MTP_FILE, F.DSPARK_GGUF_FILE]))
        with self.assertRaises(SystemExit):
            self._drafter(failing=(F.MTP_FILE, F.DSPARK_GGUF_FILE))

    def test_setup_keeps_a_drafter_the_install_already_has(self):
        self.assertEqual(self._drafter(have=(F.DSPARK_GGUF_FILE,)), (F.DSPARK_GGUF_FILE, []))     # no second download
        self.assertEqual(self._drafter(have=(F.MTP_FILE,)), (F.MTP_FILE, []))

    def test_an_engine_without_the_flag_gets_unsloths_q8_file_only(self):
        # measured on Metal: an engine without the widen drafts nothing from mtp.safetensors (1.0 tokens per step), so setup does not fetch or pass it
        self.assertEqual(self._drafter(reads_mtp=False), (F.DSPARK_GGUF_FILE, [F.DSPARK_GGUF_FILE]))
        self.assertEqual(self._drafter(reads_mtp=False, have=(F.MTP_FILE,)), (F.DSPARK_GGUF_FILE, [F.DSPARK_GGUF_FILE]))     # on disk, not used
        self.assertEqual(self._drafter(reads_mtp=False, have=(F.MTP_FILE, F.DSPARK_GGUF_FILE)), (F.DSPARK_GGUF_FILE, []))
        with self.assertRaises(SystemExit):
            self._drafter(reads_mtp=False, failing=(F.DSPARK_GGUF_FILE,))                                                   # no fallback to the unreadable file

    def test_the_flag_decides_not_the_version(self):
        from unittest import mock
        import setup
        from serve import inkling_kind
        self.assertTrue(F.engine_reads_mtp({"arch": "inkling2_loop", "dspark_f32_widen": "1"}))
        for info in (None, {}, {"arch": "inkling2_loop", "spec_opts": "1"}, {"dspark_f32_widen": "0"}):
            self.assertFalse(F.engine_reads_mtp(info), info)
        with mock.patch.object(inkling_kind, "probe", return_value={"arch": "x", "dspark_f32_widen": "1"}) as pr:
            self.assertTrue(setup.inkling_engine_reads_mtp("/e/strata-inkling", "/m/x.gguf", ["/lib"]))
            self.assertEqual(pr.call_args[0][1], ["--native", "/m/x.gguf"])
        with mock.patch.object(inkling_kind, "probe", return_value={"arch": "x"}):
            self.assertFalse(setup.inkling_engine_reads_mtp("/e/strata-inkling", "/m/x.gguf"))
        with mock.patch.object(inkling_kind, "probe", return_value=None):
            self.assertFalse(setup.inkling_engine_reads_mtp("/e/strata-inkling", "/m/x.gguf"))
        self.assertFalse(setup.inkling_engine_reads_mtp(None, "/m/x.gguf"))                    # no engine: the file every engine reads

    def test_a_start_with_an_engine_that_cannot_read_mtp_uses_the_q8_file_beside_it(self):
        import tempfile
        from serve import inkling_kind
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            mtp, q8 = td / "mtp.safetensors", td / F.DSPARK_GGUF_FILE
            args = ["--native", "m.gguf", "--dspark", str(mtp)]
            old, new = {"arch": "x", "drafter": "dspark"}, {"arch": "x", "drafter": "dspark", "dspark_f32_widen": "1"}
            self.assertEqual(inkling_kind.fit_dspark(args, new, False), (args, new))             # a capable engine keeps it
            self.assertEqual(inkling_kind.fit_dspark(args, None, False), (args, None))
            other = ["--native", "m.gguf", "--dspark", str(q8)]
            self.assertEqual(inkling_kind.fit_dspark(other, old, False), (other, old))           # the Q8_0 file is always fine
            out, info = inkling_kind.fit_dspark(args, old, False)                                 # incapable, no Q8_0 beside it: no drafter
            self.assertEqual(out, ["--native", "m.gguf"])
            self.assertEqual(info["drafter"], "none")
            q8.write_bytes(b"x")
            out, info = inkling_kind.fit_dspark(args, old, False)                                 # incapable, Q8_0 beside it: that one
            self.assertEqual(out, ["--native", "m.gguf", "--dspark", str(q8)])
            self.assertEqual(info["drafter"], "dspark")
            self.assertEqual(args[3], str(mtp))                                                   # the caller's list is not changed

    def test_engine_args_passes_the_file_the_engine_reads(self):
        import tempfile
        from unittest import mock
        from serve import inkling_kind
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / F.DSPARK_GGUF_FILE).write_bytes(b"x")
            cfg = {"args": ["--native", str(td / "m.gguf"), "--dspark", str(td / "mtp.safetensors")], "drafting": "dspark", "backend": "cuda"}
            for flag, want in (({}, F.DSPARK_GGUF_FILE), ({"dspark_f32_widen": "1"}, "mtp.safetensors")):
                with mock.patch.object(inkling_kind, "probe", return_value={"arch": "x", "drafter": "dspark", **flag}):
                    args = inkling_kind.engine_args(dict(cfg), "/e/strata-inkling", announce=False)
                self.assertEqual(Path(args[args.index("--dspark") + 1]).name, want, flag)
                self.assertIn("dspark", args[args.index("--draft") + 1])

    def test_a_local_folder_prefers_the_original_drafter_only_for_an_engine_that_reads_it(self):
        import tempfile
        import setup
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "mtp.safetensors").write_bytes(b"x")
            self.assertEqual(setup.inkling_find_local_dspark(td).name, "mtp.safetensors")
            self.assertIsNone(setup.inkling_find_local_dspark(td, False))                        # only the original there, and this engine cannot read it
            (td / "dspark-Inkling-2-Mini-Q8_0.gguf").write_bytes(b"x")
            self.assertEqual(setup.inkling_find_local_dspark(td, False).name, "dspark-Inkling-2-Mini-Q8_0.gguf")
            self.assertEqual(setup.inkling_find_local_dspark(td).name, "mtp.safetensors")

    def test_an_override_still_wins(self):
        src = F.gguf_sources(env={F.GGUF_SOURCE_ENV: "https://example.org/gguf"})
        self.assertEqual(src[0]["base"], "https://example.org/gguf")
        self.assertEqual(src[-1]["base"].rsplit("/", 1)[1], PINNED)


if __name__ == "__main__":
    unittest.main()
