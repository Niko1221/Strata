"""The tuned DFlash2 drafter and the Mac default (round 0216, DF-SHIP).

On Metal the two 4-bit sets (UD-Q4_K_XL, UD-IQ4_XS-v31) start DFlash2 under Auto: ``--spec 6``, no MTP, ``--draft-fallback none``, with the tuned pack at ``--draft-conf 0.7``
when it is here and verified, else the stock pack at 0.5, else (offline, nothing fetched) the MTP exactly as before. IQ3_S stays on the MTP until the one table entry
(drafting.SET_DEFAULTS) says otherwise. The pinned pack is replaced by a small synthetic file whose size and SHA-256 are patched in, so nothing here needs 340 MB.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

from strata_control import acquisition, drafting, drafting_fetch, transfer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packaging"))
import feed_tool as ft  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
Q4 = "qwen3.8-flash-next-unsloth-ud-q4_k_xl"
V31 = "qwen3.8-flash-next-unsloth-ud-iq4_xs-v31"
IQ3S = "qwen3.8-flash-next-iq3_s"
Q5 = "qwen3.8-flash-next-unsloth-ud-q5_k_xl"
TUNED_BYTES = b"tuned-pack-" * 200_000                      # 2.2 MB, stands in for the 340 MB file
STOCK_BYTES = b"s" * (2 << 20)


def flag(args, name):
    return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else None


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        for sub in ("dflash2", "mtp/rt", "packs/tag"):
            (self.data / sub).mkdir(parents=True)
        (self.data / "dflash2" / drafting.VOCAB_NAME).write_bytes(b"v" * 100)
        (self.data / "mtp" / "rt" / "experts.bin").write_bytes(b"e" * 100)
        (self.data / "mtp" / "rt" / drafting.VOCAB_NAME).write_bytes(b"v" * 100)
        self.exe = self.root / "strata"
        self.exe.write_bytes(b"--mtp-draft-vocab FILE (or --mtp DIR ... --draft-fallback ... selector codebooks: Q4_0 rows")   # a Metal engine of this tree
        self.args = ["--native", "m.gguf", "--pack", str(self.data / "packs" / "tag"), "--mtp", str(self.data / "mtp" / "rt"), "--spec", "2", "--max-context", "65536"]
        self.pins = mock.patch.multiple(drafting, TUNED_BYTES=len(TUNED_BYTES), TUNED_SHA256=hashlib.sha256(TUNED_BYTES).hexdigest())
        self.pins.start()
        self.addCleanup(self.pins.stop)
        patcher = mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def put_tuned(self, content=TUNED_BYTES):
        (self.data / "dflash2" / drafting.TUNED_NAME).write_bytes(content)

    def put_stock(self):
        (self.data / "dflash2" / drafting.PACK_NAME).write_bytes(STOCK_BYTES)

    def cfg(self, name=Q4, **extra):
        return {"model_name": name, "args": list(self.args), **extra}

    def apply(self, cfg, backend="metal", **kw):
        decision = drafting.apply(cfg, data_dir=self.data, backend=backend, archs=[], engine=self.exe, **kw)
        return decision, cfg["args"]


class MetalPerSet(Base):
    def test_the_two_4bit_sets_start_dflash2_with_the_tuned_pack_at_0_7(self):
        self.put_tuned()
        self.put_stock()
        for name in (Q4, V31):
            for mode in (None, "auto"):
                d, args = self.apply(self.cfg(name, **({} if mode is None else {"drafting": mode})))
                where = (name, mode)
                self.assertEqual((d.drafter, d.note), ("dflash2", ""), where)
                self.assertEqual(flag(args, "--draft"), "dflash2", where)
                self.assertEqual(flag(args, "--draft-model"), str(drafting.tuned_path(self.data)), where)
                self.assertEqual(flag(args, "--draft-conf"), "0.7", where)
                self.assertEqual(flag(args, "--spec"), "6", where)
                self.assertEqual(flag(args, "--draft-fallback"), "none", where)
                self.assertNotIn("--mtp", args, where)
                self.assertNotIn("--spec-min-p", args, where)

    def test_the_stock_pack_is_the_fallback_at_the_conf_it_was_measured_with(self):
        self.put_stock()                                      # the tuned pack was never fetched
        for name in (Q4, V31):
            d, args = self.apply(self.cfg(name))
            self.assertEqual(d.drafter, "dflash2", name)
            self.assertEqual(flag(args, "--draft-model"), str(drafting.pack_path(self.data, self.exe)), name)
            self.assertNotEqual(flag(args, "--draft-model"), str(drafting.tuned_path(self.data)), name)
            self.assertEqual((flag(args, "--draft-conf"), flag(args, "--spec")), ("0.5", "6"), name)

    def test_offline_and_nothing_fetched_is_the_mtp_exactly_as_today(self):
        for name in (Q4, V31):
            cfg = self.cfg(name)
            before = list(cfg["args"])
            d, args = self.apply(cfg)
            self.assertEqual((d.drafter, d.note), ("mtp", drafting.NOT_DOWNLOADED), name)
            self.assertEqual(args, before, "the MTP arguments are untouched")
            self.assertFalse([a for a in args if a.startswith("--draft")])

    def test_no_mtp_layer_either_is_off_with_a_note(self):
        (self.data / "mtp" / "rt" / "experts.bin").unlink()
        d, args = self.apply(self.cfg(Q4))
        self.assertEqual((d.drafter, d.note), ("off", drafting.NONE_HERE))

    def test_a_wrong_tuned_file_is_not_used(self):
        self.put_tuned(b"x" * len(TUNED_BYTES))               # the pinned size, other bytes
        self.assertFalse(drafting.tuned_ok(self.data))
        d, args = self.apply(self.cfg(Q4))
        self.assertEqual(d.drafter, "mtp", "nothing else here: the MTP")
        self.put_stock()
        d, args = self.apply(self.cfg(Q4))
        self.assertEqual((d.drafter, flag(args, "--draft-conf")), ("dflash2", "0.5"), "the stock pack runs at its own conf")
        self.assertNotEqual(flag(args, "--draft-model"), str(drafting.tuned_path(self.data)))
        self.put_tuned(TUNED_BYTES[:-1])                      # a short file
        self.assertFalse(drafting.tuned_ok(self.data))

    def test_the_marker_is_not_trusted_for_a_changed_file(self):
        self.put_tuned()
        self.assertTrue(drafting.tuned_ok(self.data))
        self.assertTrue((self.data / "dflash2" / (drafting.TUNED_NAME + ".ok")).is_file())
        self.put_tuned(b"y" * len(TUNED_BYTES))
        self.assertFalse(drafting.tuned_ok(self.data))

    def test_the_users_own_draft_conf_and_spec_are_respected(self):
        self.put_tuned()
        d, args = self.apply(self.cfg(Q4, user_args=["--draft-conf", "0.4"]))
        self.assertEqual((flag(args, "--draft-conf"), flag(args, "--spec")), ("0.4", "6"))
        mine = ["--draft-conf", "0.9", "--spec", "4"]
        cfg = self.cfg(V31, user_args=mine)
        cfg["args"] = [x for x in cfg["args"] if x not in ("--spec", "2")] + mine         # (serve/server.py merges user_args into the arguments before drafting.apply)
        d, args = self.apply(cfg)
        self.assertEqual((flag(args, "--draft-conf"), flag(args, "--spec")), ("0.9", "4"))
        self.assertEqual(flag(args, "--draft-model"), str(drafting.tuned_path(self.data)))

    def test_iq3_s_stays_on_the_mtp_until_its_table_entry_changes(self):
        self.put_tuned()
        self.put_stock()
        for mode in (None, "auto"):
            d, args = self.apply(self.cfg(IQ3S, **({} if mode is None else {"drafting": mode})))
            self.assertEqual(d.drafter, "mtp", mode)
            self.assertFalse([a for a in args if a.startswith("--draft")])
        d, args = self.apply(self.cfg(IQ3S, drafting="dflash2"))
        self.assertEqual((d.drafter, flag(args, "--draft-conf")), ("dflash2", "0.5"), "named by the person: the stock pack at the Mac's 0.5")
        with mock.patch.dict(drafting.SET_DEFAULTS, {("metal", "iq3_s"): drafting.SetDefault("dflash2")}):     # the one entry, switched
            d, args = self.apply(self.cfg(IQ3S))
            self.assertEqual((d.drafter, flag(args, "--draft-conf")), ("dflash2", "0.5"))
            self.assertEqual(flag(args, "--draft-model"), str(drafting.pack_path(self.data, self.exe)), "no tuned entry: the stock pack")
        with mock.patch.dict(drafting.SET_DEFAULTS, {("metal", "unsloth-ud-q4_k_xl"): drafting.SetDefault("mtp")}):    # and the other way
            d, args = self.apply(self.cfg(Q4))
            self.assertEqual(d.drafter, "mtp")

    def test_a_set_with_no_entry_keeps_the_general_rule(self):
        self.put_stock()
        d, args = self.apply(self.cfg(Q5))
        self.assertEqual((d.drafter, flag(args, "--draft-conf")), ("dflash2", "0.5"))
        self.assertIsNone(drafting.default_for("metal", Q5))

    def test_the_tuned_pack_needs_an_engine_that_reads_the_q4_selector(self):
        self.put_tuned()
        old = self.root / "strata-old"
        old.write_bytes(b"--mtp-draft-vocab FILE (or --mtp DIR ... --draft-fallback")      # no selector text
        cfg = self.cfg(Q4)
        d = drafting.apply(cfg, data_dir=self.data, backend="metal", archs=[], engine=old)
        self.assertEqual(d.drafter, "mtp", "only the tuned pack is here and this engine cannot read it")
        self.assertEqual(d.note, drafting.NOT_DOWNLOADED)
        self.put_stock()
        cfg = self.cfg(Q4)
        d = drafting.apply(cfg, data_dir=self.data, backend="metal", archs=[], engine=old)
        self.assertEqual((d.drafter, flag(cfg["args"], "--draft-model")), ("dflash2", str(drafting.pack_path(self.data, old))))
        self.assertNotEqual(flag(cfg["args"], "--draft-model"), str(drafting.tuned_path(self.data)))

    def test_other_backends_are_untouched(self):
        self.put_tuned()
        self.put_stock()
        for backend, archs in (("cuda", ["sm_120"]), ("hip", ["gfx1151"])):
            cfg = self.cfg(Q4)
            drafting.apply(cfg, data_dir=self.data, backend=backend, archs=archs, engine=self.exe)
            self.assertNotEqual(flag(cfg["args"], "--draft-model"), str(drafting.tuned_path(self.data)), backend)
            self.assertNotEqual(flag(cfg["args"], "--draft-conf"), "0.7", backend)
        self.assertIsNone(drafting.tuned_conf_for("cuda", Q4))
        self.assertEqual(drafting.tuned_conf_for("metal", Q4), 0.7)

    def test_needs_tuned_and_needs_pack(self):
        kw = dict(model_name=Q4, backend="metal", archs=[], data_dir=self.data)
        self.assertTrue(drafting.needs_pack("auto", **kw))
        self.assertTrue(drafting.needs_tuned("auto", **kw))
        self.put_stock()
        self.assertFalse(drafting.needs_pack("auto", **kw))
        self.assertTrue(drafting.needs_tuned("auto", **kw), "the stock pack is here, the tuned one is still worth fetching")
        self.put_tuned()
        self.assertFalse(drafting.needs_tuned("auto", **kw))
        self.assertFalse(drafting.needs_tuned("auto", **{**kw, "model_name": IQ3S}))
        self.assertFalse(drafting.needs_tuned("mtp", **kw))
        self.assertFalse(drafting.needs_tuned("auto", **{**kw, "backend": "cuda"}))

    def test_the_metal_line_per_set(self):
        h = drafting.hardware_support
        for name in (Q4, V31):
            s = h("metal", [], name)
            self.assertEqual((s.can_run, s.level, s.line), (True, "tested", drafting.LINE_METAL), name)
        for name in (IQ3S, Q5, None):
            s = h("metal", [], name)
            self.assertEqual((s.can_run, s.level, s.line), (True, "untested", drafting.LINE_UNTESTED), name)
        self.assertEqual(h("metal", []).line, drafting.LINE_UNTESTED, "the old two-argument call")
        self.assertNotIn("untested", drafting.LINE_METAL)


class Fetch(Base):
    def setUp(self):
        super().setUp()
        self.host = self.root / "host"
        self.host.mkdir()
        (self.host / drafting.TUNED_NAME).write_bytes(TUNED_BYTES)
        self.url = (self.host / drafting.TUNED_NAME).as_uri()
        env = {k: v for k, v in __import__("os").environ.items() if k not in (drafting.TUNED_URL_ENV, "STRATA_UPDATE_FEED", "STRATA_UPDATE_TRUSTED_KEYS")}
        env["STRATA_UPDATE_FEED"] = str(self.root / "no-feed-here")       # nothing in these tests reaches the network
        patcher = mock.patch.dict("os.environ", env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.keys = self.root / "keys"
        self.pub = ft.keygen(self.keys, allow_git=True)
        (self.root / "trust.json").write_text((self.keys / "feed-key.pub.json").read_text(encoding="utf-8"), encoding="utf-8")
        __import__("os").environ["STRATA_UPDATE_TRUSTED_KEYS"] = str(self.root / "trust.json")

    def make_feed(self, sub="feed", listed=True, backends=("metal",), pin_sha=None):
        """A local, test-signed release folder: the feed and the drafter beside it as a flat asset (what the GitHub release would hold)."""
        out = self.root / sub
        entries = self.root / (sub + "-entries")
        if listed:
            entry = ft.pack_drafter(self.host / drafting.TUNED_NAME, entries, version="1", backends=list(backends))
            if pin_sha:
                entry["sha256"] = pin_sha
                (entries / (drafting.TUNED_NAME + ".entry.json")).write_text(json.dumps(entry), encoding="utf-8")
        out.mkdir(parents=True, exist_ok=True)
        if listed:
            (out / drafting.TUNED_NAME).write_bytes(TUNED_BYTES)
        ft.build_feed(out, key=ft._load_key(self.keys / "feed-key.json"), sequence=1, drafters_dir=entries if listed else None)
        return out

    def test_a_fetch_from_a_file_address_checks_places_and_marks_the_file_and_copies_the_licence(self):
        seen = []
        out = drafting_fetch.ensure_tuned(self.data, url=self.url, progress=lambda *a: seen.append(a))
        self.assertEqual(out, drafting.tuned_path(self.data))
        self.assertTrue(drafting.tuned_ok(self.data))
        self.assertEqual(seen[-1][1:], (len(TUNED_BYTES), len(TUNED_BYTES)))
        self.assertFalse((self.data / "dflash2" / (drafting.TUNED_NAME + ".part")).exists())
        for name in ("LICENSE-dflash2-tuned", "NOTICE-dflash2-tuned.md"):
            self.assertTrue((self.data / "dflash2" / name).is_file(), name)
        licence = (self.data / "dflash2" / "LICENSE-dflash2-tuned").read_text(encoding="utf-8")
        notice = (self.data / "dflash2" / "NOTICE-dflash2-tuned.md").read_text(encoding="utf-8")
        self.assertIn("Apache License", licence)
        self.assertIn("Version 2.0", licence)
        for word in ("modified copy", "a22a1744e85c17364dcc54a6a420af7135d4c7f6", "Inco AI", "Strata changed"):
            self.assertIn(word, notice)
        again = drafting_fetch.ensure_tuned(self.data, url="file:///no/such/file")     # already here: no download
        self.assertEqual(again, out)

    def test_the_normal_way_is_the_signed_feed(self):
        feed = self.make_feed()
        seen = []
        out = drafting_fetch.ensure_tuned(self.data, feed=str(feed), progress=lambda *a: seen.append(a))
        self.assertEqual(out, drafting.tuned_path(self.data))
        self.assertTrue(drafting.tuned_ok(self.data))
        self.assertEqual(out.read_bytes(), TUNED_BYTES)
        self.assertTrue((self.data / "dflash2" / "LICENSE-dflash2-tuned").is_file())
        # the environment names the feed too, and ensure_for takes the tuned pack from it without touching the stock download
        out.unlink()
        out.with_name(out.name + ".ok").unlink()
        with mock.patch.dict("os.environ", {"STRATA_UPDATE_FEED": str(feed)}), mock.patch.object(drafting_fetch, "ensure", side_effect=AssertionError("stock fetched")):
            self.assertEqual(drafting_fetch.ensure_for(self.data, model_name=Q4, backend="metal"), out)

    def test_the_feed_entry_is_a_drafters_member_and_older_clients_ignore_it(self):
        feed = self.make_feed()
        doc = json.loads((feed / "strata-feed.json").read_text(encoding="utf-8"))
        entry = doc["drafters"]["packages"][0]
        self.assertEqual((entry["name"], entry["backends"], entry["size"], entry["sha256"]),
                         (drafting.TUNED_NAME, ["metal"], len(TUNED_BYTES), hashlib.sha256(TUNED_BYTES).hexdigest()))
        self.assertEqual(entry["url"], drafting.TUNED_NAME, "a flat release asset")
        parsed = drafting_fetch.feedmod.parse((feed / "strata-feed.json").read_bytes(), source=str(feed / "strata-feed.json"), key_id="k")
        self.assertEqual([d.name for d in parsed.drafters], [drafting.TUNED_NAME])
        doc.pop("drafters")
        self.assertEqual(drafting_fetch.feedmod.parse(json.dumps(doc).encode(), source="x", key_id="k").drafters, ())

    def test_a_later_per_platform_switch_is_a_table_entry_and_a_feed_backend(self):
        feed = self.make_feed(sub="both", backends=("metal", "cuda"))
        with mock.patch.dict(drafting.SET_DEFAULTS, {("cuda", "unsloth-ud-q4_k_xl"): drafting.SetDefault("dflash2", 0.35)}):
            self.assertEqual(drafting.tuned_conf_for("cuda", Q4), 0.35)
            self.assertEqual(drafting_fetch.ensure_for(self.data, model_name=Q4, backend="cuda", feed=str(feed)), drafting.tuned_path(self.data))
        self.put_stock()
        self.assertEqual(drafting.tuned_conf_for("cuda", Q4), None, "without the entry CUDA is back to the stock pack")

    def test_a_feed_that_does_not_list_it_or_lists_other_bytes_gives_nothing(self):
        empty = self.make_feed(listed=False)
        self.assertIsNone(drafting_fetch.ensure_tuned(self.data, feed=str(empty)))
        other = self.make_feed(sub="other", pin_sha="0" * 64)
        self.assertIsNone(drafting_fetch.ensure_tuned(self.data, feed=str(other)), "other bytes than the ones this Strata measured")
        elsewhere = self.make_feed(sub="mac-only", backends=["cuda"])
        self.assertIsNone(drafting_fetch.ensure_tuned(self.data, feed=str(elsewhere)))
        self.assertFalse(drafting.tuned_path(self.data).exists())

    def test_a_feed_with_a_bad_signature_is_refused_in_a_line(self):
        feed = self.make_feed()
        text = (feed / "strata-feed.json").read_text(encoding="utf-8")
        (feed / "strata-feed.json").write_text(text.replace('"sequence": 1', '"sequence": 2'), encoding="utf-8")
        with self.assertRaises(drafting_fetch.DraftingError) as caught:
            drafting_fetch.ensure_tuned(self.data, feed=str(feed))
        self.assertIn("could not be trusted", str(caught.exception))
        self.assertFalse(drafting.tuned_path(self.data).exists())
        with mock.patch.object(drafting_fetch, "ensure", return_value=Path("/stock")):
            self.assertEqual(drafting_fetch.ensure_for(self.data, model_name=Q4, backend="metal", feed=str(feed)), Path("/stock"), "the stock drafter takes over")

    def test_a_feed_that_cannot_be_read_or_a_build_with_no_key_is_silent(self):
        self.assertIsNone(drafting_fetch.ensure_tuned(self.data, feed=str(self.root / "no-such-feed")))
        feed = self.make_feed()
        with mock.patch.dict("os.environ", {"STRATA_UPDATE_TRUSTED_KEYS": ""}), mock.patch("strata_control.signing.load_trusted_keys", return_value={}):
            self.assertIsNone(drafting_fetch.ensure_tuned(self.data, feed=str(feed)))

    def test_a_launch_that_got_nothing_does_not_ask_again_for_twelve_hours_but_setup_always_asks(self):
        nofeed = str(self.root / "no-feed-here")
        self.assertIsNone(drafting_fetch.ensure_tuned(self.data, feed=nofeed, throttle=True))
        self.assertTrue((self.data / "dflash2" / drafting_fetch.MISS_MARK).is_file())
        with mock.patch.object(drafting_fetch, "tuned_from_feed", side_effect=AssertionError("asked again")):
            self.assertIsNone(drafting_fetch.ensure_tuned(self.data, feed=nofeed, throttle=True), "a launch a moment later: no wait")
        feed = self.make_feed()
        self.assertEqual(drafting_fetch.ensure_tuned(self.data, feed=str(feed)), drafting.tuned_path(self.data), "setup (no throttle) asks and gets it")
        self.assertFalse((self.data / "dflash2" / drafting_fetch.MISS_MARK).exists(), "and the mark is gone")
        old = self.data / "dflash2" / drafting_fetch.MISS_MARK
        old.write_text("x")
        import os
        os.utime(old, (1, 1))
        self.assertFalse(drafting_fetch._recent_miss(self.data), "a mark older than the interval is ignored")

    def test_with_no_feed_named_the_projects_release_feed_is_read(self):
        from strata_cli import update_cmd
        asked = []

        def fetch(source, trusted, **kw):
            asked.append(source)
            raise drafting_fetch.feedmod.FeedError("the update feed could not be read: offline", 6)
        env = {k: v for k, v in __import__("os").environ.items() if k != update_cmd.ENV_FEED}
        with mock.patch.dict("os.environ", env, clear=True), mock.patch("strata_control.feed.fetch", fetch):
            self.assertIsNone(drafting_fetch.ensure_tuned(self.data))
        self.assertEqual(asked, [update_cmd.default_feed()])
        self.assertTrue(update_cmd.default_feed().startswith("https://github.com/Niko1221/Strata/releases/"))

    def test_a_download_that_fails_the_safety_check_is_deleted_and_said(self):
        (self.host / drafting.TUNED_NAME).write_bytes(b"z" * len(TUNED_BYTES))
        with self.assertRaises(drafting_fetch.DraftingError) as caught:
            drafting_fetch.ensure_tuned(self.data, url=self.url)
        self.assertIn("safety check", str(caught.exception))
        self.assertFalse(drafting.tuned_path(self.data).exists())
        self.assertFalse((self.data / "dflash2" / (drafting.TUNED_NAME + ".part")).exists())

    def test_an_interrupted_download_resumes(self):
        partial = self.data / "dflash2" / (drafting.TUNED_NAME + ".part")
        partial.write_bytes(TUNED_BYTES[:123_456])
        out = drafting_fetch.ensure_tuned(self.data, url=self.url)
        self.assertTrue(drafting.tuned_ok(self.data))
        self.assertEqual(out.read_bytes(), TUNED_BYTES)
        partial.write_bytes(TUNED_BYTES[:99])
        out.unlink()
        calls = []
        real = transfer._transfer

        def spy(location, part, have, size, progress, cancel, timeout):
            calls.append(have)
            return real(location, part, have, size, progress, cancel, timeout)
        with mock.patch.object(transfer, "_transfer", spy):
            drafting_fetch.ensure_tuned(self.data, url=self.url)
        self.assertEqual(calls, [99], "it continued at byte 99, not from the start")

    def test_no_internet_is_silent(self):
        said = []
        with mock.patch.object(drafting_fetch, "online", return_value=False):
            self.assertIsNone(drafting_fetch.ensure_tuned(self.data, url="https://example.invalid/x.pack", progress=lambda *a: said.append(a)))
            self.assertIsNone(drafting_fetch.ensure_for(self.data, model_name=Q4, backend="metal", progress=lambda *a: said.append(a)))
        self.assertEqual(said, [], "nothing is reported and nothing is written")
        self.assertFalse((self.data / "dflash2" / drafting.TUNED_NAME).exists())
        self.assertFalse((self.data / "dflash2" / (drafting.TUNED_NAME + ".part")).exists())

    def test_a_host_that_answers_with_an_error_page_is_silent_too(self):
        def not_found(*_a, **_k):
            raise transfer.TransferError("the update server answered 404 for x")
        with mock.patch.object(drafting_fetch, "online", return_value=True), mock.patch.object(transfer, "download", not_found):
            self.assertIsNone(drafting_fetch.ensure_tuned(self.data, url="https://example.invalid/x.pack"))

    def test_a_start_never_downloads(self):
        with acquisition.forbidden():
            with self.assertRaises(acquisition.AcquisitionRefused):
                drafting_fetch.ensure_tuned(self.data, url=self.url)

    def test_the_set_gets_the_tuned_pack_and_not_the_stock_download(self):
        with mock.patch.dict("os.environ", {drafting.TUNED_URL_ENV: self.url}), mock.patch.object(drafting_fetch, "ensure", side_effect=AssertionError("stock fetched")):
            got = drafting_fetch.ensure_for(self.data, model_name=V31, backend="metal")
        self.assertEqual(got, drafting.tuned_path(self.data))

    def test_when_the_tuned_pack_cannot_be_had_the_stock_drafter_is_fetched(self):
        with mock.patch.dict("os.environ", {drafting.TUNED_URL_ENV: "file:///no/such/host/file.pack"}), \
                mock.patch.object(drafting_fetch, "ensure", return_value=Path("/stock")) as stock:
            got = drafting_fetch.ensure_for(self.data, model_name=Q4, backend="metal")
        self.assertEqual(got, Path("/stock"))
        stock.assert_called_once()

    def test_a_failed_tuned_check_falls_back_and_only_a_double_failure_is_said(self):
        (self.host / drafting.TUNED_NAME).write_bytes(b"z" * len(TUNED_BYTES))
        with mock.patch.dict("os.environ", {drafting.TUNED_URL_ENV: self.url}), mock.patch.object(drafting_fetch, "ensure", return_value=Path("/stock")):
            self.assertEqual(drafting_fetch.ensure_for(self.data, model_name=Q4, backend="metal"), Path("/stock"))
        (self.host / drafting.TUNED_NAME).write_bytes(b"z" * len(TUNED_BYTES))
        with mock.patch.dict("os.environ", {drafting.TUNED_URL_ENV: self.url}), \
                mock.patch.object(drafting_fetch, "ensure", side_effect=drafting_fetch.OfflineError("no internet")):
            with self.assertRaises(drafting_fetch.DraftingError) as caught:
                drafting_fetch.ensure_for(self.data, model_name=Q4, backend="metal")
        self.assertIn("safety check", str(caught.exception))

    def test_a_stock_pack_already_here_is_kept_when_the_tuned_one_is_offline(self):
        self.put_stock()
        with mock.patch.object(drafting_fetch, "online", return_value=False), mock.patch.object(drafting_fetch, "ensure", side_effect=AssertionError("fetched")):
            got = drafting_fetch.ensure_for(self.data, model_name=Q4, backend="metal")
        self.assertEqual(got, drafting.pack_path(self.data))

    def test_cuda_and_hip_never_read_the_feed_for_a_drafter_and_keep_their_conf_steps(self):
        for backend in ("cuda", "hip", "sycl"):
            with mock.patch.object(drafting_fetch, "tuned_from_feed", side_effect=AssertionError("feed read")),                     mock.patch.object(drafting_fetch, "ensure", return_value=Path("/stock")) as stock:
                self.assertEqual(drafting_fetch.ensure_for(self.data, model_name=Q4, backend=backend), Path("/stock"), backend)
                stock.assert_called_once()
            self.assertFalse(drafting.needs_tuned("auto", model_name=Q4, backend=backend, archs=[], data_dir=self.data), backend)
        self.assertEqual(drafting.CONF_STEPS, ((12000, 0.35), (7500, 0.5), (3000, 0.7), (0, 0.8)))
        self.assertEqual(drafting.BACKEND_CONF, {"sycl": 0.6, "metal": 0.5})
        self.assertEqual({k for k in drafting.SET_DEFAULTS}, {("metal", "unsloth-ud-q4_k_xl"), ("metal", "unsloth-ud-iq4_xs-v31"), ("metal", "iq3_s")})

    def test_a_set_without_a_tuned_entry_is_fetched_as_before(self):
        with mock.patch.object(drafting_fetch, "ensure", return_value=Path("/stock")) as stock:
            self.assertEqual(drafting_fetch.ensure_for(self.data, model_name=IQ3S, backend="metal"), Path("/stock"))
            self.assertEqual(drafting_fetch.ensure_for(self.data, model_name=Q4, backend="cuda"), Path("/stock"))
        self.assertEqual(stock.call_count, 2)


class SetupOnAMac(Base):
    """metal/setup_mac.fetch_drafter: the line appears when a download starts, never offline; the MTP stays the fallback."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location("t_setup_mac", ROOT / "metal" / "setup_mac.py")
        cls.mac = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("t_setup_mac", cls.mac)
        try:
            spec.loader.exec_module(cls.mac)
        except SystemExit:
            raise unittest.SkipTest("setup.py would not import here")

    def setUp(self):
        super().setUp()
        self.host = self.root / "host"
        self.host.mkdir()
        (self.host / drafting.TUNED_NAME).write_bytes(TUNED_BYTES)
        self.said = []
        for name in ("say", "ok", "warn"):
            patcher = mock.patch.object(self.mac.S, name, lambda msg="", _n=name: self.said.append((_n, msg)))
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_it_fetches_the_tuned_drafter_with_one_line_before_and_one_after(self):
        with mock.patch.dict("os.environ", {drafting.TUNED_URL_ENV: (self.host / drafting.TUNED_NAME).as_uri()}):
            got = self.mac.fetch_drafter(Q4, self.data)
        self.assertEqual(got, drafting.tuned_path(self.data))
        self.assertEqual([k for k, _ in self.said], ["say", "ok"])
        self.assertIn("Downloading the DFlash2 drafter (0.34 GB, once)", self.said[0][1])

    def test_offline_it_says_nothing(self):
        with mock.patch.object(drafting_fetch, "online", return_value=False):
            self.assertIsNone(self.mac.fetch_drafter(Q4, self.data))
        self.assertEqual(self.said, [])

    def test_iq3_s_and_other_sets_fetch_nothing(self):
        with mock.patch.object(drafting_fetch, "ensure_for", side_effect=AssertionError("fetched")):
            self.assertIsNone(self.mac.fetch_drafter(IQ3S, self.data))
        self.assertEqual(self.said, [])

    def test_a_failed_safety_check_is_one_warning_and_the_setup_goes_on(self):
        (self.host / drafting.TUNED_NAME).write_bytes(b"z" * len(TUNED_BYTES))
        with mock.patch.dict("os.environ", {drafting.TUNED_URL_ENV: (self.host / drafting.TUNED_NAME).as_uri()}), \
                mock.patch.object(drafting_fetch, "ensure", side_effect=drafting_fetch.OfflineError("no internet")):
            self.assertIsNone(self.mac.fetch_drafter(Q4, self.data))
        self.assertEqual([k for k, _ in self.said], ["say", "warn"])
        self.assertIn("safety check", self.said[1][1])
        self.assertIn("keeps using the MTP", self.said[1][1])


class Pins(unittest.TestCase):
    def test_the_pinned_numbers_are_the_final_pack(self):
        self.assertEqual(drafting.TUNED_BYTES, 339694592)
        self.assertEqual(drafting.TUNED_BYTES, drafting.CONVERTED_BYTES, "the same format and size as the stock Q4_0-selector pack")
        self.assertEqual(drafting.TUNED_SHA256, "4df2e5ab2b47c064afbe662e16e4f06569877125c2d0e2123a4ac755266c664c")
        self.assertNotEqual(drafting.TUNED_SHA256, drafting.CONVERTED_SHA256)
        self.assertNotEqual(drafting.TUNED_NAME, drafting.CONVERTED_NAME)

    def test_the_licence_files_ship_in_data(self):
        folder = ROOT / drafting.TUNED_LICENCE_FOLDER
        self.assertIn("Apache License", (folder / "LICENSE").read_text(encoding="utf-8"))
        notice = (folder / "NOTICE.md").read_text(encoding="utf-8")
        self.assertIn(drafting.TUNED_SHA256, notice)
        self.assertIn("a22a1744e85c17364dcc54a6a420af7135d4c7f6", notice)
        self.assertIn("dflash2-tuned", (ROOT / "docs" / "DETAILS.md").read_text(encoding="utf-8"), "the third-party notices name it")

    def test_the_base_revision_is_the_one_the_stock_drafter_is_pinned_to(self):
        self.assertIn(drafting.REVISION, drafting.TUNED_BASE)

    def test_set_names(self):
        self.assertEqual(drafting.set_of(Q4), "unsloth-ud-q4_k_xl")
        self.assertEqual(drafting.set_of(V31), "unsloth-ud-iq4_xs-v31")
        self.assertEqual(drafting.set_of("Qwen3.8-Flash-Next-IQ3_S"), "iq3_s")
        self.assertEqual(drafting.set_of(None), "")
        self.assertTrue(drafting.supported_model(V31), "the v31 name passes the supported-model pattern")


if __name__ == "__main__":
    unittest.main()
