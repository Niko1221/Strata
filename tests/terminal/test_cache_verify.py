"""tools/cache_verify.py: the offline recompute-gate verifier (sidecar v1/v2, the ledger hash and format, sample
choice, digest comparison, atomic ledger update, and the CLI against mocked engine/digest subprocesses)."""
from __future__ import annotations

import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import cache_verify as CV  # noqa: E402

MODEL = 0x92015B27A6E3EFB8
OLD_CONFIG = 0x3074D88348B69B80
NEW_CONFIG = 0xD88978F614302157


def sidecar_bytes(version=2, model=MODEL, config=OLD_CONFIG, ids=(1, 2, 3), images=(), checkpoints=(2,),
                  last_used=1793037175538, corrupt=False):
    body = b"SCSM" + struct.pack("<QQQQQ", version, model, config, 1, len(ids))
    for token in ids:
        body += struct.pack("<q", token)
    body += struct.pack("<Q", len(images))
    for start, key in images:
        body += struct.pack("<qQ", start, key)
    body += struct.pack("<Q", len(checkpoints))
    for length in checkpoints:
        body += struct.pack("<Q", length)
    if version >= 2:
        body += struct.pack("<Q", last_used)
    data = body + struct.pack("<Q", CV.session_hash64(body, 0))
    if corrupt:
        data = data[:-1] + bytes([data[-1] ^ 0xFF])
    return data


def digest(tokens=1, kv=2, state=3, exact=True, ref=False, rht=False):
    return CV.Digest(3, tokens, kv, state, exact, ref, rht)


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def cache_file(self, stem, size, version=2, config=OLD_CONFIG, ids=(1, 2, 3), images=(), checkpoints=(2,),
                   write_meta=True, dlog=False):
        (self.root / f"{stem}.sess").write_bytes(b"x" * size)
        if write_meta:
            (self.root / f"{stem}.meta").write_bytes(
                sidecar_bytes(version=version, config=config, ids=ids, images=images, checkpoints=checkpoints))
        if dlog:
            (self.root / f"{stem}.dlog").write_bytes(b"delta")
        return self.root / f"{stem}.sess"


class Hash(TempDir):
    def test_matches_the_cpp_implementation(self):
        # produced by the C++ SessionHasher (src/core/conversation_file.cpp) in a throwaway harness
        vectors = [(b"", 0, "3fdf455f9dcf1e62"), (b"", 1, "1157986175266e71"), (b"a", 0, "722f437eb72ecc23"),
                   (b"abc", 0, "c6286258a8ea2c6a"), (b"x" * 31, 0, "282b32bacd563b74"),
                   (b"x" * 32, 0, "e2df261fc2ec30eb"), (b"x" * 33, 0, "b3fa465f554208a6"),
                   (b"x" * 64, 0, "a97161d5402f516e"),
                   (b"strata-cache-gate 1\nbatch b1 0x0000000000000001 0x0000000000000002 0x0000000000000001 "
                    b"0x0000000000000003 0.1.52.20 s1 pass 123 -", 0, "c6e1a559c896059f")]
        for data, seed, want in vectors:
            self.assertEqual(f"{CV.session_hash64(data, seed):016x}", want, data)

    def test_streaming_equals_one_shot(self):
        data = b"q" * 1000
        hasher = CV.SessionHasher(0)
        for at in range(0, len(data), 7):
            hasher.update(data[at:at + 7])
        self.assertEqual(hasher.digest(), CV.session_hash64(data, 0))


class Sidecar(TempDir):
    def test_v2_roundtrip(self):
        parsed = CV.parse_sidecar(sidecar_bytes())
        self.assertEqual((parsed.version, parsed.model, parsed.config, parsed.cvec), (2, MODEL, OLD_CONFIG, True))
        self.assertEqual(parsed.ids, (1, 2, 3))
        self.assertEqual(parsed.checkpoints, (2,))
        self.assertEqual(parsed.last_used, 1793037175538)

    def test_v1_has_no_last_use_stamp(self):
        parsed = CV.parse_sidecar(sidecar_bytes(version=1))
        self.assertEqual(parsed.version, 1)
        self.assertEqual(parsed.last_used, 0)
        self.assertEqual(parsed.ids, (1, 2, 3))

    def test_images_are_parsed(self):
        parsed = CV.parse_sidecar(sidecar_bytes(images=((1, 99),)))
        self.assertEqual(parsed.images, ((1, 99),))

    def test_bad_checksum_is_refused(self):
        with self.assertRaises(CV.SidecarError):
            CV.parse_sidecar(sidecar_bytes(corrupt=True))

    def test_truncation_and_bad_magic_are_refused(self):
        with self.assertRaises(CV.SidecarError):
            CV.parse_sidecar(sidecar_bytes()[:-8])
        with self.assertRaises(CV.SidecarError):
            CV.parse_sidecar(b"XCSM" + sidecar_bytes()[4:])

    def test_unsupported_version_is_refused(self):
        with self.assertRaises(CV.SidecarError):
            CV.parse_sidecar(sidecar_bytes(version=3))

    def test_invalid_chains_are_refused(self):
        with self.assertRaises(CV.SidecarError):
            CV.parse_sidecar(sidecar_bytes(checkpoints=(9,)))            # past the live ids
        with self.assertRaises(CV.SidecarError):
            CV.parse_sidecar(sidecar_bytes(images=((5, 1),)))           # image start past the ids
        with self.assertRaises(CV.SidecarError):
            CV.parse_sidecar(sidecar_bytes(ids=()))                     # no tokens: nothing to replay
        with self.assertRaises(CV.SidecarError):
            CV.parse_sidecar(sidecar_bytes(ids=(-1,)))                  # outside the int32 token range

    def test_reads_from_disk(self):
        path = self.root / "a.meta"
        path.write_bytes(sidecar_bytes())
        self.assertEqual(CV.read_sidecar(path).ids, (1, 2, 3))


class Scan(TempDir):
    def test_groups_and_skips(self):
        self.cache_file("strata-conv-a", 100)
        smallest = self.cache_file("strata-conv-b", 50)
        self.cache_file("strata-conv-same", 60, config=NEW_CONFIG)
        self.cache_file("strata-conv-img", 70, images=((1, 5),))
        self.cache_file("strata-conv-chain", 80, dlog=True)
        self.cache_file("strata-conv-nometa", 90, write_meta=False)
        self.cache_file("strata conv space", 40)
        self.cache_file("strata-conv-v1", 55, version=1, ids=(4, 5, 6, 7))
        files = CV.scan_cache_dir(self.root)
        eligible = [f for f in files if not f.skip]
        self.assertEqual([f.stem for f in eligible], ["strata-conv-a", "strata-conv-b", "strata-conv-same",
                                                      "strata-conv-v1"])
        skipped = {f.stem: f.skip for f in files if f.skip}
        self.assertIn("images", skipped["strata-conv-img"])
        self.assertIn("delta chain", skipped["strata-conv-chain"])
        self.assertIn("sidecar is missing", skipped["strata-conv-nometa"])
        self.assertIn("whitespace", skipped["strata conv space"])
        legacy = CV.group_by_source(eligible)[CV.Identity(MODEL, OLD_CONFIG)]
        self.assertEqual(CV.choose_sample(legacy).stem, "strata-conv-b")
        self.assertEqual({f.stem for f in legacy}, {"strata-conv-a", "strata-conv-b", "strata-conv-v1"})
        self.assertEqual(len(CV.group_by_source(eligible)), 2)

    def test_a_stem_that_is_not_utf8_is_skipped(self):
        # os.listdir surfaces undecodable name bytes as lone surrogates; the ledger is UTF-8, so such a file can
        # never be named in one and must be skipped rather than crash the write in format_ledger.
        raw = os.fsencode(self.root) + b"/strata-conv-\xff"
        with open(raw + b".sess", "wb") as out:
            out.write(b"x" * 10)
        with open(raw + b".meta", "wb") as out:
            out.write(sidecar_bytes())
        files = CV.scan_cache_dir(self.root)
        self.assertEqual([f.skip for f in files],
                         ["stem is not valid UTF-8 (cannot be written to the ledger)"])
        self.assertEqual(CV.group_by_source([f for f in files if not f.skip]), {})


class Comparison(unittest.TestCase):
    def sidecar(self, checkpoints=(2,), exact=True):
        return CV.Sidecar(2, MODEL, OLD_CONFIG, True, (1, 2, 3), (), checkpoints, 0)

    def test_equal_digests_pass(self):
        self.assertEqual(CV.compare_digests(digest(), digest(), self.sidecar(checkpoints=())), "")

    def test_kv_always_counts(self):
        self.assertEqual(CV.compare_digests(digest(), digest(kv=9), self.sidecar()), "kv_mismatch")

    def test_live_only_token_mismatch_fails(self):
        self.assertEqual(CV.compare_digests(digest(), digest(tokens=9), self.sidecar(checkpoints=())),
                         "tokens_mismatch")

    def test_live_only_state_mismatch_fails(self):
        self.assertEqual(CV.compare_digests(digest(), digest(state=9), self.sidecar(checkpoints=())),
                         "state_mismatch")

    def test_rht_state_is_not_compared(self):
        self.assertEqual(CV.compare_digests(digest(), digest(state=9, exact=False), self.sidecar(checkpoints=())),
                         "")

    def test_a_checkpoint_bearing_file_compares_kv_only(self):
        # a fresh --session-out export holds no checkpoint, so its token/state digests are not comparable
        self.assertEqual(CV.compare_digests(digest(), digest(tokens=9, state=9), self.sidecar()), "")


class DigestParsing(unittest.TestCase):
    TEXT = ("version=2\ntokens=0000000000000001\nkv=0000000000000002\nstate=0000000000000003\n"
            "state_exact=1\nstate_ref=0\nstate_rht=0\n")

    def test_parses_the_documented_keys(self):
        parsed = CV.parse_digest(self.TEXT)
        self.assertEqual((parsed.version, parsed.tokens, parsed.kv, parsed.state), (2, 1, 2, 3))
        self.assertTrue(parsed.state_exact)
        self.assertFalse(parsed.state_rht)

    def test_missing_and_unknown_keys_are_refused(self):
        with self.assertRaises(CV.VerifyError) as caught:
            CV.parse_digest(self.TEXT.replace("state_rht=0\n", ""))
        self.assertEqual(caught.exception.reason, "digest_failed")
        with self.assertRaises(CV.VerifyError):
            CV.parse_digest(self.TEXT + "extra=1\n")


class FreshLog(unittest.TestCase):
    def test_identity_and_token_count(self):
        text = ("strata generate: --session-out: wrote 7175 tokens and 60 K/V layers (model 92015b27a6e3efb8, "
                "config d88978f614302157) to /tmp/x.sess, 237000000 bytes\n")
        identities, counts = CV.parse_fresh_log(text)
        self.assertEqual(identities, [CV.Identity(MODEL, NEW_CONFIG)])
        self.assertEqual(counts, [7175])

    def test_ambiguity_is_visible(self):
        text = ("(model 92015b27a6e3efb8, config d88978f614302157)\n"
                "(model 92015b27a6e3efb8, config 3074d88348b69b80)\n")
        identities, _ = CV.parse_fresh_log(text)
        self.assertEqual(len(set(identities)), 2)


class Ledger(TempDir):
    def batch(self, source=CV.Identity(MODEL, OLD_CONFIG), target=CV.Identity(MODEL, NEW_CONFIG), passed=True):
        return CV.GateBatch("b" + "a" * 16, source, target, "0.1.52.20", "strata-conv-1", passed,
                            1793037175538, "" if passed else "kv_mismatch")

    def test_roundtrip_write_and_read(self):
        entry = CV.GateEntry("strata-conv-1", "b" + "a" * 16, 249000000, 8170, 1793037175538)
        CV.write_ledger(self.root, [self.batch()], [entry])
        batches, entries = CV.read_ledger(self.root)
        self.assertEqual(batches, [self.batch()])
        self.assertEqual(entries, [entry])
        self.assertFalse((self.root / CV.GATE_TEMP).exists())
        self.assertEqual((self.root / CV.GATE_NAME).stat().st_mode & 0o777, 0o600)

    def test_empty_reason_is_written_as_dash_and_sanitised(self):
        failed = CV.GateBatch("b1", CV.Identity(1, 2), CV.Identity(1, 3), "build", "s", False, 5,
                              "route counts differ")
        text = CV.format_ledger([failed], []).decode()
        self.assertIn(" fail 5 route_counts_differ\n", text)
        passed = CV.GateBatch("b1", CV.Identity(1, 2), CV.Identity(1, 3), "build", "s", True, 5, "")
        self.assertIn(" pass 5 -\n", CV.format_ledger([passed], []).decode())

    def test_a_token_with_u0085_and_u2028_roundtrips(self):
        # std::getline splits on LF only: these characters are ordinary bytes inside a token, not line breaks
        batch = CV.GateBatch("b1", CV.Identity(1, 2), CV.Identity(1, 3), "build", "sample\x85name", True, 5,
                             "reason\u2028here")
        CV.write_ledger(self.root, [batch], [])
        self.assertEqual((self.root / CV.GATE_NAME).read_bytes().count(b"\n"), 3)
        self.assertEqual(CV.read_ledger(self.root), ([batch], []))

    def test_the_closing_newline_ends_the_line_and_is_not_one(self):
        body = "strata-cache-gate 1\n"
        data = body.encode() + f"sum {CV.session_hash64(body.encode(), 0):016x}\n".encode()
        self.assertEqual(CV.parse_ledger(data), ([], []))

    def test_an_empty_line_before_the_checksum_is_refused(self):
        # std::getline yields that empty line, and the engine refuses it as an unknown line
        body = "strata-cache-gate 1\n\n"
        data = body.encode() + f"sum {CV.session_hash64(body.encode(), 0):016x}\n".encode()
        with self.assertRaises(CV.LedgerError):
            CV.parse_ledger(data)

    def test_truncated_edited_and_unknown_ledgers_are_refused(self):
        CV.write_ledger(self.root, [self.batch()], [])
        path = self.root / CV.GATE_NAME
        original = path.read_bytes()
        for broken in (original[:-1], original.replace(b"pass", b"fail"), original.replace(b"strata-cache-gate 1",
                                                                                         b"strata-cache-gate 2"),
                       original + b"junk\n"):
            path.write_bytes(broken)
            with self.assertRaises(CV.LedgerError):
                CV.read_ledger(self.root)

    def test_a_malformed_field_is_refused(self):
        body = ("strata-cache-gate 1\n"
                "entry strata-conv-1 b1 12 NOTANUMBER 0\n")
        data = body.encode() + f"sum {CV.session_hash64(body.encode(), 0):016x}\n".encode()
        with self.assertRaises(CV.LedgerError):
            CV.parse_ledger(data)

    def test_merge_replaces_only_the_processed_source(self):
        old_batch = CV.GateBatch("bold", CV.Identity(MODEL, OLD_CONFIG), CV.Identity(MODEL, NEW_CONFIG),
                                 "0.1.52.20", "strata-conv-1", True, 1)
        other = CV.GateBatch("bother", CV.Identity(7, 8), CV.Identity(7, 9), "build", "strata-conv-9", True, 2)
        entries = [CV.GateEntry("strata-conv-1", "bold", 10, 3, 1),
                   CV.GateEntry("strata-conv-9", "bother", 20, 4, 2)]
        new_batch = self.batch()
        result = CV.BatchResult(CV.Identity(MODEL, OLD_CONFIG), "pass", new_batch,
                                [CV.GateEntry("strata-conv-1", new_batch.id, 11, 5, 3)])
        batches, kept = CV.merge_ledger([old_batch, other], entries, [result])
        self.assertEqual(batches, [other, new_batch])
        self.assertEqual(kept, [CV.GateEntry("strata-conv-9", "bother", 20, 4, 2),
                                CV.GateEntry("strata-conv-1", new_batch.id, 11, 5, 3)])

    def test_a_failed_batch_leaves_no_entry_rows(self):
        old_batch = CV.GateBatch("bold", CV.Identity(MODEL, OLD_CONFIG), CV.Identity(MODEL, NEW_CONFIG),
                                 "0.1.52.20", "strata-conv-1", True, 1)
        failed = CV.GateBatch("bnew", CV.Identity(MODEL, OLD_CONFIG), CV.Identity(MODEL, NEW_CONFIG),
                              "0.1.52.20", "strata-conv-1", False, 2, "kv_mismatch")
        result = CV.BatchResult(CV.Identity(MODEL, OLD_CONFIG), "fail", failed, [])
        batches, entries = CV.merge_ledger([old_batch], [CV.GateEntry("strata-conv-1", "bold", 10, 3, 1)], [result])
        self.assertEqual(batches, [failed])
        self.assertEqual(entries, [])

    def test_a_failed_run_without_target_drops_old_rows(self):
        old_batch = CV.GateBatch("bold", CV.Identity(MODEL, OLD_CONFIG), CV.Identity(MODEL, NEW_CONFIG),
                                 "0.1.52.20", "strata-conv-1", True, 1)
        result = CV.BatchResult(CV.Identity(MODEL, OLD_CONFIG), "fail", None, [], "engine_failed")
        batches, entries = CV.merge_ledger([old_batch], [CV.GateEntry("strata-conv-1", "bold", 10, 3, 1)], [result])
        self.assertEqual(batches, [])
        self.assertEqual(entries, [])

    def test_a_skip_leaves_everything_untouched(self):
        old_batch = CV.GateBatch("bold", CV.Identity(MODEL, OLD_CONFIG), CV.Identity(MODEL, NEW_CONFIG),
                                 "0.1.52.20", "strata-conv-1", True, 1)
        result = CV.BatchResult(CV.Identity(MODEL, OLD_CONFIG), "skip", None, [], "same identity")
        batches, entries = CV.merge_ledger([old_batch], [CV.GateEntry("strata-conv-1", "bold", 10, 3, 1)], [result])
        self.assertEqual(batches, [old_batch])
        self.assertEqual(entries, [CV.GateEntry("strata-conv-1", "bold", 10, 3, 1)])


class CommandLine(TempDir):
    def setUp(self):
        super().setUp()
        self.engine = self.root / "engine"
        self.engine.write_text("#!/bin/sh\n")
        self.engine.chmod(0o755)
        self.digest_tool = self.root / "digest"
        self.digest_tool.write_text("#!/bin/sh\n")
        self.digest_tool.chmod(0o755)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.work = self.root / "work"
        patcher = mock.patch.object(CV, "resolve_executable", side_effect=lambda value, what: value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_main(self, argv, generate, digest_run):
        out = io.StringIO()
        with mock.patch.object(CV, "run_generate", side_effect=generate) as gen, \
                mock.patch.object(CV, "run_digest", side_effect=digest_run), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = CV.main(argv)
        return code, gen, out.getvalue()

    def base_args(self):
        return ["--cache-dir", str(self.cache), "--engine", str(self.engine),
                "--digest-tool", str(self.digest_tool), "--engine-build", "0.1.52.20",
                "--target-model", hex(MODEL), "--target-config", hex(NEW_CONFIG), "--work-dir", str(self.work)]

    def test_a_passing_sample_writes_a_batch_and_every_eligible_row(self):
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 100)
        (self.cache / "strata-conv-a.meta").write_bytes(
            sidecar_bytes(ids=(1, 2, 3, 4, 5), checkpoints=(), last_used=11))
        (self.cache / "strata-conv-b.sess").write_bytes(b"b" * 50)
        (self.cache / "strata-conv-b.meta").write_bytes(
            sidecar_bytes(ids=(1, 2, 3), checkpoints=(), last_used=22))
        def generate(engine, config_args, extra_args, ids, ids_path, out_path, cwd, timeout):
            self.assertEqual(list(ids), [1, 2, 3])           # the smallest file is the sample
            return CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), len(ids))
        def digest_run(tool, path, identity, timeout):
            return digest()
        code, gen, text = self.run_main(self.base_args(), generate, digest_run)
        self.assertEqual(code, 0, text)
        self.assertEqual(gen.call_count, 1)
        batches, entries = CV.read_ledger(self.cache)
        self.assertEqual(len(batches), 1)
        self.assertTrue(batches[0].passed)
        self.assertEqual(batches[0].sample, "strata-conv-b")
        self.assertEqual(batches[0].source, CV.Identity(MODEL, OLD_CONFIG))
        self.assertEqual(batches[0].target, CV.Identity(MODEL, NEW_CONFIG))
        self.assertEqual({(e.stem, e.file_bytes, e.tokens) for e in entries},
                         {("strata-conv-a", 100, 5), ("strata-conv-b", 50, 3)})
        self.assertTrue(all(e.batch_id == batches[0].id for e in entries))

    def test_a_kv_mismatch_writes_a_failed_batch_and_no_rows(self):
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        def generate(*args):
            return CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), 3)
        def digest_run(tool, path, identity, timeout):
            return digest() if path == self.cache / "strata-conv-a.sess" else digest(kv=99)
        code, _, text = self.run_main(self.base_args(), generate, digest_run)
        self.assertEqual(code, 1, text)
        batches, entries = CV.read_ledger(self.cache)
        self.assertEqual(len(batches), 1)
        self.assertFalse(batches[0].passed)
        self.assertEqual(batches[0].reason, "kv_mismatch")
        self.assertEqual(entries, [])

    def test_a_changed_target_identity_refuses_the_whole_run(self):
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        def generate(*args):
            return CV.FreshRun(CV.Identity(MODEL, OLD_CONFIG), 3)
        code, _, _ = self.run_main(self.base_args(), generate, lambda *a: digest())
        self.assertEqual(code, 3)
        self.assertFalse((self.cache / CV.GATE_NAME).exists())

    def test_a_foreign_model_is_skipped_without_running(self):
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(model=MODEL ^ 1))
        code, gen, text = self.run_main(self.base_args(), mock.Mock(), lambda *a: digest())
        self.assertEqual(code, 0, text)
        self.assertEqual(gen.call_count, 0)
        self.assertIn("another model", text)
        self.assertFalse((self.cache / CV.GATE_NAME).exists())

    def test_the_build_is_probed_from_version_when_not_given(self):
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        argv = [a for a in self.base_args() if a not in ("--engine-build", "0.1.52.20")]
        with mock.patch.object(CV, "probe_engine_build", return_value="0.1.52.20") as probe:
            code, _, _ = self.run_main(argv, lambda *a: CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), 3),
                                       lambda *a: digest())
        self.assertEqual(code, 0)
        probe.assert_called_once()

    def test_serve_in_the_passthrough_is_a_usage_error(self):
        code = CV.main(self.base_args() + ["--", "--serve"])
        self.assertEqual(code, 2)

    def test_the_verifier_owns_its_engine_arguments(self):
        for owned in ("--session-out", "--tokens-file", "--tokens", "--max-new", "--serve"):
            with self.assertRaises(CV.UsageError):
                CV.check_engine_args(["--pack", "p", owned])

    def test_the_serve_refusal_does_not_claim_the_verifier_sets_it(self):
        for given in ("--serve", "--serve=8080"):
            with self.assertRaises(CV.UsageError) as caught:
                CV.check_engine_args(["--pack", "p", given])
            self.assertIn("--serve is not allowed", str(caught.exception))
            self.assertNotIn("sets --serve", str(caught.exception))

    def test_a_malformed_ledger_starts_empty_and_unrelated_records_survive(self):
        (self.cache / CV.GATE_NAME).write_bytes(b"strata-cache-gate 1\ngarbage\n")
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        code, _, text = self.run_main(self.base_args(), lambda *a: CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), 3),
                                      lambda *a: digest())
        self.assertEqual(code, 0, text)
        batches, _ = CV.read_ledger(self.cache)
        self.assertEqual(len(batches), 1)
        self.assertTrue(batches[0].passed)

    def test_an_existing_unrelated_batch_is_preserved(self):
        other = CV.GateBatch("bother", CV.Identity(7, 8), CV.Identity(7, 9), "old", "strata-conv-z", True, 1)
        stale = CV.GateBatch("bold", CV.Identity(MODEL, OLD_CONFIG), CV.Identity(MODEL, NEW_CONFIG), "old",
                             "strata-conv-a", False, 2, "kv_mismatch")
        CV.write_ledger(self.cache, [other, stale], [CV.GateEntry("strata-conv-z", "bother", 5, 1, 1)])
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        code, _, _ = self.run_main(self.base_args(), lambda *a: CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), 3),
                                   lambda *a: digest())
        self.assertEqual(code, 0)
        batches, entries = CV.read_ledger(self.cache)
        self.assertIn(other, batches)
        self.assertNotIn(stale, batches)
        self.assertIn(CV.GateEntry("strata-conv-z", "bother", 5, 1, 1), entries)
        self.assertEqual(len(entries), 2)

    def test_dry_run_starts_nothing_and_writes_nothing(self):
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        argv = [a for a in self.base_args() if a not in ("--digest-tool", str(self.digest_tool))]
        code, gen, text = self.run_main(argv + ["--dry-run"], mock.Mock(), mock.Mock())
        self.assertEqual(code, 0, text)
        self.assertEqual(gen.call_count, 0)
        self.assertIn("plan", text)
        self.assertFalse((self.cache / CV.GATE_NAME).exists())

    def test_the_target_is_learned_from_the_fresh_export(self):
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        argv = [a for a in self.base_args()
                if a not in ("--target-model", hex(MODEL), "--target-config", hex(NEW_CONFIG))]
        code, _, text = self.run_main(argv, lambda *a: CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), 3),
                                      lambda *a: digest())
        self.assertEqual(code, 0, text)
        batches, _ = CV.read_ledger(self.cache)
        self.assertEqual(batches[0].target, CV.Identity(MODEL, NEW_CONFIG))

    def test_a_learned_target_of_another_model_fails_each_group(self):
        # a foreign model must not be pinned as the target: the later groups would then be skipped as
        # "another model" and the run would end 0 with a ledger that approves nothing
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(config=OLD_CONFIG, checkpoints=()))
        (self.cache / "strata-conv-b.sess").write_bytes(b"b" * 60)
        (self.cache / "strata-conv-b.meta").write_bytes(sidecar_bytes(config=OLD_CONFIG ^ 1, checkpoints=()))
        argv = [a for a in self.base_args()
                if a not in ("--target-model", hex(MODEL), "--target-config", hex(NEW_CONFIG))]
        code, gen, text = self.run_main(argv, lambda *a: CV.FreshRun(CV.Identity(MODEL ^ 1, NEW_CONFIG), 3),
                                        lambda *a: digest())
        self.assertEqual(code, 1, text)
        self.assertEqual(gen.call_count, 2)              # the second group was verified, not skipped
        self.assertIn("the fresh export is model", text)
        self.assertEqual(CV.read_ledger(self.cache), ([], []))

    def test_two_different_identities_refuse_the_run(self):
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(config=OLD_CONFIG, checkpoints=()))
        (self.cache / "strata-conv-b.sess").write_bytes(b"b" * 60)
        (self.cache / "strata-conv-b.meta").write_bytes(sidecar_bytes(config=OLD_CONFIG ^ 1, checkpoints=()))
        argv = [a for a in self.base_args()
                if a not in ("--target-model", hex(MODEL), "--target-config", hex(NEW_CONFIG))]
        answers = [CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), 3),
                   CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG ^ 1), 3)]
        code, _, _ = self.run_main(argv, lambda *a: answers.pop(0), lambda *a: digest())
        self.assertEqual(code, 3)
        self.assertFalse((self.cache / CV.GATE_NAME).exists())

    def test_an_engine_failure_is_a_failed_batch_with_no_rows(self):
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        def generate(*args):
            raise CV.VerifyError("engine_failed", "exit 1: no model")
        code, _, text = self.run_main(self.base_args(), generate, lambda *a: digest())
        self.assertEqual(code, 1, text)
        batches, entries = CV.read_ledger(self.cache)
        self.assertEqual(len(batches), 1)
        self.assertFalse(batches[0].passed)
        self.assertEqual(batches[0].reason, "engine_failed")
        self.assertEqual(batches[0].target, CV.Identity(MODEL, NEW_CONFIG))
        self.assertEqual(entries, [])

    def test_a_work_dir_inside_the_cache_is_refused(self):
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        argv = [a for a in self.base_args() if a not in ("--work-dir", str(self.work))]
        code = CV.main(argv + ["--work-dir", str(self.cache / "work")])
        self.assertEqual(code, 2)

    def test_target_model_without_config_is_a_usage_error(self):
        self.assertEqual(CV.main(["--cache-dir", str(self.cache), "--engine", str(self.engine),
                                  "--target-model", "1"]), 2)

    def test_a_config_json_supplies_the_args_and_never_prints_them(self):
        config = self.root / "cfg.json"
        config.write_text(json.dumps({"args": ["--pack", "/p", "--api-key", "SECRET"], "cwd": str(self.root)}))
        (self.cache / "strata-conv-a.sess").write_bytes(b"a" * 50)
        (self.cache / "strata-conv-a.meta").write_bytes(sidecar_bytes(checkpoints=()))
        seen = {}
        def generate(engine, config_args, extra_args, ids, ids_path, out_path, cwd, timeout):
            seen["args"] = config_args
            return CV.FreshRun(CV.Identity(MODEL, NEW_CONFIG), 3)
        code, _, text = self.run_main(self.base_args() + ["--engine-config", str(config)], generate,
                                      lambda *a: digest())
        self.assertEqual(code, 0)
        self.assertEqual(seen["args"], ["--pack", "/p", "--api-key", "SECRET"])
        self.assertNotIn("SECRET", text)


if __name__ == "__main__":
    unittest.main()
