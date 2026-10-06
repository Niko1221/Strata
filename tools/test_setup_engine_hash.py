"""Tests for setup.py's verification of the ready-made engine archive against GitHub's SHA-256.

`get_prebuilt()` used to install the engine on nothing more than the byte count matching the server's
Content-Length, so a substituted file of the same length passed. These cover the check that closes that,
the refusal when GitHub will not give a hash, and the order it happens in - it must run before the archive
is opened, or a wrong engine still reaches the disk. Mocked network - nothing is downloaded.

    python -m unittest tools.test_setup_engine_hash
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import setup  # noqa: E402

ASSET = "strata-windows-x64.zip"
TAG_BASE = "https://github.com/Niko1221/Strata/releases/download/v0.1.40/"
LATEST_BASE = "https://github.com/Niko1221/Strata/releases/latest/download/"


def zip_bytes() -> bytes:
    """A stand-in engine archive. Small; its hash is what matters, not its size."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("strata.exe", b"not a real engine")
        zf.writestr("BUILD.json", b'{"version": "0.1.40"}')
    return buf.getvalue()


def sha_of(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class DigestLookup(unittest.TestCase):
    """`engine_digest`: the size and hash for one asset, from the releases API."""

    def release_json(self, tag, assets):
        import json
        return json.dumps({"tag_name": tag, "assets": assets}).encode("utf-8")

    def fake_open(self, payload, seen=None):
        """A urlopen that serves `payload` and records the URL asked for.

        `seen` is filled in place, because patching urlopen with a plain function gives back the function
        rather than a mock, and the URL is the thing under test here.  It defaults to a throwaway list for
        the tests that only care about the answer.
        """
        seen = [] if seen is None else seen
        class R:
            def __init__(self, data):
                self._b = io.BytesIO(data)

            def read(self, *a):
                return self._b.read(*a)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def open_it(req, timeout=None):
            seen.append(req.full_url)
            return R(payload)

        return open_it

    def asked(self, payload):
        """Patch urlopen with the fake; returns the list the URL is appended to."""
        seen = []
        patcher = mock.patch.object(setup.urllib.request, "urlopen", self.fake_open(payload, seen))
        patcher.start()
        self.addCleanup(patcher.stop)
        return seen

    def test_reads_the_tag_out_of_the_download_url(self):
        # The URL already names the release, so the API call must ask for THAT release and not "latest":
        # the tag URL is tried first (#214) so a checkout normally gets its own version's engine.
        seen = self.asked(self.release_json(
            "v0.1.40", [{"name": ASSET, "size": 42, "digest": "sha256:" + "a" * 64}]))
        got = setup.engine_digest(ASSET, TAG_BASE)
        self.assertEqual(got, (42, "a" * 64))
        self.assertEqual(seen, ["https://api.github.com/repos/Niko1221/Strata/releases/tags/v0.1.40"])

    def test_latest_when_the_url_says_latest(self):
        seen = self.asked(self.release_json(
            "v0.1.40.1", [{"name": ASSET, "size": 7, "digest": "sha256:" + "b" * 64}]))
        setup.engine_digest(ASSET, LATEST_BASE)
        self.assertEqual(seen, ["https://api.github.com/repos/Niko1221/Strata/releases/latest"])

    def test_answers_for_a_mirror_too_because_the_hash_is_from_elsewhere(self):
        # --prebuilt can point the download at a mirror. The hash still comes from api.github.com, which
        # is the whole value: a compromised mirror cannot supply bytes with a matching digest.
        seen = self.asked(self.release_json(
            "v0.1.40", [{"name": ASSET, "size": 7, "digest": "sha256:" + "c" * 64}]))
        got = setup.engine_digest(ASSET, "https://mirror.invalid/dl/")
        self.assertEqual(got, (7, "c" * 64))
        self.assertIn("api.github.com/repos/Niko1221/Strata", seen[0])

    def test_none_when_no_digest_is_published(self):
        # No digest is not a pass. The caller decides, and it refuses.
        with mock.patch.object(setup.urllib.request, "urlopen",
                               self.fake_open(self.release_json(
                                   "v0.1.40", [{"name": ASSET, "size": 42}]))):
            self.assertIsNone(setup.engine_digest(ASSET, TAG_BASE))

    def test_none_when_the_digest_is_not_a_sha256(self):
        for other in ("md5:" + "d" * 32, "sha512:" + "e" * 128, "", None):
            with mock.patch.object(setup.urllib.request, "urlopen",
                                   self.fake_open(self.release_json(
                                       "v0.1.40",
                                       [{"name": ASSET, "size": 42, "digest": other}]))):
                self.assertIsNone(setup.engine_digest(ASSET, TAG_BASE), other)

    def test_none_when_the_asset_is_not_in_that_release(self):
        # A CUDA 12 asset asked for on a release that has none, or a renamed asset. Refuse, do not
        # verify against some other file's hash.
        with mock.patch.object(setup.urllib.request, "urlopen",
                               self.fake_open(self.release_json(
                                   "v0.1.40",
                                   [{"name": "other.zip", "size": 1, "digest": "sha256:" + "f" * 64}]))):
            self.assertIsNone(setup.engine_digest(ASSET, TAG_BASE))

    def test_none_when_github_cannot_be_reached(self):
        # Offline, rate-limited, DNS gone: an OSError, not a crash, and not a pass.
        for exc in (OSError("no network"), TimeoutError("slow")):
            with mock.patch.object(setup.urllib.request, "urlopen", side_effect=exc):
                self.assertIsNone(setup.engine_digest(ASSET, TAG_BASE), exc)

    def test_picks_the_asset_the_url_actually_wants(self):
        # v0.1.40.1 ships two engine builds with different hashes; the one asked for must be the one used.
        payload = self.release_json("v0.1.40.1", [
            {"name": "strata-windows-x64.zip", "size": 1, "digest": "sha256:" + "1" * 64},
            {"name": "strata-windows-x64-cuda12.zip", "size": 2, "digest": "sha256:" + "2" * 64},
        ])
        with mock.patch.object(setup.urllib.request, "urlopen", self.fake_open(payload)):
            self.assertEqual(setup.engine_digest("strata-windows-x64-cuda12.zip", LATEST_BASE),
                             (2, "2" * 64))


class ArchiveVerification(unittest.TestCase):
    """`verify_engine_archive`: the downloaded file against that size and hash, before it is unpacked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.z = Path(self.tmp.name) / ASSET
        self.data = zip_bytes()
        self.z.write_bytes(self.data)

    def env(self, **kw):
        """No STRATA_ALLOW_UNVERIFIED_ENGINE unless a test wants it."""
        patch = mock.patch.dict(os.environ, {}, clear=False)
        env = patch.start()
        self.addCleanup(patch.stop)
        env.pop("STRATA_ALLOW_UNVERIFIED_ENGINE", None)
        env.update(kw)
        return env

    def test_passes_when_the_hash_matches(self):
        self.env()
        with mock.patch.object(setup, "engine_digest",
                               return_value=(len(self.data), sha_of(self.data))):
            setup.verify_engine_archive(self.z, ASSET, TAG_BASE)      # must not exit
        self.assertTrue(self.z.exists())

    def test_refuses_a_wrong_hash_and_deletes_the_file(self):
        # A wrong engine left on disk is the thing this exists to prevent, so it goes.
        self.env()
        with mock.patch.object(setup, "engine_digest",
                               return_value=(len(self.data), "0" * 64)):
            with self.assertRaises(SystemExit):
                setup.verify_engine_archive(self.z, ASSET, TAG_BASE)
        self.assertFalse(self.z.exists())
        self.assertFalse(self.z.with_name(self.z.name + ".done").exists())

    def test_refuses_a_wrong_size_before_hashing(self):
        self.env()
        with mock.patch.object(setup, "engine_digest",
                               return_value=(len(self.data) + 1, sha_of(self.data))):
            with self.assertRaises(SystemExit):
                setup.verify_engine_archive(self.z, ASSET, TAG_BASE)

    def test_the_verified_hash_is_kept_in_the_download_mark(self):
        # So the ~190 MB is not hashed again on the next run, the same idiom as verify_sha256 uses for
        # the Unsloth shards - a re-run must skip straight past it.
        self.env()
        sha = sha_of(self.data)
        with mock.patch.object(setup, "engine_digest", return_value=(len(self.data), sha)):
            setup.verify_engine_archive(self.z, ASSET, TAG_BASE)
            calls = []
            real = hashlib.sha256

            def spy(data=b""):
                calls.append(len(data))
                return real(data)

            with mock.patch.object(hashlib, "sha256", spy):
                setup.verify_engine_archive(self.z, ASSET, TAG_BASE)
        self.assertEqual(calls, [], "the second call hashed nothing")

    def test_refuses_when_github_gives_no_hash(self):
        # No digest means no verification is possible. Installing unchecked is the decision the person
        # running setup has to make explicitly, not something that happens by default.
        self.env()
        with mock.patch.object(setup, "engine_digest", return_value=None):
            with self.assertRaises(SystemExit):
                setup.verify_engine_archive(self.z, ASSET, TAG_BASE)
        self.assertTrue(self.z.exists(), "the file is left alone; only the install is refused")

    def test_the_escape_hatch_is_opt_in_and_says_so(self):
        self.env(STRATA_ALLOW_UNVERIFIED_ENGINE="1")
        said = io.StringIO()
        with mock.patch.object(setup, "say", said.write), \
                mock.patch.object(setup, "warn", said.write), \
                mock.patch.object(setup, "engine_digest", return_value=None):
            setup.verify_engine_archive(self.z, ASSET, TAG_BASE)      # must not exit
        out = said.getvalue()
        self.assertIn("UNVERIFIED", out)
        self.assertIn(ASSET, out)


class Wiring(unittest.TestCase):
    """The check is reached, and reached before the archive is opened."""

    def test_get_prebuilt_verifies_before_it_unpacks(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        body = src[src.index("def get_prebuilt("):]
        body = body[:body.index("\ndef ")]
        i_verify = body.index("verify_engine_archive(")
        # Everything that reads the archive's bytes has to come after it.
        for opener in ("ZipFile(", "_unpack", "install_unpacked"):
            self.assertGreater(body.index(opener), i_verify,
                               f"{opener} happens before the SHA-256 is checked")

    def test_the_digest_is_read_from_the_api_not_the_download_host(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        body = src[src.index("def engine_digest("):]
        body = body[:body.index("\ndef ")]
        self.assertIn("api.github.com/repos/", body)


if __name__ == "__main__":
    unittest.main()
