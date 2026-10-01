"""#214: where setup.py gets what it installs.  The ready-made engine comes from the release this checkout is (the
newest release only when that one has none), every Hugging Face download names a commit, the Python packages are
exact versions from requirements.txt, and mtp_fetch reads its inventory again when that came from another commit.
No network: the HEAD requests and downloads are stubbed.

    python -m unittest tools.test_setup_sources
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import urllib.error
import zipfile
from email.message import Message
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import mtp_fetch  # noqa: E402
import setup  # noqa: E402

VERSION = ".".join(map(str, setup.MIN_ENGINE))
TAGGED = f"{setup.RELEASES}download/v{VERSION}/"
LATEST = f"{setup.RELEASES}latest/download/"


class Engine(unittest.TestCase):
    """get_prebuilt against a stubbed GitHub: URLs under `missing` answer 404, under `broken` 503, the rest 200."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.missing, self.broken, self.heads, self.downloads = set(), set(), [], []
        self.version = VERSION                         # the version in the downloaded archive's BUILD.json
        for p in (mock.patch.object(setup, "ROOT", self.root), mock.patch.object(setup, "source_version", lambda: VERSION),
                  mock.patch.object(setup, "download", self.download), mock.patch("urllib.request.urlopen", self.head),
                  mock.patch("sys.stdout", io.StringIO())):
            p.start()
            self.addCleanup(p.stop)

    def head(self, req, timeout=None):
        self.heads.append(req.full_url)
        for prefixes, code in ((self.missing, 404), (self.broken, 503)):
            if any(req.full_url.startswith(p) for p in prefixes):
                raise urllib.error.HTTPError(req.full_url, code, "stub", Message(), None)
        return io.BytesIO()

    def download(self, url, dst, what=None):
        self.downloads.append(url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dst, "w") as z:
            z.writestr("BUILD.json", json.dumps({"version": self.version, "archs": [120]}))
            z.writestr(setup.EXE, b"")
        setup.mark(dst)

    def get(self, url_base=None):
        return setup.get_prebuilt(url_base, {"arch": "120"}, "none")

    def test_the_checkouts_release(self):
        self.assertEqual(self.get(), self.root / "engine")
        self.assertEqual(self.downloads, [TAGGED + setup.PREBUILT_ASSET])

    def test_the_newest_release_when_the_checkouts_has_none(self):
        self.missing.add(TAGGED)
        self.assertEqual(self.get(), self.root / "engine")
        self.assertEqual(self.downloads, [LATEST + setup.PREBUILT_ASSET])

    def test_a_failing_server_is_not_a_missing_release(self):
        """A 503 (or no internet) is not a reason to install another release's engine: compile instead."""
        self.broken.add(TAGGED)
        self.assertIsNone(self.get())
        self.assertEqual((self.heads, self.downloads), ([TAGGED + setup.PREBUILT_ASSET], []))

    def test_prebuilt_is_taken_as_given(self):
        self.missing.add("https://mirror.example/strata/")
        self.assertIsNone(self.get("https://mirror.example/strata"))
        self.assertIsNone(self.get(""))                # --prebuilt "": no ready-made engine, nothing asked
        self.assertEqual((self.heads, self.downloads), (["https://mirror.example/strata/" + setup.PREBUILT_ASSET], []))

    def test_a_refused_archive_is_not_reused(self):
        """An engine older than MIN_ENGINE (the newest release while the checkout's is still uploading) leaves no
        archive behind, so the next run downloads the release it picks instead of reusing this file."""
        self.missing.add(TAGGED)
        self.version = "0.1.0"
        self.assertIsNone(self.get())
        self.missing.clear()
        self.version = VERSION
        self.assertEqual(self.get(), self.root / "engine")
        self.assertEqual(self.downloads, [LATEST + setup.PREBUILT_ASSET, TAGGED + setup.PREBUILT_ASSET])


class Requirements(unittest.TestCase):
    def test_every_package_is_an_exact_version(self):
        lines = [x for x in setup.REQUIREMENTS.read_text().splitlines() if x.strip() and not x.startswith("#")]
        self.assertTrue(lines)
        for line in lines:
            self.assertRegex(line, r'^[A-Za-z0-9_.-]+==[0-9][0-9.]*(; [a-z_]+ [<>=!]+ "[^"]+")?$')

    def test_a_changed_pin_installs_again(self):
        with tempfile.TemporaryDirectory() as d:
            req, calls = Path(d, "requirements.txt"), []
            with mock.patch("sys.prefix", d), mock.patch.object(setup, "run", lambda cmd: calls.append(cmd[-2:])), \
                    mock.patch("sys.stdout", io.StringIO()):
                req.write_text("# pins\nnumpy==2.5.3\n")
                setup.pip_install(req, "numpy")
                setup.pip_install(req, "numpy")        # the same file: skipped
                req.write_text("# pins\nnumpy==2.5.4\n")
                setup.pip_install(req, "numpy")
        self.assertEqual(calls, [["-r", req], ["-r", req]])


class HuggingFace(unittest.TestCase):
    def test_every_url_names_a_commit(self):
        urls = [f[k] for f in setup.FAMILIES.values() for k in ("hf", "mmproj_hf")] + [mtp_fetch.REPO]
        for url in urls:
            self.assertRegex(url, r"^https://huggingface\.co/[^/]+/[^/]+/resolve/[0-9a-f]{40}/")

    def fetch(self, inventory_repo):
        """mtp_fetch.fetch over an inventory read from inventory_repo and a tensor fetched with it: how often it read
        the inventory again, and whether that tensor is still there."""
        with tempfile.TemporaryDirectory() as d:
            Path(d, "mtp-inventory.json").write_text(json.dumps({"repo": inventory_repo, "tensors": []}))
            tensor = Path(d, "tensors", "mtp.fc.weight.bin")
            tensor.parent.mkdir()
            tensor.write_bytes(b"\0")
            with mock.patch.object(mtp_fetch, "inventory", return_value=[]) as inventory, \
                    mock.patch.object(mtp_fetch, "get", side_effect=AssertionError("no download")):
                mtp_fetch.fetch(d, None)
            return inventory.call_count, tensor.exists()

    def test_mtp_inventory_from_the_pinned_commit_is_reused(self):
        self.assertEqual(self.fetch(mtp_fetch.REPO), (0, True))

    def test_mtp_inventory_from_another_commit_is_read_again(self):
        """And its tensors fetched again: resuming one by its size could join bytes from two commits."""
        self.assertEqual(self.fetch("https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"), (1, False))


if __name__ == "__main__":
    unittest.main()
