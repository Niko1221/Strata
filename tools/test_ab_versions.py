"""Tests for ab_versions: release pairs and version specs.  No network, no GPU."""
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_versions import (Packages, ReleasePair, ensure_packages, isolated_env, list_releases, pairs_from_releases,  # noqa: E402
                         parse_sums, provision_home, resolve_versions)


def a(name):
    return {"name": name, "browser_download_url": f"https://example.invalid/dl/{name}"}


FIXTURE = [  # newest first, as the API returns them
    {"tag_name": "v0.2.10", "assets": [a("strata-engine-0.1.56-windows-x86_64-cuda-cuda13.tar.gz"),
                                       a("strata-control-0.1.56-windows-x86_64.zip"), a("SHA256SUMS.txt")]},
    {"tag_name": "v0.2.9", "assets": [a("strata-engine-0.1.55-windows-x86_64-cuda-cuda13.tar.gz"),
                                      a("strata-control-0.1.55-windows-x86_64.zip"), a("SHA256SUMS.txt")]},
    {"tag_name": "v0.2.8", "assets": [a("strata-engine-0.1.54-windows-x86_64-cuda-cuda12.tar.gz"),
                                      a("strata-control-0.1.54-windows-x86_64.zip"), a("SHA256SUMS.txt")]},
]
WIN13 = ("windows-x86_64", "cuda-cuda13")


class PairsTest(unittest.TestCase):
    def test_pairs_need_engine_and_control_for_backend(self):
        p = pairs_from_releases(FIXTURE, *WIN13)
        self.assertEqual(sorted(p), ["0.1.55", "0.1.56"])
        self.assertEqual(p["0.1.55"].tag, "v0.2.9")

    def test_pair_fields(self):
        p = pairs_from_releases(FIXTURE, *WIN13)["0.1.56"]
        self.assertEqual(p.engine_name, "strata-engine-0.1.56-windows-x86_64-cuda-cuda13.tar.gz")
        self.assertTrue(p.engine_url.endswith(p.engine_name))
        self.assertEqual(p.control_name, "strata-control-0.1.56-windows-x86_64.zip")
        self.assertTrue(p.control_url.endswith(p.control_name))
        self.assertTrue(p.sums_url.endswith("SHA256SUMS.txt"))

    def test_control_without_engine_is_dropped(self):
        rel = [{"tag_name": "v1", "assets": [a("strata-control-0.1.60-windows-x86_64.zip"), a("SHA256SUMS.txt")]}]
        self.assertEqual(pairs_from_releases(rel, *WIN13), {})

    def test_other_target_is_dropped(self):
        rel = [{"tag_name": "v1", "assets": [a("strata-engine-0.1.60-linux-x86_64-cuda-cuda13.tar.gz"),
                                             a("strata-control-0.1.60-linux-x86_64.tar.gz"), a("SHA256SUMS.txt")]}]
        self.assertEqual(pairs_from_releases(rel, *WIN13), {})
        self.assertEqual(sorted(pairs_from_releases(rel, "linux-x86_64", "cuda-cuda13")), ["0.1.60"])

    def test_first_occurrence_wins(self):
        rel = [{"tag_name": "v0.2.11", "assets": FIXTURE[0]["assets"]}, FIXTURE[0]]
        self.assertEqual(pairs_from_releases(rel, *WIN13)["0.1.56"].tag, "v0.2.11")


class ResolveTest(unittest.TestCase):
    def setUp(self):
        self.p = pairs_from_releases(FIXTURE, *WIN13)

    def test_range_is_inclusive_and_sorted(self):
        self.assertEqual(resolve_versions("0.1.50..0.1.56", self.p), ["0.1.55", "0.1.56"])
        self.assertEqual(resolve_versions("0.1.56,0.1.55", self.p), ["0.1.55", "0.1.56"])

    def test_numeric_not_lexical_order(self):
        rel = [{"tag_name": f"t{n}", "assets": [a(f"strata-engine-0.1.{n}-windows-x86_64-cuda-cuda13.tar.gz"),
                                                a(f"strata-control-0.1.{n}-windows-x86_64.zip"), a("SHA256SUMS.txt")]}
               for n in (9, 10)]
        self.assertEqual(resolve_versions("0.1.9..0.1.10", pairs_from_releases(rel, *WIN13)), ["0.1.9", "0.1.10"])

    def test_unknown_version_in_list_fails_early(self):
        with self.assertRaisesRegex(ValueError, "0.1.54"):
            resolve_versions("0.1.54,0.1.55", self.p)

    def test_empty_range_fails(self):
        with self.assertRaises(ValueError):
            resolve_versions("0.1.60..0.1.70", self.p)


class ListReleasesTest(unittest.TestCase):
    def test_paginates_until_empty_page_without_token(self):
        pages = {1: [{"tag_name": "v2"}], 2: [{"tag_name": "v1"}], 3: []}
        seen = []

        def opener(req, timeout=None):
            seen.append(req)
            page = int(req.full_url.rsplit("page=", 1)[1].split("&")[0])
            return io.BytesIO(json.dumps(pages[page]).encode())

        out = list_releases("o/r", opener=opener)
        self.assertEqual([r["tag_name"] for r in out], ["v2", "v1"])
        self.assertEqual(len(seen), 3)
        self.assertTrue(all("/repos/o/r/releases" in r.full_url for r in seen))
        self.assertTrue(all("authorization" not in {k.lower() for k in r.headers} for r in seen))


def tar_bytes(name, data=b"x"):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def zip_bytes(name, data=b"x"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, data)
    return buf.getvalue()


class FakeResponse:
    """Like a urlopen response: context manager, `.read(n)`; fails mid-stream when `fail_after` bytes were read."""

    def __init__(self, data, fail_after=None):
        self.buf, self.fail_after, self.read_total = io.BytesIO(data), fail_after, 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=-1):
        if self.fail_after is not None and self.read_total >= self.fail_after:
            raise OSError("connection reset")
        chunk = self.buf.read(n if self.fail_after is None else min(n, 4))
        self.read_total += len(chunk)
        return chunk


def make_opener(files, fail_engine=False):
    def opener(url, timeout=None):
        if url not in files:
            raise OSError(f"no network for {url}")
        return FakeResponse(files[url], fail_after=4 if fail_engine and url.endswith("/e") else None)
    return opener


def no_network(url, timeout=None):
    raise AssertionError(f"unexpected download of {url}")


class PackagesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = Path(self.tmp.name) / "packages"
        self.engine, self.control = tar_bytes("strata-engine.exe", b"engine"), zip_bytes("strata.exe", b"control")
        self.pair = ReleasePair("0.1.56", "v0.2.10", "strata-engine-0.1.56-windows-x86_64-cuda-cuda13.tar.gz", "https://x.invalid/e",
                                "strata-control-0.1.56-windows-x86_64.zip", "https://x.invalid/c", "https://x.invalid/sums")
        self.sums = self.sums_for(self.engine, self.control)

    def sums_for(self, engine, control):
        h = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
        return f"{h(engine)} *{self.pair.engine_name}\n{h(control)}  {self.pair.control_name}\n".encode()

    def files(self, sums=None):
        return {"https://x.invalid/e": self.engine, "https://x.invalid/c": self.control,
                "https://x.invalid/sums": self.sums if sums is None else sums}

    def test_downloads_extracts_and_marks(self):
        pk = ensure_packages(self.pair, self.cache, make_opener(self.files()))
        self.assertEqual(pk.version, "0.1.56")
        self.assertEqual(pk.engine_dir, self.cache / "strata-engine-0.1.56-windows-x86_64-cuda-cuda13")
        self.assertEqual(pk.control_dir, self.cache / "strata-control-0.1.56-windows-x86_64")
        self.assertEqual((pk.engine_dir / "strata-engine.exe").read_bytes(), b"engine")
        self.assertEqual((pk.control_dir / "strata.exe").read_bytes(), b"control")
        self.assertTrue((pk.engine_dir / ".ok").is_file() and (pk.control_dir / ".ok").is_file())
        self.assertEqual(sorted(p.name for p in self.cache.iterdir()), sorted([pk.engine_dir.name, pk.control_dir.name]))

    def test_checksum_mismatch_raises_and_leaves_no_cache(self):
        bad = self.sums_for(b"other", self.control)
        with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
            ensure_packages(self.pair, self.cache, make_opener(self.files(bad)))
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_missing_checksum_entry_raises(self):
        with self.assertRaisesRegex(ValueError, "no sha256"):
            ensure_packages(self.pair, self.cache, make_opener(self.files(b"")))
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_cache_hit_needs_no_network(self):
        first = ensure_packages(self.pair, self.cache, make_opener(self.files()))
        self.assertEqual(ensure_packages(self.pair, self.cache, no_network), first)

    def test_partial_download_not_cached(self):
        with self.assertRaises(OSError):
            ensure_packages(self.pair, self.cache, make_opener(self.files(), fail_engine=True))
        self.assertEqual(list(self.cache.iterdir()), [])
        pk = ensure_packages(self.pair, self.cache, make_opener(self.files()))
        self.assertEqual((pk.engine_dir / "strata-engine.exe").read_bytes(), b"engine")

    def test_cached_engine_is_kept_when_control_fails(self):
        files = self.files()
        del files["https://x.invalid/c"]
        with self.assertRaises(OSError):
            ensure_packages(self.pair, self.cache, make_opener(files))
        self.assertEqual([p.name for p in self.cache.iterdir()], [self.pair.engine_name[:-len(".tar.gz")]])
        ensure_packages(self.pair, self.cache, make_opener(self.files()))

    def test_parse_sums_handles_star_and_spaces(self):
        self.assertEqual(parse_sums("abc *x.tar.gz\ndef  y.zip\n"), {"x.tar.gz": "abc", "y.zip": "def"})


def fake_packages(root: Path, version: str) -> Packages:
    """Flat engine and control package folders with the smallest manifest read_manifest accepts."""
    engine = root / f"strata-engine-{version}-windows-x86_64-cuda-cuda13"
    control = root / f"strata-control-{version}-windows-x86_64"
    engine.mkdir(parents=True)
    control.mkdir(parents=True)
    (engine / "strata-engine.exe").write_bytes(b"engine")
    (control / "strata.py").write_text("print('control')", encoding="utf-8")
    (engine / "engine-component.json").write_text(json.dumps({
        "schema_version": 1, "id": "strata-engine", "version": version, "platform": "windows-x86_64", "backend": "cuda",
        "entrypoint": "strata-engine.exe", "files": [{"path": "strata-engine.exe", "sha256": "0" * 64, "bytes": 6}]}), encoding="utf-8")
    return Packages(version, engine, control)


class HomeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home = self.root / "home"
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from strata_control.engine_component import components_dir, scan
        self.components_dir, self.scan = components_dir, scan

    def test_env_points_inside_home_and_scrubs_strata(self):
        env = isolated_env(self.home, {"LOCALAPPDATA": "/real", "STRATA_COMPONENTS_DIR": "x", "PATH": "p"})
        for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_STATE_HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME"):
            self.assertTrue(Path(env[key]).is_relative_to(self.home), key)
        self.assertNotIn("STRATA_COMPONENTS_DIR", env)
        self.assertEqual(env["STRATA_NO_ACQUIRE"], "1")
        self.assertEqual(env["PATH"], "p")

    def test_components_dir_follows_env_on_every_platform(self):
        env = isolated_env(self.home, {})
        for windows in (True, False):
            self.assertTrue(self.components_dir(env, windows=windows).is_relative_to(self.home))

    def test_provisioned_engine_is_the_only_one_scan_sees(self):
        pk = fake_packages(self.root / "pk", "0.1.55")
        control = provision_home(self.home, pk)
        components = self.components_dir(isolated_env(self.home, {}))
        found, problems = self.scan(components)
        self.assertEqual([c.version for c in found], ["0.1.55"])
        self.assertEqual(problems, [])
        self.assertEqual(control, components / "control" / "0.1.55")
        self.assertTrue((control / "strata.py").is_file())
        self.assertEqual(json.loads((components / "control" / "active.json").read_text(encoding="utf-8")),
                         {"version": "0.1.55", "folder": "0.1.55"})

    def test_provision_twice_reuses_and_deletes_nothing(self):
        pk = fake_packages(self.root / "pk", "0.1.55")
        control = provision_home(self.home, pk)
        (control / "tuning.json").write_text("{}", encoding="utf-8")        # something a run left behind
        self.assertEqual(provision_home(self.home, pk), control)
        self.assertTrue((control / "tuning.json").is_file())
        self.assertEqual(len(self.scan(self.components_dir(isolated_env(self.home, {})))[0]), 1)


if __name__ == "__main__":
    unittest.main()
