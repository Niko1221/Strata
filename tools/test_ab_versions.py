"""Tests for ab_versions: release pairs and version specs.  No network, no GPU."""
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_versions import list_releases, pairs_from_releases, resolve_versions  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
