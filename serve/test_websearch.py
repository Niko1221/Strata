"""serve/test_websearch.py - the serve-layer web search stays guarded and budgeted (web-search fork).

The fork injects web results and full result-page text into the model's prompt, so three
properties must never regress: (1) only http(s) URLs are ever fetched or rendered, a
file://, javascript: or similar scheme is refused with an explicit reason before any network
call; (2) only whitelisted engines run, and max_results is clamped to a sane range; (3) the
text injected per search is a slice of the model's own context window (10 %, or 25 % when the
per-page limit is explicitly off, with a 1500-char design floor per page), so small-context
instances are never overloaded. Reader tiers (off / fallback / always), the Jina rescue path
and its combined both-paths-fail reason are pinned here too. Offline by design: urllib.request
.urlopen is faked, no real network, no model needed.

    python -m unittest serve.test_websearch -v
"""
from __future__ import annotations

import unittest
import urllib.error
import urllib.request

from serve import websearch as w

POISON = ["file:///etc/passwd", "gopher://x/y", "javascript:alert(1)", "data:text/html,x",
          "ftp://host/f", "dict://h:11211/", "ldap://h/x", "sftp://h/x", "ssh://h/x",
          "mailto:a@b.c", "telnet://h:23", "file://C:/Windows/x", "jar:http://h!/x",
          "chrome-extension://a/b", "vbscript:x"]

LONG = "page body text " * 40


class FakeResp:
    def __init__(self, body, ctype=None):
        self.body = body.encode() if isinstance(body, str) else body
        self.headers = {"Content-Type": ctype or "text/html; charset=utf-8"}

    def read(self, n=-1):
        return self.body[:n] if n and n > 0 else self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_urlopen(routes):
    """routes checked in order: (substring, body-or-Exception, ctype); returns (fu, calls)."""
    calls = []

    def fu(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        calls.append(url)
        for key, body, ctype in routes:
            if key in url:
                if isinstance(body, Exception):
                    raise body
                return FakeResp(body, ctype)
        raise urllib.error.URLError("no route for " + url)

    return fu, calls


class WebSearchBase(unittest.TestCase):
    """Restores the patched fetch path and engine table after every test."""

    def setUp(self):
        self._urlopen = w.urllib.request.urlopen
        self._engines = dict(w.ENGINES)

    def tearDown(self):
        w.urllib.request.urlopen = self._urlopen
        w.ENGINES.clear()
        w.ENGINES.update(self._engines)

    def patch_routes(self, routes):
        fu, calls = fake_urlopen(routes)
        w.urllib.request.urlopen = fu
        return calls


class SchemeGuardTests(WebSearchBase):
    def test_poison_schemes_rejected_before_any_fetch(self):
        for u in POISON:
            with self.subTest(url=u):
                with self.assertRaises(w.SearchError):
                    w._http_url(u, "url")

    def test_http_and_https_accepted(self):
        self.assertIsNotNone(w._http_url("http://a.example/x", "url"))
        self.assertIsNotNone(w._http_url("https://a.example/x", "url"))


class EngineWhitelistTests(WebSearchBase):
    def test_only_whitelisted_engines_run(self):
        for bad in ["google", "bing", "file", "DDG", "BROWSER"]:
            with self.subTest(engine=bad):
                with self.assertRaises(w.SearchError):
                    w.search("q", {"engine": bad})


class ReaderTierTests(WebSearchBase):
    def test_tier_off_uses_direct_only(self):
        self.patch_routes([("evil.example", "<html><body><p>" + LONG + "</p></body></html>", None)])
        text, via, ok = w._read_with_reader({"url": "https://evil.example/x"}, 5000, "off")
        self.assertTrue(ok)
        self.assertIsNone(via)

    def test_tier_fallback_rescues_via_jina_after_direct(self):
        calls = self.patch_routes([("r.jina.ai", "JINA BODY " + LONG, "text/plain"),
                                   ("evil.example", urllib.error.URLError("nope"), None)])
        text, via, ok = w._read_with_reader({"url": "https://evil.example/x"}, 5000, "fallback")
        self.assertTrue(ok)
        self.assertEqual(via, "jina")
        self.assertTrue(calls[0].startswith("https://evil"))

    def test_tier_always_proxies_through_jina(self):
        calls = self.patch_routes([("r.jina.ai", "JINA BODY " + LONG, "text/plain")])
        text, via, ok = w._read_with_reader({"url": "https://evil.example/x"}, 5000, "always")
        self.assertTrue(ok)
        self.assertEqual(via, "jina")
        self.assertIn(w.JINA_BASE, calls[0])

    def test_both_paths_fail_reads_as_combined_reason(self):
        self.patch_routes([("evil.example", urllib.error.URLError("nope"), None)])
        text, via, ok = w._read_with_reader({"url": "https://evil.example/x"}, 5000, "fallback")
        self.assertFalse(ok)
        self.assertIsNone(via)
        self.assertIn("Jina too", text)

    def test_reader_refuses_poison_urls_in_every_tier(self):
        for bad_url in ("file:///etc/passwd", "javascript:alert(1)", "data:text/html,x"):
            for reader in ("off", "fallback", "always"):
                with self.subTest(url=bad_url, reader=reader):
                    t2, v2, o2 = w._read_with_reader({"url": bad_url}, 5000, reader)
                    self.assertFalse(o2)
                    self.assertIsNone(v2)
                    self.assertIn("http(s) page", t2)

    def test_bad_reader_value_degrades_to_direct(self):
        self.patch_routes([("evil.example", "<html><body><p>" + LONG + "</p></body></html>", None)])
        text, via, ok = w._read_with_reader({"url": "https://evil.example/x"}, 5000, "nonsense")
        self.assertTrue(ok)

    def test_non_html_content_refused(self):
        self.patch_routes([("evil.example", b"%PDF-1.4 ...", "application/pdf")])
        text, via, ok = w._read_with_reader({"url": "https://evil.example/x"}, 5000, "off")
        self.assertFalse(ok)
        self.assertIn("not an HTML page", text)


class PageTextTests(WebSearchBase):
    def test_extraction_drops_script_and_style_keeps_text(self):
        t = w._page_text("<html><body><script>var x=1;</script><style>a{}</style>"
                         "<h1>Head</h1><p>Body&#160;text</p></body></html>")
        self.assertNotIn("var x", t)
        self.assertNotIn("a{}", t)
        self.assertIn("Head", t)
        self.assertIn("Body", t)


class BudgetTests(WebSearchBase):
    def test_reading_budget_is_a_slice_of_the_context(self):
        def fake_engine(query, cfg, k):
            return [{"url": f"https://ex.example/{i}", "title": f"t{i}", "snippet": "s"}
                    for i in range(k)]

        self.patch_routes([("ex.example",
                            "<html><body><p>" + ("content words " * 5000) + "</p></body></html>", None)])
        w.ENGINES["duckduckgo"] = fake_engine
        for n_ctx, limit in ((16384, True), (16384, False), (4096, True)):
            with self.subTest(n_ctx=n_ctx, page_limit=limit):
                out = w.search("q", {"engine": "duckduckgo", "max_results": 5, "read_pages": True,
                                     "read_max": 5, "n_ctx": n_ctx, "page_limit": limit, "reader": "off"})
                self.assertEqual(len(out), 5)
                lens = [len(r.get("content", "")) for r in out]
                budget_total = int(n_ctx * (0.10 if limit else 0.25) * 3)
                per_page = max(1500, budget_total // 5)
                self.assertTrue(all(l <= per_page for l in lens))
                self.assertLessEqual(sum(lens), 5 * max(1500, budget_total // 5))

    def test_unknown_context_falls_back_to_page_cap(self):
        def fake_engine(query, cfg, k):
            return [{"url": f"https://ex.example/{i}", "title": f"t{i}", "snippet": "s"}
                    for i in range(k)]

        self.patch_routes([("ex.example",
                            "<html><body><p>" + ("content words " * 5000) + "</p></body></html>", None)])
        w.ENGINES["duckduckgo"] = fake_engine
        out = w.search("q", {"engine": "duckduckgo", "max_results": 3, "read_pages": True,
                             "read_max": 3, "reader": "off"})
        self.assertTrue(all(len(r.get("content", "")) <= w.PAGE_CHARS for r in out))


class MaxResultsClampTests(WebSearchBase):
    def test_max_results_clamped_to_sane_range(self):
        seen = []

        def spy(query, cfg, k):
            seen.append(k)
            return []

        w.ENGINES["duckduckgo"] = spy
        w.search("q", {"engine": "duckduckgo", "max_results": 0})
        w.search("q", {"engine": "duckduckgo", "max_results": 999})
        w.search("q", {"engine": "duckduckgo", "max_results": "junk"})
        w.search("q", {"engine": "duckduckgo", "max_results": -5})
        self.assertEqual(seen, [6, 20, 6, 1])


if __name__ == "__main__":
    unittest.main()
