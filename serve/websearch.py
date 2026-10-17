"""serve/websearch.py - web search for the web app (POST /search).

The web app's Web search section picks the engine; every request carries its own config and nothing
is stored on the server (API keys stay in the browser that set them, like its API key field):

    cfg = {"engine": "searxng" | "duckduckgo" | "brave" | "tavily" | "serper",
           "url": SearXNG base, default http://127.0.0.1:8888,
           "api_key": Brave / Tavily / Serper,
           "max_results": 1..20, default 6,
           "read_pages": true/false, "read_max": 1..5  (fetch the top pages in full),
           "page_limit": true/false + "page_chars": per-page cap (default 5000),
           "reader": "off" | "fallback" | "always"  (Jina Reader r.jina.ai for JS-heavy pages),
           "n_ctx": the model context - the reading budget is a slice of it (10%, or 25% when page_limit is off)}

All providers go through this server (a browser may not call them directly: CORS), stdlib only.
search(query, cfg) -> [{"title", "url", "snippet"}]; SearchError carries a readable message.
"""
from __future__ import annotations

import html as _html
import os
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

DEFAULT_SEARCH_URL = "http://127.0.0.1:8888"
TIMEOUT_S = 15
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


class SearchError(Exception):
    pass


def _get(url, headers=None, timeout=TIMEOUT_S):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _post_json(url, payload, headers=None, timeout=TIMEOUT_S):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"User-Agent": UA, "Content-Type": "application/json",
                                          "Accept": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _clean(text):
    return _html.unescape(re.sub(r"<[^>]+>", " ", text or "")).strip()


def _searxng(query, cfg, n):
    base = (cfg.get("url") or DEFAULT_SEARCH_URL).rstrip("/")
    q = urllib.parse.urlencode({"q": query, "format": "json"})
    try:
        data = json.loads(_get(f"{base}/search?{q}"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        raise SearchError(f"SearXNG at {base} is not reachable ({e}); start it or pick another engine") from e
    out = []
    for row in data.get("results") or []:
        url = (row.get("url") or "").strip()
        if url:
            out.append({"title": (row.get("title") or "").strip() or url, "url": url,
                        "snippet": (row.get("content") or "").strip()})
        if len(out) >= n:
            break
    return out


def _duckduckgo(query, cfg, n):
    # DDG lite serves a bot-check ("anomaly") page to browser-impersonating UAs but answers an
    # honest custom UA - so this engine deliberately does not pretend to be Chrome.
    req = urllib.request.Request("https://lite.duckduckgo.com/lite/",
                                 data=urllib.parse.urlencode({"q": query}).encode(), method="POST",
                                 headers={"User-Agent": "StrataWebSearch/1.0 (personal local AI chat)",
                                          "Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            page = r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as e:
        raise SearchError(f"DuckDuckGo is not reachable ({e})") from e
    if "/anomaly.js" in page:   # the bot-check page's own marker; a query about "anomaly" must not trip this
        raise SearchError("DuckDuckGo's bot check blocked this PC. Wait a minute and retry, "
                          "or switch to SearXNG")
    attrs_title = re.findall(r"<a\b([^>]*)>(.*?)</a>", page, re.S)   # DDG lite: single quotes, href before class
    snips = re.findall(r"class=[\"']result-snippet[\"'][^>]*>(.*?)</td>", page, re.S)
    out = []
    for k, (attrs, title) in enumerate(a for a in attrs_title if "result-link" in a[0]):
        m = re.search(r"href=[\"']([^\"']+)[\"']", attrs)
        if not m:
            continue
        url = _html.unescape(m.group(1))
        if url.startswith("//duckduckgo.com/lite/"):      # DDG's own redirect wrapper
            url = urllib.parse.parse_qs(urllib.parse.urlparse("https:" + url).query).get("uddg", [url])[0]
        out.append({"title": _clean(title) or url, "url": url,
                    "snippet": _clean(snips[k]) if k < len(snips) else ""})
        if len(out) >= n:
            break
    return out


def _brave(query, cfg, n):
    key = (cfg.get("api_key") or "").strip()
    if not key:
        raise SearchError("Brave Search needs its API key (Chat Settings > Web search)")
    q = urllib.parse.urlencode({"q": query, "count": n})
    try:
        data = json.loads(_get(f"https://api.search.brave.com/res/v1/web/search?{q}",
                               {"X-Subscription-Token": key, "Accept": "application/json"}))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 422):   # Brave answers 422 for a bad key
            raise SearchError("Brave Search rejected the API key (HTTP " + str(e.code) + ")") from e
        raise SearchError(f"Brave Search is not reachable (HTTP {e.code})") from e
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        raise SearchError(f"Brave Search is not reachable ({e})") from e
    return [{"title": r.get("title") or r.get("url", ""), "url": r.get("url", ""),
             "snippet": r.get("description") or ""}
            for r in (data.get("web") or {}).get("results", [])[:n] if r.get("url")]


def _tavily(query, cfg, n):
    key = (cfg.get("api_key") or "").strip()
    if not key:
        raise SearchError("Tavily needs its API key (Chat Settings > Web search)")
    try:
        data = _post_json("https://api.tavily.com/search",
                          {"query": query, "max_results": n, "include_answer": False},
                          {"Authorization": f"Bearer {key}"})
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise SearchError("Tavily rejected the API key (HTTP " + str(e.code) + ")") from e
        raise SearchError(f"Tavily is not reachable (HTTP {e.code})") from e
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        raise SearchError(f"Tavily is not reachable ({e})") from e
    return [{"title": r.get("title") or r.get("url", ""), "url": r.get("url", ""),
             "snippet": r.get("content") or ""} for r in data.get("results", [])[:n] if r.get("url")]


def _serper(query, cfg, n):
    key = (cfg.get("api_key") or "").strip()
    if not key:
        raise SearchError("Serper needs its API key (Chat Settings > Web search)")
    try:
        data = _post_json("https://google.serper.dev/search", {"query": query, "numResults": n},
                          {"X-API-KEY": key})
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise SearchError("Serper rejected the API key (HTTP " + str(e.code) + ")") from e
        raise SearchError(f"Serper is not reachable (HTTP {e.code})") from e
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        raise SearchError(f"Serper is not reachable ({e})") from e
    return [{"title": r.get("title") or r.get("link", ""), "url": r.get("link", ""),
             "snippet": r.get("snippet") or ""} for r in data.get("organic", [])[:n] if r.get("link")]


PAGE_TIMEOUT_S = 10
PAGE_MAX_BYTES = 2_000_000
PAGE_CHARS = 5000
# Jina Reader (https://r.jina.ai/<url>): a hosted headless browser renders the page and returns
# clean text. Anonymous use is free but rate-limited (~20/min). STRATA_JINA_URL overrides the base
# (self-hosted instance, or tests). Privacy: with fallback/always the page URLs pass through it.
JINA_BASE = os.environ.get("STRATA_JINA_URL", "https://r.jina.ai")
JINA_TIMEOUT_S = 25


class _TextExtractor(HTMLParser):
    """Readable text from HTML: skips chrome (script/nav/footer/...), prefers main/article."""
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "button",
            "select", "option", "iframe", "template"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.all_parts = []
        self.main_parts = []
        self._skip = 0
        self._main = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in ("main", "article"):
            self._main += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in ("main", "article") and self._main:
            self._main -= 1

    def handle_data(self, data):
        if self._skip:
            return
        data = data.strip()
        if data:
            (self.main_parts if self._main else self.all_parts).append(data)


def _page_text(page: str) -> str:
    ex = _TextExtractor()
    ex.feed(page)
    main = " ".join(ex.main_parts)
    text = main if len(main) >= 800 else " ".join(ex.all_parts)
    return re.sub(r"\s+", " ", text).strip()


def _read_page(result: dict, budget: int) -> tuple[str, bool]:
    """Fetch one result page -> (readable text, ok). On failure the text is a short readable reason."""
    try:
        req = urllib.request.Request(result["url"], headers={"User-Agent": UA, "Accept": "text/html"})
        with urllib.request.urlopen(req, timeout=PAGE_TIMEOUT_S) as r:
            ctype = (r.headers.get("Content-Type") or "").lower()
            if "html" not in ctype:
                return "(not an HTML page: " + (ctype.split(";")[0] or "unknown type") + ")", False
            page = r.read(PAGE_MAX_BYTES).decode("utf-8", "replace")
    except Exception as e:  # one unreadable page must not break the search
        return "(page could not be read: " + e.__class__.__name__ + ")", False
    text = _page_text(page)
    if len(text) < 300:
        return "(page has almost no readable text - it likely needs JavaScript)", False
    return text[:budget], True


def _read_page_jina(result: dict, budget: int) -> tuple[str, bool]:
    """One page through Jina Reader (renders JavaScript first) -> (text, ok); reason on failure."""
    url = JINA_BASE.rstrip("/") + "/" + result["url"]
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/plain",
                                                   "X-Return-Format": "text"})
        with urllib.request.urlopen(req, timeout=JINA_TIMEOUT_S) as r:
            text = r.read(PAGE_MAX_BYTES).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        why = {402: "asked for an API key", 403: "blocked the request", 404: "found no page there",
               422: "rejected the URL", 429: "is rate-limited (wait a minute)",
               500: "had an internal error", 502: "could not reach the page",
               503: "is busy", 504: "timed out fetching the page"}.get(e.code, f"returned HTTP {e.code}")
        return "(Jina Reader " + why + ")", False
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return "(Jina Reader at " + JINA_BASE + " is not reachable: " + str(e) + ")", False
    except Exception as e:  # one unreadable page must not break the search
        return "(Jina Reader failed: " + e.__class__.__name__ + ")", False
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 300:
        return "(Jina Reader returned almost no text for this page)", False
    return text[:budget], True


def _read_with_reader(result: dict, budget: int, reader: str) -> tuple[str, str | None, bool]:
    """(text-or-reason, read_via, ok). reader: off = direct only; fallback = Jina when direct fails;
    always = Jina first, direct as its fallback. Both failing gives both reasons."""
    if reader == "always":
        jina, jok = _read_page_jina(result, budget)
        if jok:
            return jina, "jina", True
        direct, dok = _read_page(result, budget)
        if dok:
            return direct, None, True
        return direct + " " + jina, None, False
    direct, dok = _read_page(result, budget)
    if reader == "fallback" and not dok:
        jina, jok = _read_page_jina(result, budget)
        if jok:
            return jina, "jina", True
        return direct + " " + jina, None, False
    return direct, None, dok


ENGINES = {"searxng": _searxng, "duckduckgo": _duckduckgo, "brave": _brave, "tavily": _tavily,
           "serper": _serper}


def search(query: str, cfg: dict | None = None) -> list[dict]:
    cfg = cfg or {}
    engine = str(cfg.get("engine") or "duckduckgo").strip().lower()
    fn = ENGINES.get(engine)
    if fn is None:
        raise SearchError(f"unknown search engine {engine!r} (choose one of: {', '.join(ENGINES)})")
    try:
        n = int(cfg.get("max_results") or 6)
    except (TypeError, ValueError):
        n = 6
    results = fn(query, cfg, max(1, min(20, n)))
    if cfg.get("read_pages"):
        try:
            k = max(1, min(5, int(cfg.get("read_max") or 2)))
        except (TypeError, ValueError):
            k = 2
        # the reading budget is a slice of the model's window (~3 chars/token): 10% normally,
        # 25% when the user explicitly asked for whole pages; unknown context -> the plain default
        try:
            n_ctx = int(cfg.get("n_ctx") or 0)
        except (TypeError, ValueError):
            n_ctx = 0
        limit = cfg.get("page_limit", True)
        if n_ctx > 0:
            budget_total = int(n_ctx * (0.10 if limit else 0.25) * 3)
        else:
            budget_total = PAGE_CHARS * k
        if limit:
            try:
                per_page = max(500, min(100_000, int(cfg.get("page_chars") or PAGE_CHARS)))
            except (TypeError, ValueError):
                per_page = PAGE_CHARS
            per_page = min(per_page, max(1500, budget_total // k))
        else:
            per_page = max(1500, budget_total // k)   # "unlimited": the context slice is the only cap
        reader = str(cfg.get("reader") or "off").strip().lower()
        if reader not in ("off", "fallback", "always"):
            reader = "off"
        for r in results[:k]:
            text, via, ok = _read_with_reader(r, per_page, reader)
            if ok:
                r["content"] = text
                if via:
                    r["read_via"] = via
            else:
                r["read_error"] = text   # explicit: never confused with real page text
    return results
