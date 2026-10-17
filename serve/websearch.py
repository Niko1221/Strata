"""serve/websearch.py - web search for the web app (POST /search).

The web app's Web search section picks the engine; every request carries its own config and nothing
is stored on the server (API keys stay in the browser that set them, like its API key field):

    cfg = {"engine": "searxng" | "duckduckgo" | "brave" | "tavily" | "serper",
           "url": SearXNG base, default http://127.0.0.1:8888,
           "api_key": Brave / Tavily / Serper,
           "max_results": 1..20, default 6}

All providers go through this server (a browser may not call them directly: CORS), stdlib only.
search(query, cfg) -> [{"title", "url", "snippet"}]; SearchError carries a readable message.
"""
from __future__ import annotations

import html as _html
import json
import re
import urllib.error
import urllib.parse
import urllib.request

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
    req = urllib.request.Request("https://lite.duckduckgo.com/lite/",
                                 data=urllib.parse.urlencode({"q": query}).encode(), method="POST",
                                 headers={"User-Agent": UA, "Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            page = r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as e:
        raise SearchError(f"DuckDuckGo is not reachable ({e})") from e
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
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        raise SearchError(f"Serper is not reachable ({e})") from e
    return [{"title": r.get("title") or r.get("link", ""), "url": r.get("link", ""),
             "snippet": r.get("snippet") or ""} for r in data.get("organic", [])[:n] if r.get("link")]


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
    return fn(query, cfg, max(1, min(20, n)))
