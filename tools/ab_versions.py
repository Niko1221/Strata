"""Strata release helpers for ab_engine's versions mode: find engine/control pairs and resolve version specs."""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass

VER = r"(\d+(?:\.\d+)*)"


@dataclass(frozen=True)
class ReleasePair:
    version: str
    tag: str
    engine_name: str
    engine_url: str
    control_name: str
    control_url: str
    sums_url: str


def _key(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in version.split("."))


def pairs_from_releases(releases: list[dict], target: str, backend: str) -> dict[str, ReleasePair]:
    """Map version -> ReleasePair for `target`/`backend`; newest release first wins; engine and control both needed."""
    engine_re = re.compile(rf"strata-engine-{VER}-{re.escape(target)}-{re.escape(backend)}\.tar\.gz")
    control_re = re.compile(rf"strata-control-{VER}-{re.escape(target)}\.(?:zip|tar\.gz)")
    pairs: dict[str, ReleasePair] = {}
    for rel in releases:
        assets = rel.get("assets", [])
        sums = next((x for x in assets if x["name"] == "SHA256SUMS.txt"), None)
        engines = {m[1]: x for x in assets if (m := engine_re.fullmatch(x["name"]))}
        controls = {m[1]: x for x in assets if (m := control_re.fullmatch(x["name"]))}
        for ver in engines.keys() & controls.keys():
            if ver in pairs or sums is None:
                continue
            e, c = engines[ver], controls[ver]
            pairs[ver] = ReleasePair(ver, rel["tag_name"], e["name"], e["browser_download_url"],
                                     c["name"], c["browser_download_url"], sums["browser_download_url"])
    return pairs


def resolve_versions(spec: str, available: dict[str, ReleasePair]) -> list[str]:
    """`a,b` or `a..b` (inclusive) -> versions in ascending order; ValueError names versions that have no pair."""
    if ".." in spec:
        lo, hi = (_key(x.strip()) for x in spec.split("..", 1))
        found = [v for v in available if lo <= _key(v) <= hi]
        if not found:
            have = ", ".join(sorted(available, key=_key))
            raise ValueError(f"no release with engine and control in range {spec}; have: {have}")
    else:
        found = [v.strip() for v in spec.split(",") if v.strip()]
        missing = [v for v in found if v not in available]
        if missing:
            raise ValueError(f"no release with engine and control for: {', '.join(missing)}")
    return sorted(set(found), key=_key)


def list_releases(repo: str = "Niko1221/Strata", opener=urllib.request.urlopen) -> list[dict]:
    """All releases of `repo`, newest first, via the public API (no token)."""
    out: list[dict] = []
    page = 1
    while True:
        req = urllib.request.Request(f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}",
                                     headers={"Accept": "application/vnd.github+json", "User-Agent": "strata-ab-engine"})
        with opener(req, timeout=30) as r:
            batch = json.load(r)
        if not batch:
            return out
        out += batch
        page += 1
