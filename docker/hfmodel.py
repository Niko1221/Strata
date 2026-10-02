#!/usr/bin/env python3
"""Find a Strata model inside the host's Hugging Face cache - the container's only model source.

The container never downloads.  It reads whatever `hf download` already put in the HF cache, mounted
read-only.  Two layout facts drive this code (both verified on the reference host):

* The cache is  hub/models--<org>--<repo>/snapshots/<rev>/<path>  where each file is a **relative
  symlink** into  ../../blobs/<sha256>.  So the whole  hub/  directory must be mounted, not just a
  snapshot folder, or every shard resolves to a dangling link.  Paths are therefore reported *inside*
  the mount (e.g. /hf-cache) and not resolved to the blob, which also keeps the real HF filename in
  the engine log.  `--resolve` prints blob paths instead, for debugging.
* Model families, quantisations and file names mirror setup.py's FAMILIES/MODELS (setup.py:64-105);
  keep them in sync when upstream renames a release.

Examples:

    hfmodel.py --print shard1                     # IQ3_XXS shard 1, or a helpful error
    hfmodel.py --model IQ2_XS --print json
    hfmodel.py --print available                  # what Strata can actually run from this cache
    hfmodel.py --cache /hf-cache --model IQ3_XXS --print both
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# setup.py:83-105.  "subdir" = the GGUFs live under <quant>/ inside the repo.
FAMILIES = {
    "qwen": {"repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
             "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "subdir": True,
             "title": "Qwen3.8-Flash-Next"},
    "swift": {"repo": "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
              "file": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "subdir": False,
              "title": "Swift 1.5"},
    "coder": {"repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF",
              "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "subdir": True,
              "title": "Qwen3.8-Flash-Next Coder"},
}
# The sizes Strata knows: quant -> family, download GB, RAM GB, experts.bin GB (setup.py:64-79).
# download_gb is what `hf download` writes into the cache; arena_gb is what iq_pack.py --experts-bin
# adds on top of it (the HIP mmap path needs that file, docs/AMD_HIP.md:64-67).  Both feed the
# free-space gate in docker/bootstrap-model.sh.  IQ3_XXS is the default for the gfx1101 container:
# the best quality this card's RAM/VRAM budget takes comfortably.
MODELS = {
    "IQ3_XXS": {"family": "qwen", "download_gb": 75.8, "ram_gb": 60, "arena_gb": 42.9},
    "IQ3_S": {"family": "qwen", "download_gb": 83.6, "ram_gb": 62, "arena_gb": 50.3},
    "Q2_0": {"family": "qwen", "download_gb": 66.4, "ram_gb": 48, "arena_gb": 34.0},
    "IQ2_XS": {"family": "qwen", "download_gb": 68.0, "ram_gb": 48, "arena_gb": 35.5},
    "IQ1_M": {"family": "coder", "download_gb": 58.4, "ram_gb": 32, "arena_gb": 23.4},
}
DEFAULT_MODEL = "IQ3_XXS"


def cache_root(cli: str | None) -> Path:
    """Cache root: --cache, then $HF_HUB_CACHE / $HF_HOME/hub, then the location ./run.sh defaults
    to when it exists (the big disk), then the stock ~/.cache/huggingface/hub."""
    if cli:
        return Path(cli)
    for var in ("HF_HUB_CACHE", "HF_CACHE_HOME"):
        if os.environ.get(var):
            return Path(os.environ[var])
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    big = Path.home() / "Development" / "models"
    if big.is_dir():
        return big
    return Path.home() / ".cache" / "huggingface" / "hub"


def repo_dir(root: Path, repo: str) -> Path:
    return root / ("models--" + repo.replace("/", "--"))


def snapshot_dir(rd: Path, rev: str = "") -> Path | None:
    """The revision's snapshot: refs/<branch> if present, else --rev, else the newest snapshot dir."""
    snaps = rd / "snapshots"
    if not snaps.is_dir():
        return None
    if not rev and (rd / "refs" / "main").is_file():
        rev = (rd / "refs" / "main").read_text().strip()
    if rev:
        for cand in (snaps / rev, snaps / rev[:40]):
            if cand.is_dir():
                return cand
    dirs = sorted((d for d in snaps.iterdir() if d.is_dir()), key=lambda d: d.name)
    return dirs[-1] if dirs else None


def family_of(model: str, repo: str) -> dict | None:
    """The release this quant belongs to: by repo id if given, else by the quant's family."""
    if repo:
        for fam in FAMILIES.values():
            if fam["repo"] == repo:
                return fam
        # A release that is not in setup.py: the layout is unknown, so discover the shards by glob.
        return {"repo": repo, "file": None, "subdir": True, "title": repo}
    info = MODELS.get(model)
    return FAMILIES[info["family"]] if info else None


def shard_path(root: Path, model: str, i: int, rev: str = "", repo: str = "") -> Path | None:
    """The path of shard `i` (1 = the experts, 2 = the PLE table) inside the cache, or None."""
    fam = family_of(model, repo)
    if fam is None:
        return None
    snap = snapshot_dir(repo_dir(root, fam["repo"]), rev)
    if snap is None:
        return None
    if fam["file"]:
        name = fam["file"].format(q=model, i=i)
        cand = snap / (f"{model}/{name}" if fam["subdir"] else name)
        if cand.is_file():
            return cand
    # Upstream renamed something, or an unknown repo: find the shard by its naming convention instead.
    for pat in (f"{model}/*{model}*0000{i}*.gguf", f"*{model}*0000{i}*.gguf", f"*0000{i}-of-00002.gguf"):
        hits = sorted(h for h in snap.glob(pat) if h.is_file())
        if hits:
            return hits[-1]
    return None


def present(root: Path, model: str, rev: str = "") -> tuple[bool, bool]:
    return (shard_path(root, model, 1, rev) is not None, shard_path(root, model, 2, rev) is not None)


def available(root: Path) -> list[dict]:
    out = []
    for model, info in MODELS.items():
        rd = repo_dir(root, FAMILIES[info["family"]]["repo"])
        have1, have2 = present(root, model)
        out.append({"model": model, "repo": FAMILIES[info["family"]]["repo"], "repo_in_cache": rd.is_dir(),
                    "shard1": have1, "shard2": have2, "download_gb": info["download_gb"],
                    "ram_gb": info["ram_gb"]})
    return out


def describe_available(root: Path) -> str:
    rows = available(root)
    lines = [f"{r['model']:8s} {r['repo']:58s} "
             + ("both shards" if r["shard1"] and r["shard2"] else
                "shard 1 only" if r["shard1"] else "repo in cache, files missing" if r["repo_in_cache"]
                else "not in cache")
             + f"   ({r['download_gb']} GB download)" for r in rows]
    other = ""
    if root.is_dir():
        extras = sorted(d.name for d in root.iterdir()
                        if d.is_dir() and d.name.startswith("models--")
                        and d.name not in {("models--" + r["repo"].replace("/", "--")) for r in rows})
        if extras:
            other = ("\n  Other repos in this cache (Strata cannot run them - it needs one of the "
                     "releases above):\n    " + "\n    ".join(extras))
    return "\n".join(lines) + other


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="", help="model cache root (default: $HF_HUB_CACHE, else "
                    "~/Development/models when it exists, else ~/.cache/huggingface/hub)")
    ap.add_argument("--model", default=os.environ.get("STRATA_MODEL", DEFAULT_MODEL))
    ap.add_argument("--repo", default="", help="override the repo id for a release not in setup.py")
    ap.add_argument("--rev", default="", help="revision/commit (default: refs/main or newest snapshot)")
    ap.add_argument("--print", dest="what", default="both",
                    choices=["shard1", "shard2", "both", "json", "shell", "available"])
    ap.add_argument("--allow-missing", action="store_true",
                    help="with shell/json: exit 0 and empty paths when the shards are not cached yet")
    ap.add_argument("--resolve", action="store_true", help="print resolved blob paths instead of snapshot paths")
    a = ap.parse_args()

    root = cache_root(a.cache or None)
    if a.what == "available":
        print(f"HF cache: {root}\n{describe_available(root)}")
        return 0
    if a.model not in MODELS and not a.repo:
        sys.exit(f"hfmodel: unknown model '{a.model}'. Known: {', '.join(sorted(MODELS))}")

    want = {"shard1": [1], "shard2": [2], "both": [1, 2], "json": [1, 2], "shell": [1, 2]}[a.what]
    paths = {i: shard_path(root, a.model, i, a.rev, a.repo) for i in want}
    fam = family_of(a.model, a.repo) or FAMILIES["qwen"]
    info = MODELS.get(a.model, {"download_gb": "?", "ram_gb": "?", "arena_gb": "?"})
    include = f"{a.model}/*" if fam["subdir"] else f"*{a.model}*.gguf"
    missing = [i for i, p in paths.items() if p is None]
    if missing and not a.allow_missing:
        names = ", ".join(f"shard {i}" for i in missing)
        print(f"hfmodel: {names} of '{a.model}' not found in {root}", file=sys.stderr)
        print(f"  repo asked for: {fam['repo']}; looked for "
              f"<snapshot>/{a.model}/*{a.model}*0000N*.gguf and *{a.model}*0000N*.gguf", file=sys.stderr)
        print(f"  {describe_available(root)}", file=sys.stderr)
        print(f"\n  Fetch it (docker/bootstrap-model.sh does this on first start unless "
              f"STRATA_DOWNLOAD_MODEL=0), e.g.:\n    hf download {fam['repo']} --include '{include}'\n"
              f"  {info['download_gb']} GB: check the free space first, and set HF_HUB_CACHE if the "
              f"default cache's filesystem is too small.", file=sys.stderr)
        return 1

    def shown(i: int) -> str:
        p = paths.get(i)
        return "" if p is None else str(p.resolve() if a.resolve else p)

    if a.what == "shell":       # eval'able assignments for bootstrap-model.sh / the entrypoint
        import shlex
        for key, value in {"MODEL": a.model, "CACHED": int(not missing), "REPO": fam["repo"],
                           "HF_CACHE": str(root), "SHARD1": shown(1), "SHARD2": shown(2),
                           "DOWNLOAD_GB": info["download_gb"], "ARENA_GB": info["arena_gb"],
                           "RAM_GB": info["ram_gb"], "HF_INCLUDE": include}.items():
            print(f"STRATA_{key}={shlex.quote(str(value))}")
        return 0

    out = {f"shard{i}": shown(i) for i in want}
    if a.what == "json":
        out.update({"model": a.model, "repo": fam["repo"], "cached": not missing, "cache": str(root),
                    "download_gb": info["download_gb"], "ram_gb": info["ram_gb"],
                    "arena_gb": info["arena_gb"], "hf_include": include})
        print(json.dumps(out, indent=1))
    elif a.what == "both":
        print(shown(1) + "\n" + shown(2))
    else:
        print(shown(want[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
