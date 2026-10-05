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

# setup.py:83-105.  "subdir" = the GGUFs live under <quant>/ inside the repo.  "ple" is the file
# holding per_layer_token_embd.weight (default 2); the Swift repo packs that table into file 1 and
# puts expert continuation in file 2 (measured on Swift IQ3_XXS on this host, 2026-10-04:
# bench/results/2026-10-04-iq3s-tuning/swift-launch.log - the engine finds the 320,001,536-row
# table in 00001 and fatal-errors that the tensor "is not in" 00002).  "tag" prefixes pack
# directories (packs/swift-iq3_xxs beside qwen's packs/iq3_xxs - one quant, two releases, one
# pack each), "name" prefixes the advertised model id, and "license" is what the launcher logs
# before starting a release that is not the original.  Mirrors setup.py's FAMILIES (name/tag/
# license); a new release is incomplete without that sync.
FAMILIES = {
    "qwen": {"repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
             "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "subdir": True,
             "title": "Qwen3.8-Flash-Next", "tag": "", "name": "qwen3.8-flash-next"},
    "swift": {"repo": "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
              "file": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "subdir": False,
              "title": "Swift 1.5", "ple": 1, "tag": "swift-", "name": "swift-1.5",
              "license": "Swift Open License 1.0: https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"},
    "coder": {"repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF",
              "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "subdir": True,
              "title": "Qwen3.8-Flash-Next Coder", "tag": "coder-",
              "name": "qwen3.8-flash-next-coder"},
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


def family_of(model: str, repo: str, release: str = "") -> dict | None:
    """The release this quant belongs to: named release, else by repo id, else by the quant."""
    if release:
        return FAMILIES.get(release)
    if repo:
        for fam in FAMILIES.values():
            if fam["repo"] == repo:
                return fam
        # A release that is not in setup.py: the layout is unknown, so discover the shards by glob.
        return {"repo": repo, "file": None, "subdir": True, "title": repo}
    info = MODELS.get(model)
    return FAMILIES[info["family"]] if info else None


def release_of(model: str, repo: str = "", release: str = "") -> str:
    """The FAMILIES key for a selection; "" when it names a repo the table does not know."""
    if release:
        return release if release in FAMILIES else ""
    for key, fam in FAMILIES.items():
        if repo:
            if fam["repo"] == repo:
                return key
        elif model in MODELS and MODELS[model]["family"] == key:
            return key
    return ""


def n_files(fam: dict) -> int:
    """How many GGUF files this release has: 2 for the Qwen3.8 family (experts + PLE), 1 for a
    single-file release."""
    return int(fam.get("files", 2))


def ple_index(fam: dict) -> int:
    """Which 1-based file of the release holds per_layer_token_embd.weight (Swift: 1, others: 2)."""
    return int(fam.get("ple", 2))


def _find_in_snapshot(snap: Path, model: str, i: int, fam: dict) -> Path | None:
    """One release file inside a snapshot directory.  The exact name first, then a naming-convention
    fallback.  A release whose naming the table knows matches ONLY its own files: the quant name
    must appear in the file name, or the file does not count as this quant (measured 2026-10-05:
    the old catch-all `*0000{i}-of-00002.gguf` let a request for swift IQ2_XS resolve the swift
    IQ3_XXS shard, so the launcher reported 'cached' and would have advertised one quantization
    while loading another).  Only for a release the table has not been taught (no file pattern)
    is any shard of that position accepted."""
    if n_files(fam) == 1:
        if fam.get("single_name") and fam.get("file"):
            cand = snap / fam["file"]
            if cand.is_file():
                return cand
        if fam.get("title"):
            hits = sorted(h for h in snap.glob(f"{fam['title']}-*.gguf") if h.is_file())
            if hits:
                return hits[0]
        hits = sorted(h for h in snap.glob(f"*{model}*.gguf") if h.is_file())
        return hits[0] if hits else None
    if fam.get("file"):
        name = fam["file"].format(q=model, i=i)
        cand = snap / (f"{model}/{name}" if fam["subdir"] else name)
        if cand.is_file():
            return cand
        for pat in (f"{model}/*{model}*0000{i}*.gguf", f"*{model}*0000{i}*.gguf"):
            hits = sorted(h for h in snap.glob(pat) if h.is_file())
            if hits:
                return hits[-1]
        return None
    for pat in (f"{model}/*{model}*0000{i}*.gguf", f"*{model}*0000{i}*.gguf",
                f"*0000{i}-of-00002.gguf"):
        hits = sorted(h for h in snap.glob(pat) if h.is_file())
        if hits:
            return hits[-1]
    return None


def shard_path(root: Path, model: str, i: int, rev: str = "", repo: str = "",
               release: str = "") -> Path | None:
    """The path of file `i` (1 = the model, 2 = Qwen3.8's PLE shard) inside the cache, or None.

    A one-file release returns its single GGUF for i == 1 and None for i == 2."""
    fam = family_of(model, repo, release)
    if fam is None:
        return None
    if n_files(fam) == 1 and i != 1:
        return None
    snap = snapshot_dir(repo_dir(root, fam["repo"]), rev)
    if snap is None:
        return None
    return _find_in_snapshot(snap, model, i, fam)


def mtp_path(root: Path, model: str, rev: str = "", repo: str = "",
             release: str = "") -> Path | None:
    """The external MTP draft GGUF for a family that declares one, or None."""
    fam = family_of(model, repo, release)
    if not fam or not fam.get("mtp"):
        return None
    m = fam["mtp"]
    snap = snapshot_dir(repo_dir(root, m["repo"]), rev)
    if snap is None:
        return None
    cand = snap / m["file"]
    return cand if cand.is_file() else None


def present(root: Path, model: str, rev: str = "") -> tuple[bool, bool]:
    have1 = shard_path(root, model, 1, rev) is not None
    return (have1, shard_path(root, model, 2, rev) is not None)


def available(root: Path) -> list[dict]:
    out = []
    for model, info in MODELS.items():
        fam = FAMILIES[info["family"]]
        rd = repo_dir(root, fam["repo"])
        have1, have2 = present(root, model)
        out.append({"model": model, "repo": fam["repo"], "repo_in_cache": rd.is_dir(),
                    "shard1": have1, "shard2": have2, "files": n_files(fam),
                    "mtp": mtp_path(root, model) is not None,
                    "download_gb": info["download_gb"], "ram_gb": info["ram_gb"]})
    return out


def describe_available(root: Path) -> str:
    rows = available(root)
    lines = []
    for r in rows:
        if r["files"] == 1:
            state = ("model file" if r["shard1"] else
                     "repo in cache, file missing" if r["repo_in_cache"] else "not in cache")
            if r["shard1"] and r["mtp"]:
                state += " + MTP"
            elif r["shard1"]:
                state += " (no MTP draft)"
        else:
            state = ("both shards" if r["shard1"] and r["shard2"] else
                     "shard 1 only" if r["shard1"] else
                     "repo in cache, files missing" if r["repo_in_cache"] else "not in cache")
        lines.append(f"{r['model']:8s} {r['repo']:58s} {state}   ({r['download_gb']} GB download)")
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
    ap.add_argument("--release", default="", choices=["", *sorted(FAMILIES)],
                    help="named release (qwen, swift, coder); beats --repo and the quant's own family")
    ap.add_argument("--rev", default="", help="revision/commit (default: refs/main or newest snapshot)")
    ap.add_argument("--print", dest="what", default="both",
                    choices=["shard1", "shard2", "both", "json", "shell", "available", "mtp", "ple",
                             "release"])
    ap.add_argument("--allow-missing", action="store_true",
                    help="with shell/json: exit 0 and empty paths when the shards are not cached yet")
    ap.add_argument("--resolve", action="store_true", help="print resolved blob paths instead of snapshot paths")
    a = ap.parse_args()

    root = cache_root(a.cache or None)
    if a.what == "available":
        print(f"HF cache: {root}\n{describe_available(root)}")
        return 0
    if a.release and release_of(a.model, a.repo, a.release) != a.release:
        sys.exit(f"hfmodel: unknown release '{a.release}'. Known: {', '.join(sorted(FAMILIES))}")
    # The Coder release is expert-pruned: its quantizations are a different model, and its quant
    # names are its own (docs: half of each layer's experts removed).  The other releases share
    # the quant vocabulary, so a shared name under another named release is allowed - with a
    # warning, because the download numbers then come from the qwen row of MODELS.
    if a.model in MODELS and MODELS[a.model]["family"] == "coder" \
            and (a.release and a.release != "coder") and not a.repo:
        sys.exit(f"hfmodel: {a.model} is only released for the expert-pruned 'coder' release, "
                 f"not '{a.release}'; use --release coder")
    if a.model in MODELS and a.release and MODELS[a.model]["family"] != a.release:
        print(f"hfmodel: note: {a.model} is catalogued for '{MODELS[a.model]['family']}'; using "
              f"'{a.release}' names and sizes as asked", file=sys.stderr)
    if a.model not in MODELS and not a.repo and not a.release:
        sys.exit(f"hfmodel: unknown model '{a.model}'. Known: {', '.join(sorted(MODELS))}")

    if a.what == "release":
        print(release_of(a.model, a.repo, a.release))
        return 0

    if a.what == "mtp":
        p = mtp_path(root, a.model, a.rev, a.repo, a.release)
        print("" if p is None else str(p.resolve() if a.resolve else p))
        if p is None and not a.allow_missing:
            fam = family_of(a.model, a.repo, a.release) or {}
            m = fam.get("mtp", {})
            if m:
                print(f"hfmodel: the MTP draft ({m['repo']} / {m['file']}) is not in {root}", file=sys.stderr)
                print(f"  hf download {m['repo']} --include '{m['file']}'", file=sys.stderr)
                return 1
        return 0

    fam = family_of(a.model, a.repo, a.release) or FAMILIES["qwen"]
    if a.what == "ple":
        p = shard_path(root, a.model, min(ple_index(fam), n_files(fam)), a.rev, a.repo, a.release)
        print("" if p is None else str(p.resolve() if a.resolve else p))
        return 0 if p is not None or a.allow_missing else 1
    want = {"shard1": [1], "shard2": [2], "both": [1, 2], "json": [1, 2], "shell": [1, 2]}[a.what]
    info = MODELS.get(a.model, {"download_gb": "?", "ram_gb": "?", "arena_gb": "?"})
    paths = {i: shard_path(root, a.model, i, a.rev, a.repo, a.release) for i in want}
    if n_files(fam) == 1 and fam.get("single_name") and fam.get("file"):
        include = fam["file"]
    else:
        include = f"{a.model}/*" if fam["subdir"] else f"*{a.model}*.gguf"
    missing = [i for i, p in paths.items() if p is None and i <= n_files(fam)]
    if missing and not a.allow_missing:
        names = ", ".join(f"file {i}" for i in missing)
        print(f"hfmodel: {names} of '{a.model}' not found in {root}", file=sys.stderr)
        if n_files(fam) == 1:
            print(f"  repo asked for: {fam['repo']}; looked for {fam.get('file', '<title>-*.gguf')}", file=sys.stderr)
        else:
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
        m = fam.get("mtp", {})
        mp = mtp_path(root, a.model, a.rev, a.repo, a.release)
        quant_lower = a.model.lower()
        for key, value in {"MODEL": a.model, "CACHED": int(not missing), "REPO": fam["repo"],
                           "FAMILY": release_of(a.model, a.repo, a.release),
                           "PACK_TAG": fam.get("tag", ""),
                           "MODEL_NAME_DEFAULT": f"{fam.get('name', fam.get('title', fam['repo']))}-{quant_lower}",
                           "LICENSE": fam.get("license", ""),
                           "HF_CACHE": str(root), "SHARD1": shown(1), "SHARD2": shown(2),
                           "PLE_FILE": shown(min(ple_index(fam), n_files(fam))),
                           "FILES": n_files(fam),
                           "DOWNLOAD_GB": info["download_gb"], "ARENA_GB": info["arena_gb"],
                           "RAM_GB": info["ram_gb"], "HF_INCLUDE": include,
                           "MTP_REPO": m.get("repo", ""), "MTP_FILE": m.get("file", ""),
                           # NOT "MTP": entrypoint-hip.sh evaluates this output and keeps its own
                           # STRATA_MTP (the packed runtime directory).  A same-named key here would
                           # clobber it and silently disable --spec for the Qwen3.8 packs.
                           "MTP_GGUF": "" if mp is None else (str(mp.resolve()) if a.resolve else str(mp)),
                           "MTP_GGUF_CACHED": int(mp is not None),
                           "MTP_DOWNLOAD_GB": m.get("download_gb", 0)}.items():
            print(f"STRATA_{key}={shlex.quote(str(value))}")
        return 0

    out = {f"shard{i}": shown(i) for i in want}
    if a.what == "json":
        out.update({"model": a.model, "repo": fam["repo"], "cached": not missing, "cache": str(root),
                    "download_gb": info["download_gb"], "ram_gb": info["ram_gb"],
                    "arena_gb": info["arena_gb"], "hf_include": include, "files": n_files(fam),
                    "release": release_of(a.model, a.repo, a.release),
                    "mtp": "" if mtp_path(root, a.model, a.rev, a.repo, a.release) is None else
                           str(mtp_path(root, a.model, a.rev, a.repo, a.release))})
        print(json.dumps(out, indent=1))
    elif a.what == "both":
        print(shown(1) + "\n" + shown(2))
    else:
        print(shown(want[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
