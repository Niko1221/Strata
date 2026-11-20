"""Read-only plan; explicit local/public release gates use the owned Job guard."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile

import prepare as p

sys.path.insert(0, str(p.ROOT))
from terminal.strata_control import engine_component as ec, feed, signing, updater


def gh_json(*args: str) -> dict:
    result = subprocess.run(["gh", *args], check=True, capture_output=True, timeout=90)
    return json.loads(result.stdout)


def stable_snapshot() -> dict:
    value = gh_json("api", "repos/zapabob/Strata/releases/latest")
    ref = gh_json("api", "repos/zapabob/Strata/git/ref/tags/latest")
    return {"id": value["id"], "tag": value["tag_name"], "prerelease": value["prerelease"],
            "draft": value["draft"], "latest_ref": ref["object"],
            "assets": {a["name"]: {k: a[k] for k in ("id", "size", "digest")} for a in value["assets"]}}


def docs_check(prepared: dict) -> None:
    names = sorted(p.ROOT.glob("README*.md"))
    for path in names:
        text = path.read_text(encoding="utf-8")
        prior = subprocess.check_output(["git", "show", "f8249a46:" + path.name], cwd=p.ROOT).decode("utf-8")
        marker = r"<!-- supporters:start.*?<!-- supporters:end -->"
        p.require(re.search(marker, text, re.S).group() == re.search(marker, prior, re.S).group(), "Supporter credits changed")
        p.require(p.INSTALLER in text and p.TAG in text and "releaseQualified=false" in text, "README preview scope missing")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(p.URL):
                p.require(target[len(p.URL):] in prepared["assets"], "README asset not in prepared set")
            elif "://" not in target and not target.startswith("#"):
                p.require((path.parent / urllib.parse.unquote(target.split("#")[0])).exists(), "Broken README local link")


def local() -> None:
    prepared = p.load(p.BASE / "prepared.json")
    p.require(prepared["ok"] and prepared["releaseQualified"] is False, "Prepared scope changed")
    for name, wanted in prepared["assets"].items():
        path = p.ASSETS / name
        p.require(path.stat().st_size == wanted["bytes"] and p.sha(path) == wanted["sha256"], "Prepared asset changed")
    source, quality = p.load(p.ASSETS / "SOURCE.json"), p.load(p.ASSETS / "QUALITY.json")
    p.require(source["recipe_sha256"] == p.sha(Path(p.__file__)), "Preparation recipe changed")
    p.require(quality["releaseQualified"] is False and len(quality["nativeResourceGuard"]["violations"]) == 12, "Resource limits lost")
    docs_check(prepared)
    trusted = signing.load_trusted_keys(environ={})
    document = feed.fetch(str(p.ASSETS), trusted, channel="stable")
    hardware = {"platform": "windows-x86_64", "backend": "cuda", "cuda_archs": ["sm_120"], "nvidia_driver": "621.01", "runtime": "0.5.5"}
    selected, _ = feed.select_engine(document.engine, hardware)
    p.require(selected and selected.version == "0.5.4" and selected.url == p.URL + p.ENGINE, "Wrong preview selection")
    p.require(feed.select_engine(document.engine, {**hardware, "cuda_archs": ["sm_86"]})[0] is None, "Unsupported architecture admitted")
    try:
        feed.verify_and_parse(document.raw + b" ", (p.ASSETS / "strata-feed.json.sig").read_bytes(), source=str(p.ASSETS), trusted=trusted)
    except feed.FeedError:
        pass
    else:
        raise RuntimeError("Tampered feed accepted")
    scratch = p.BASE / "validation-local"
    p.require(not scratch.exists(), "Retain prior local validation")
    scratch.mkdir()
    context = updater.Context(components=scratch / "components", source=str(p.ASSETS), trusted=trusted,
                              hardware=hardware, state_base=scratch / "state", serving=lambda: [], environ={})
    state = updater.State(context.components)
    updater.open_feed(context, state)
    stable_url = "https://github.com/zapabob/Strata/releases/download/v0.5.1"
    try:
        updater.open_feed(replace(context, source=stable_url), state)
    except feed.FeedDowngrade:
        pass
    else:
        raise RuntimeError("Cross-source replay guard failed")
    # Exercise production extraction/signature/hash/fit checks on the new tar.gz,
    # without activation, model execution or a synthetic passing test-start.
    _, component = updater.stage_engine(context, document, selected, p.ASSETS / p.ENGINE)
    p.require(p.sha(component.root / ec.MANIFEST) == source["native_manifest_sha256"], "Staged native binding changed")
    control = scratch / "control"
    with zipfile.ZipFile(p.ASSETS / p.CONTROL) as archive:
        archive.extractall(control)
    env = dict(os.environ)
    env.update(STRATA_NO_HANDOVER="1", STRATA_COMPONENTS_DIR=str(scratch / "cli-components"),
               STRATA_PROFILE_DIR=str(scratch / "profiles"), STRATA_RUNTIME_DIR=str(scratch / "runtime"),
               STRATA_DATA_DIR=str(scratch / "data"), STRATA_STATE_DIR=str(scratch / "state"))
    python = str(control / "python/python.exe")
    for args in (("--version",), ("update", "--check", "--no-engine", "--feed", str(p.ASSETS))):
        result = subprocess.run([python, "-B", str(control / "strata.py"), *args], env=env,
                                check=True, capture_output=True, timeout=120)
        p.require(b"0.5.5" in result.stdout or b"0.5.4" in result.stdout, "Packaged CLI did not report the preview")
    p.run(sys.executable, "-B", str(p.ROOT / "tools/bump_hakua_product_version.py"), "--check", seconds=120)
    p.run(sys.executable, "-B", "-m", "unittest", "tests.terminal.test_tui2_023_versions", seconds=180)
    before = stable_snapshot()
    p.require(before["tag"] == "v0.5.1" and not before["prerelease"] and not before["draft"], "Stable baseline changed")
    p.save(p.BASE / "stable-before.json", before)
    p.save(p.BASE / "verified-local.json", {"ok": True, "assets": len(prepared["assets"]), "supporterBlocksMatched": len(list(p.ROOT.glob("README*.md"))),
        "productionTarStageVerified": True, "packagedPrivatePythonCLI": True, "feedTamperRejected": True,
        "unsupportedArchitectureRejected": True, "crossSourceDowngradeRejected": True, "releaseQualified": False})


def public() -> None:
    prepared = p.load(p.BASE / "prepared.json")
    release = gh_json("api", "repos/zapabob/Strata/releases/tags/" + p.TAG)
    p.require(release["prerelease"] and not release["draft"], "Expected public prerelease")
    remote = {a["name"]: a for a in release["assets"]}
    p.require(set(remote) == set(prepared["assets"]), "Public membership differs")
    downloads = {}
    # No authentication, gh tokens or default feed overrides in anonymous reads.
    opener = urllib.request.build_opener()
    for name, expected in prepared["assets"].items():
        asset = remote[name]
        p.require(asset["state"] == "uploaded" and asset["size"] == expected["bytes"] and asset["digest"] == "sha256:" + expected["sha256"], "GitHub asset binding differs")
        request = urllib.request.Request(p.URL + name, headers={"User-Agent": "Hakua-release-readback", "Accept": "application/octet-stream"})
        digest, count = hashlib.sha256(), 0
        with opener.open(request, timeout=60) as response:
            p.require(response.status == 200 and response.url.startswith("https://"), "Anonymous download failed")
            while block := response.read(1024**2):
                count += len(block)
                p.require(count <= expected["bytes"], "Oversized public asset")
                digest.update(block)
        p.require(count == expected["bytes"] and digest.hexdigest() == expected["sha256"], "Anonymous bytes differ")
        downloads[name] = {"bytes": count, "sha256": digest.hexdigest(), "authenticated": False}
        print(json.dumps({"downloadVerified": name, "bytes": count}))
    trusted = signing.load_trusted_keys(environ={})
    document = feed.fetch(p.URL, trusted, channel="stable", timeout=60)
    p.require(document.sequence == 20261110000100 and document.engine[0].url == p.URL + p.ENGINE, "Public signed feed differs")
    p.require(stable_snapshot() == p.load(p.BASE / "stable-before.json"), "Stable release/alias/assets changed")
    local_tag = subprocess.check_output(["git", "rev-parse", p.TAG + "^{}"], cwd=p.ROOT).decode().strip()
    ref = gh_json("api", "repos/zapabob/Strata/git/ref/tags/" + p.TAG)["object"]
    p.require(ref["type"] == "tag", "Annotated tag required")
    tag = gh_json("api", "repos/zapabob/Strata/git/tags/" + ref["sha"])
    p.require(tag["object"]["sha"] == local_tag, "Remote tag target differs")
    p.save(p.BASE / "verified-public.json", {"ok": True, "tag": p.TAG, "tagTarget": local_tag,
        "url": release["html_url"], "releaseID": release["id"], "prerelease": True, "releaseQualified": False,
        "stableReleaseAliasAssetsUnchanged": True, "builtinFeedSignatureVerified": True, "anonymousDownloads": downloads})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "local", "public"), nargs="?", default="plan")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.action == "plan":
        print(json.dumps({"execute": False, "local": "production archive/CLI/feed/docs checks in isolated H: scratch", "public": "anonymous complete asset hashes + stable/tag readback"}))
        return 0
    p.require(args.execute, "Explicit execute required")
    guard = p.guard_module()
    if args.worker:
        guard.require_current_job(p.ROOT)
        {"local": local, "public": public}[args.action]()
        return 0
    env = guard.environment()
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", CARGO_BUILD_JOBS="5", CMAKE_BUILD_PARALLEL_LEVEL="5")
    gh = shutil.which("gh")
    p.require(gh, "Authenticated gh CLI required")
    env["PATH"] = str(Path(gh).parent) + ";" + env["PATH"]
    row = guard.owned(p.ROOT, "verify-release-" + args.action + "-r1", [sys.executable, "-B", str(Path(__file__).resolve()), args.action, "--execute", "--worker"],
                      1800, env=env, output_bytes=8 * 1024**2)
    print(json.dumps(row, indent=2))
    if not row["passed"]:
        print(guard.text("verify-release-" + args.action + "-r1"))
    return 0 if row["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
