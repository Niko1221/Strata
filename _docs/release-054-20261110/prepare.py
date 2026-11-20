"""Prepare and verify the measured preview assets. Default plan has no writes.

No model execution, installation, source rewrite or key inspection occurs here.
Only the production feed signer reads the explicitly supplied private-key path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("H:/Strata/releases/public-054-20261110-opencode")
ASSETS = BASE / "assets"
BUILD = Path("H:/Strata/releases/upstream-0213-20261109-opencode/app-native054")
VERSION = "0.5.4-local.1"
TAG = "v" + VERSION
URL = "https://github.com/zapabob/Strata/releases/download/" + TAG + "/"
ENGINE = "strata-engine-0.5.4-windows-x86_64-cuda-cuda13.tar.gz"
CONTROL = "strata-control-0.5.5-windows-x86_64.zip"
INSTALLER = f"Strata-{VERSION}-windows-x64-setup.exe"
ARCHIVE = f"Strata-{VERSION}-windows-x64.tar.7z"
SEVEN = Path("C:/Program Files/7-Zip/7z.exe")


def require(value: Any, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def run(*args: str, seconds: int = 1800, quiet: bool = False) -> None:
    subprocess.run(list(args), cwd=ROOT, check=True, timeout=seconds,
                   stdout=subprocess.DEVNULL if quiet else None,
                   stderr=subprocess.DEVNULL if quiet else None)


def guard_module():
    path = ROOT / "_docs/local-053-20261109/native/common.py"
    spec = importlib.util.spec_from_file_location("release_guard", path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.BASE = BASE
    return guard


def signed_source() -> tuple[Path, dict, dict]:
    sys.path.insert(0, str(ROOT))
    from terminal.strata_control import engine_component, signing
    receipt, build = load(BUILD / "staged-native.json"), load(BUILD / "app-build.json")
    require(receipt["ok"] and receipt["signatureVerified"] and receipt["releaseQualified"] is False, "Unknown native staging")
    require(build["sourceCommit"] == "85a78234cfcdccd8741cc921677802aba1869e44", "Unknown tested app source")
    for name in ("installed-q36-e2e.json", "installed-prism-e2e.json", "normal-startup.json"):
        require(load(BUILD / name)["ok"] is True, "Required installed evidence missing")
    folder = ROOT / "apps/desktop/src-tauri/resources/engine-bundle/0.5.4-windows-x86_64-cuda-cuda13"
    require(sha(folder / "engine-component.json") == receipt["signedManifestSHA256"], "Signed manifest changed")
    component = engine_component.read_manifest(folder)
    engine_component.verify(component, full=True)
    document = load(folder / "engine-component.json")
    signing.verify_bytes(signing.load_trusted_keys(environ={}), engine_component.signing_payload(document), document["signature"])
    require(sha(Path(build["installer"])) == build["sha256"] == "ae05518656ee5afdf92665f5dd71a06a69425ae1957a9ac31468d8a4a1790750", "Tested installer changed")
    return folder, receipt, build


def verify_archives(folder: Path, receipt: dict) -> None:
    manifest = load(folder / "engine-component.json")
    expected = {row["path"]: (row["bytes"], row["sha256"]) for row in manifest["files"]}
    expected["engine-component.json"] = ((folder / "engine-component.json").stat().st_size, sha(folder / "engine-component.json"))
    found = {}
    with tarfile.open(ASSETS / ENGINE, "r|gz") as archive:
        for member in archive:
            require(member.isfile() and member.name in expected and member.name not in found, "Unexpected engine archive member")
            stream = archive.extractfile(member)
            found[member.name] = (member.size, hashlib.file_digest(stream, "sha256").hexdigest())
    require(found == expected, "Engine tar.gz differs from the signed measured payload")
    with zipfile.ZipFile(ASSETS / CONTROL) as archive:
        require(archive.testzip() is None, "Control ZIP CRC failed")
        names = [n for n in archive.namelist() if not n.endswith("/")]
        require(len(names) == len(set(names)) and "strata.py" in names and "COMMIT" in names, "Invalid Control membership")
        require(archive.read("COMMIT").decode().strip() == "85a78234cfcdccd8741cc921677802aba1869e44", "Control source changed")
        require(archive.read("VERSION").decode().strip() == "0.5.5", "Control version changed")
        control_root = ROOT / "apps/desktop/src-tauri/resources/control"
        expected_control = {p.relative_to(control_root).as_posix(): (p.stat().st_size, sha(p))
                            for p in control_root.rglob("*") if p.is_file()}
        require(set(names) == set(expected_control), "Control ZIP membership differs from staged Control")
        for name in names:
            with archive.open(name) as stream:
                actual = (archive.getinfo(name).file_size, hashlib.file_digest(stream, "sha256").hexdigest())
            require(actual == expected_control[name], "Control ZIP payload differs")
        require(not any(n.lower().endswith((".db", ".sqlite", ".log")) or "/.env" in "/" + n.lower() for n in names), "Private payload in Control")
        ggufs = [n for n in names if n.lower().endswith(".gguf")]
        require(ggufs == ["data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf"], "Unreviewed model data in Control")
    require(sha(folder / "engine-component.json") == receipt["signedManifestSHA256"], "Original component changed during packaging")


def prepare(key: str) -> None:
    guard = guard_module()
    guard.require_current_job(ROOT)
    folder, receipt, build = signed_source()
    copy = ASSETS / INSTALLER
    shutil.copy2(build["installer"], copy)
    require(sha(copy) == build["sha256"], "Installer copy differs")
    tool = str(ROOT / "packaging/feed_tool.py")
    run(sys.executable, "-B", tool, "pack-engine", str(folder), "--out", str(ASSETS), "--format", "tar.gz", "--url-prefix=" + URL)
    control = ROOT / "apps/desktop/src-tauri/resources/control"
    require((control / "COMMIT").read_text().strip() == build["controlSourceCommit"], "Staged Control source changed")
    run(sys.executable, "-B", tool, "pack-control", str(control), "--out", str(ASSETS), "--version", "0.5.5", "--platform", "windows-x86_64", "--format", "zip", "--url-prefix=" + URL)
    engine_dir, control_dir = BASE / "engine-entries", BASE / "control-entries"
    engine_dir.mkdir(); control_dir.mkdir()
    engine_entry = load(ASSETS / (ENGINE + ".entry.json"))
    engine_entry["notes"] = "Preview v0.5.4-local.1; releaseQualified=false; CUDA resource guard has 12 violations. See QUALITY.json."
    save(engine_dir / "engine.entry.json", engine_entry)
    shutil.copy2(ASSETS / (CONTROL + ".entry.json"), control_dir / "control.entry.json")
    # The unchanged clients use the stable transport channel. Opt-in is the
    # immutable prerelease URL; it is never the default Latest endpoint.
    run(sys.executable, "-B", tool, "build-feed", "--out", str(ASSETS), "--sequence", "20261110000100", "--channel", "stable", "--expires-days", "365",
        "--engine", str(engine_dir), "--control", str(control_dir), "--key", key, quiet=True)
    from terminal.strata_control import feed, signing
    verified = feed.fetch(str(ASSETS), signing.load_trusted_keys(environ={}))
    require(verified.sequence == 20261110000100 and len(verified.engine) == len(verified.control) == 1, "Feed verification failed")
    verify_archives(folder, receipt)
    for name in ("install.ps1", "install.sh"):
        shutil.copy2(ROOT / "packaging" / name, ASSETS / name)
    shutil.copy2(ROOT / "_docs/release-054-20261110/INSTALL.md", ASSETS / "INSTALL.md")
    quality = receipt["quality"]
    save(ASSETS / "QUALITY.json", {"release_tag": TAG, "releaseQualified": False, "nativeFunctionalChecksPassed": True,
        "nativeResourceGuard": quality["nativeResourceGuard"], "actualModelChecks": quality["actualModelChecks"],
        "fullWindowQualified": False, "generalQualityQualified": False, "lowPrecisionModelQualityQualified": False,
        "q38ProjectionQualified": False, "prismRestartWarmupTimeoutObserved": True,
        "installedPayloadFilesMatched": 29544, "userFilesPreservedDuringMeasuredUpgrade": 4028})
    save(ASSETS / "SOURCE.json", {"release_tag": TAG, "product": VERSION, "control": "0.5.5", "native": "0.5.4",
        "upstream": "f33c8c30588ff00e24b6b63bad0bf6a14da19081", "app_source": build["appSourceCommit"], "control_source": build["controlSourceCommit"],
        "native_source": receipt["nativeSourceCommit"], "installer_sha256": build["sha256"], "native_manifest_sha256": receipt["signedManifestSHA256"],
        "recipe": "_docs/release-054-20261110/prepare.py", "recipe_sha256": sha(Path(__file__))})
    members = [INSTALLER, ENGINE, CONTROL, "strata-feed.json", "strata-feed.json.sig", "install-manifest.txt", "install.ps1", "install.sh", "INSTALL.md", "QUALITY.json", "SOURCE.json"]
    checks = "".join(f"{sha(ASSETS / n)}  {n}\n" for n in members)
    (ASSETS / "BUNDLE-SHA256SUMS.txt").write_text(checks, encoding="utf-8", newline="\n")
    members.append("BUNDLE-SHA256SUMS.txt")
    expected = {n: {"bytes": (ASSETS / n).stat().st_size, "sha256": sha(ASSETS / n)} for n in members}
    tar_path = BASE / f"Strata-{VERSION}-windows-x64.tar"
    with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as archive:
        for name in members:
            path = ASSETS / name
            info = archive.gettarinfo(str(path), arcname=f"Strata-{VERSION}-windows-x64/{name}")
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    run(str(SEVEN), "a", "-t7z", "-mx=5", "-md=64m", "-mmt=5", str(ASSETS / ARCHIVE), str(tar_path), seconds=3600)
    run(str(SEVEN), "t", str(ASSETS / ARCHIVE), seconds=900)
    process = subprocess.Popen([str(SEVEN), "x", "-so", str(ASSETS / ARCHIVE), tar_path.name], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    found = {}
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        for member in archive:
            prefix = f"Strata-{VERSION}-windows-x64/"
            require(member.name.startswith(prefix) and member.isfile(), "Unsafe combined archive member")
            name = member.name[len(prefix):]
            require(name in expected and name not in found, "Unexpected combined archive membership")
            found[name] = {"bytes": member.size, "sha256": hashlib.file_digest(archive.extractfile(member), "sha256").hexdigest()}
    while process.stdout.read(1024**2):
        pass  # Drain tar padding so the decompressor exits normally.
    process.stdout.close()
    require(process.wait(timeout=60) == 0 and found == expected, "Combined archive readback differs")
    public = members + [ARCHIVE]
    require(all((ASSETS / n).stat().st_size < 2 * 1024**3 for n in public), "GitHub per-asset bound exceeded")
    (ASSETS / "SHA256SUMS.txt").write_text("".join(f"{sha(ASSETS / n)}  {n}\n" for n in public), encoding="utf-8", newline="\n")
    public.append("SHA256SUMS.txt")
    result = {"ok": True, "tag": TAG, "prerelease": True, "releaseQualified": False, "bundleReadbackMatched": True,
              "assets": {n: {"bytes": (ASSETS / n).stat().st_size, "sha256": sha(ASSETS / n)} for n in public}}
    save(BASE / "prepared.json", result)
    print(json.dumps(result, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "_worker"), default="plan", nargs="?")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--key", type=str)
    args = parser.parse_args()
    if args.action == "plan":
        print(json.dumps({"execute": False, "tag": TAG, "product": VERSION, "out": str(BASE), "prerelease": True,
                          "releaseQualified": False, "installerRebuilt": False, "engineSeparate": True}, indent=2))
        return 0
    require(args.execute and args.key and Path(args.key).is_absolute(), "Explicit execute and opaque absolute signer key path required")
    require(not Path(args.key).is_relative_to(ROOT) and not Path(args.key).is_relative_to(BASE), "Signer key must be outside source/output")
    if args.action == "_worker":
        prepare(args.key)
        return 0
    require(BASE.parent.is_dir() and not BASE.exists(), "Fresh approved output root required; retain prior attempts")
    BASE.mkdir()
    for name in ("assets", "receipts", "temp", "cache"):
        (BASE / name).mkdir()
    guard = guard_module()
    env = guard.environment()
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", CARGO_BUILD_JOBS="5", CMAKE_BUILD_PARALLEL_LEVEL="5")
    row = guard.owned(ROOT, "prepare-release", [sys.executable, "-B", str(Path(__file__).resolve()), "_worker", "--execute", "--key", args.key],
                      5400, env=env, output_bytes=16 * 1024**2)
    print(json.dumps(row, indent=2))
    if not row["passed"]:
        print(guard.text("prepare-release"))
    return 0 if row["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
