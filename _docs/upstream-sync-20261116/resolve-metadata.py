"""Resolve reviewed release/version and installer diff3 conflicts for fork 0.6.0."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ns = runpy.run_path(str(Path(__file__).with_name("resolve-native.py")))
resolve = ns["resolve"]


def version(ours: str, theirs: str) -> str:
    if "0.5.9" not in ours or "0.2.14" not in theirs:
        raise ValueError("Version hunk changed")
    return ours.replace("0.5.9", "0.6.0")


def install_download(ours: str, theirs: str) -> str:
    # New bounded transfer; old .NET hashing keeps PowerShell 5.1 independent
    # of an inherited PowerShell 7-only PSModulePath.
    transfer = theirs.split("    $got = (Get-FileHash", 1)[0]
    hashing = ours[ours.index("    # Use the runtime directly:"):]
    return transfer + hashing


def main() -> None:
    rows = []
    for path, count in (
        ("apps/host/package.json", 1), ("apps/host/package-lock.json", 2),
        ("apps/desktop/package.json", 1),
        ("apps/desktop/src-tauri/Cargo.toml", 1),
        ("apps/desktop/src-tauri/Cargo.lock", 1),
    ):
        rows.append(resolve(path, [version] * count))
    fork_url = lambda a, b: b.replace("Niko1221/Strata", "zapabob/Strata")
    rows.append(resolve("packaging/install.ps1", [fork_url, install_download]))
    rows.append(resolve("packaging/install.sh", [fork_url]))
    old = json.loads(subprocess.check_output([
        "git", "-C", str(ROOT), "show", "6f13c755:docs/data/release.json",
    ]))
    old["historical_releases"].append({
        "tag": old["tag"], "product_version": old["product_version"],
        "assets": old["assets"],
    })
    old.update(tag="v0.6.0", product_version="0.6.0", control_version="0.6.0",
               engine_version="0.6.0", assets=[
                   "Strata-0.6.0-windows-x64-setup.exe",
                   "Strata-0.6.0-windows-x64.tar.7z",
               ])
    upstream = json.loads(Path(__file__).with_name("upstream-release.json").read_text())
    old["upstream"] = dict(upstream, published="2026-11-16")
    (ROOT / "docs/data/release.json").write_text(json.dumps(old, indent=2) + "\n")
    receipt = Path("H:/Strata/evidence/upstream-sync-20261116/metadata-resolution.json")
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)


if __name__ == "__main__":
    main()
