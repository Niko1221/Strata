"""Download the pinned browser engine from npm, verifying its registry integrity.

Run with katex or mathjax. End users never run this or contact npm.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {"katex": "0.19.0", "mathjax": "4.1.3", "@mathjax/mathjax-newcm-font": "4.1.3"}


def vendor(name):
    version = VERSIONS[name]
    with urlopen(f"https://registry.npmjs.org/{name}/{version}") as response:
        meta = json.load(response)
    with urlopen(meta["dist"]["tarball"]) as response:
        payload = response.read()
    integrity = "sha512-" + base64.b64encode(hashlib.sha512(payload).digest()).decode()
    if integrity != meta["dist"]["integrity"]:
        raise ValueError("npm package integrity mismatch")
    dest = ROOT / "serve/web/vendor" / name.replace("@mathjax/", "")
    dest.mkdir(parents=True, exist_ok=True)
    retained = set()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            path = member.name.removeprefix("package/")
            keep = path.lower().startswith("license")
            if name == "katex":
                keep |= path in ("dist/katex.min.js", "dist/katex.min.css") or path.startswith("dist/fonts/") and path.endswith(".woff2")
            elif name == "mathjax":
                keep |= path in ("tex-chtml.js", "ui/safe.js", "a11y/assistive-mml.js")
            else:
                keep |= path.startswith(("chtml/", "woff2/")) and path.endswith((".js", ".woff2"))
            target = (dest / path).resolve()
            if not keep or not member.isfile() or not target.is_relative_to(dest.resolve()):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
            retained.add(target)
    # Remove only stale generated package files, after verifying the new package.
    for old in dest.rglob("*"):
        resolved = old.resolve()
        if not resolved.is_relative_to(dest.resolve()):
            raise ValueError("vendor path leaves package directory")
        if old.is_file() and resolved not in retained:
            old.unlink()
    if name == "@mathjax/mathjax-newcm-font":
        # This npm archive omits the license text; its metadata declares Apache-2.0.
        if meta.get("license") != "Apache-2.0":
            raise ValueError("font package license changed")
        (dest / "LICENSE").write_bytes((ROOT / "serve/web/vendor/mathjax/LICENSE").read_bytes())
    (dest / "PROVENANCE.json").write_text(json.dumps({"package": name, "version": version,
        "url": meta["dist"]["tarball"], "integrity": integrity, "license": meta.get("license")}, indent=2) + "\n")
    print(name, version, sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()), "bytes")


if __name__ == "__main__":
    engine = sys.argv[1]
    vendor(engine)
    if engine == "mathjax":
        vendor("@mathjax/mathjax-newcm-font")
