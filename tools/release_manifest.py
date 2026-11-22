#!/usr/bin/env python3
"""Independent fork release and upstream documentation reference: docs/data/release.json.

    python tools/release_manifest.py --check     fail when a current page or a version source disagrees
    python tools/release_manifest.py --write     update owned upstream-reference version lines

What it covers (standard library only, no network):
  * the manifest against the code: app/Terminal version (terminal/strata_cli/__init__.py, app/host and app/desktop package
    files, Cargo.toml) and Control/engine version (the CMake project version);
  * the three installer file names in the manifest's asset list;
  * fork product versions against the app/Terminal sources, Control against
    CONTROL_VERSION and the engine against the CMake project version;
  * the declared platform installers (all three for an upstream-only manifest,
    Windows only for the Hakua fork);
  * every installer file name (Strata-<version>-windows-x64-setup.exe, ...-linux-x86_64.AppImage, ...-macos-arm64.dmg)
    in a current page: the version must be the manifest's, and the file must be in the release's asset list;
  * the version lines of the pages listed in LINES below.
Release notes, measurement logs and docs/history/ keep the versions they were written with; tools/doc_pages.py says which
pages are archived.  To bump a release: change the version sources, edit docs/data/release.json, run --write, run --check.
    in a current page: each must be in the current, upstream or explicitly
    declared historical release's asset list. Immutable URL tags must agree;
  * upstream-reference version lines listed in LINES below, independently of
    fork product and component versions.
Release notes, measurement logs and history/ keep the versions they were written with; tools/doc_pages.py says which
pages are archived. Review new installer links explicitly: --write preserves
every immutable URL and historical installer name. Run --check after a bump.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doc_pages  # noqa: E402

ROOT = doc_pages.ROOT
MANIFEST = os.path.join(ROOT, "docs", "data", "release.json")

INSTALLER_NAMES = ("Strata-{v}-windows-x64-setup.exe", "Strata-{v}-linux-x86_64.AppImage", "Strata-{v}-macos-arm64.dmg")
# Any installer-looking file name in a current page.  Group 1 is the version.
INSTALLER_RE = re.compile(r"Strata-(\d+\.\d+(?:\.\d+)*)-(?:windows-x64-setup\.exe|linux-x86_64\.AppImage|macos-arm64\.dmg)")

V = r"(\d+\.\d+(?:\.\d+)*)"
# (page, regex, manifest keys): the regex has one group per key, each holding a version.  A rule that matches nothing is
# an error, so rewording a line cannot quietly switch its check off.
LINES = [
    ("docs/README.md", r"This documentation is for version " + V + r"\.", ["product_version"]),
    ("docs/INSTALL.md", r"\(version " + V + r"\)", ["product_version"]),
    ("docs/AI_SETUP.md", r"install Strata \(version " + V + r"\)", ["product_version"]),
    ("docs/SOURCE_INSTALL.md", r"use Strata \(version " + V + r"\)", ["product_version"]),
    # Upstream v0.2.19 reworded the AGENTS.md version line; the version and the
    # three installer asset names are now on separate lines.
    ("AGENTS.md", r"Version " + V + r", engine " + V + r"\.", ["product_version", "engine_version"]),
    # Upstream installer names in AGENTS.md are validated by INSTALLER_RE
    # against the declared upstream.assets list (see the check loop above).
    # A version-line rule here would also match this fork's own v0.5.9 Hakua
    # installer line and falsely report it as an upstream version mismatch.
    ("docs/AI_REFERENCE.md", r"against the " + V + r" release tree \(app and Terminal " + V + r"; Control " + V + r", engines " + V + r"\)",
     ["product_version", "product_version", "control_version", "engine_version"]),
    ("tools/packaging/UPDATES.md", r"one per OS \(" + V + r"\)", ["product_version"]),
    ("tools/packaging/UPDATES.md", r'"app": \{[^\n]*\n\s*"version": "' + V + '"', ["product_version"]),
    ("tools/packaging/UPDATES.md", r"--target [a-z0-9_-]+ +--version " + V, ["product_version"]),
    ("tools/packaging/UPDATES.md", r'"control": \{"version": "' + V + '"', ["control_version"]),
    ("tools/packaging/UPDATES.md", r"pack-control <control folder> --version " + V, ["control_version"]),
    ("tools/packaging/UPDATES.md", r"VERSION file\s*\n\s*# \(" + V + r"\)", ["control_version"]),
    ("tools/packaging/UPDATES.md", r'"version": "' + V + r'", "platform": "linux-x86_64", "backend": "cuda"', ["engine_version"]),
    ("tools/packaging/UPDATES.md", r"engine/strata-engine-" + V + r"-linux-x86_64-cuda-cuda13", ["engine_version"]),
    ("docs/MEDIA_KIT.md", r"for Strata " + V + r"\.", ["product_version"]),
    ("tools/packaging/README.md", r"\(app and Terminal " + V + r", Control " + V + r", engines " + V + r"\)", ["product_version", "control_version", "engine_version"]),
    ("tools/packaging/ENGINE-COMPONENT.md", r'"name": "strata-engine", "version": "' + V + '"', ["engine_version"]),
    ("tools/packaging/ENGINE-COMPONENT.md", r"package_engine.py --src <build folder> --backend cuda --version " + V, ["engine_version"]),
]


def load():
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def read(rel_path):
    with open(os.path.join(ROOT, rel_path), encoding="utf-8", newline="") as f:
        return f.read()


def first(rel_path, pattern):
    m = re.search(pattern, read(rel_path), re.M)
    return m.group(1) if m else None


def code_versions():
    """Where the versions come from in the source tree: name -> (value found, manifest key)."""
    cmake = first("engine/CMakeLists.txt", r"^project\(strata VERSION ([0-9.]+) ")
    return {
        # the Terminal says the product version, or (a Control-only hotfix such as 0.2.13.1) the manifest's optional terminal_version
        "terminal/strata_cli/__init__.py": (first("terminal/strata_cli/__init__.py", r'^PRODUCT_VERSION = "([^"]+)"'), "terminal_version"),
        "app/host/package.json": (json.loads(read("app/host/package.json")).get("version"), "product_version"),
        "app/desktop/package.json": (json.loads(read("app/desktop/package.json")).get("version"), "product_version"),
        "app/desktop/src-tauri/Cargo.toml": (first("app/desktop/src-tauri/Cargo.toml", r'^version = "([^"]+)"'), "product_version"),
        "CMakeLists.txt (engine)": (cmake, "engine_version"),
        "CONTROL_VERSION": (read("CONTROL_VERSION").strip(), "control_version"),
    }


def expand(manifest):
    return [n.format(v=manifest["product_version"]) for n in manifest.get("installer_names", INSTALLER_NAMES)]


def rewrite(text, regex, keys, manifest):
    """Replace each version group of every match of `regex` with the manifest's value."""
    def sub(m):
        out, last = [], m.start()
        for i, key in enumerate(keys, 1):
            out.append(text[last:m.start(i)])
            out.append(manifest[key])
            last = m.end(i)
        out.append(text[last:m.end()])
        return "".join(out)

    return re.sub(regex, sub, text, flags=re.M)


def write(manifest):
    changed = []
    for rel_path in doc_pages.current_pages():
        text = read(rel_path)
        # Immutable release URLs and historical installer references must never
        # be upgraded by an unscoped text replacement. Only owned version lines
        # are generated here; new installer links are reviewed with the release.
        new = text
        for page, regex, keys in LINES:
            if page == rel_path:
                new = rewrite(new, regex, keys, manifest.get("upstream", manifest))
        if new != text:
            with open(os.path.join(ROOT, rel_path), "w", encoding="utf-8", newline="") as f:
                f.write(new)
            changed.append(rel_path)
    return changed


def _vkey(version):
    return tuple(int(p) for p in str(version).split("."))


def check(manifest):
    problems = []
    for key in ("tag", "product_version", "control_version", "engine_version", "assets"):
        if key not in manifest:
            problems.append("docs/data/release.json has no %r" % key)
    if problems:
        return problems
    if manifest["tag"] != "v" + manifest["product_version"]:
        problems.append("tag %s does not match product_version %s" % (manifest["tag"], manifest["product_version"]))
    for where, (found, key) in code_versions().items():
        want = manifest.get(key, manifest["product_version"]) if key == "terminal_version" else manifest[key]
        if where == "CMakeLists.txt (engine)" and "upstream" in manifest:
            # The Hakua fork keeps its own engine component version (engine_version,
            # e.g. 0.6.3) while engine/CMakeLists.txt follows the absorbed upstream
            # engine (upstream.engine_version, e.g. 0.1.63). Compare the CMake project
            # version against the upstream reference, not the fork component version.
            up_want = manifest["upstream"]["engine_version"]
            if found != up_want and not found.startswith(up_want + "."):
                problems.append("%s says %s, upstream reference engine_version is %s" % (where, found, up_want))
        elif where == "CMakeLists.txt (engine)" and "upstream" not in manifest:
            # the CMake project version is Control's. The engine packages carry it, except after a Control-only hotfix (Control 0.1.59.1,
            # engines still 0.1.59): then Control is the engine version plus one hotfix part.
            # A release that builds no engine (0.2.15: Control 0.1.61, the 0.1.60 engines of v0.2.14 carried by absolute url) names the release
            # it takes them from in "engines_from"; then Control may be ahead of the engines, and the engine packages at engine_version must be
            # in the asset list (the packages the feed offers).
            carried = bool(manifest.get("engines_from")) and manifest["engines_from"] != manifest["tag"] \
                and _vkey(manifest["control_version"]) > _vkey(want) \
                and any(a.startswith("strata-engine-%s-" % want) for a in manifest["assets"])
            if found != manifest["control_version"] or not (found == want or found.startswith(want + ".") or carried):
                problems.append("%s says %s, docs/data/release.json %s is %s (control_version %s)" % (where, found, key, want, manifest["control_version"]))
        elif found != want:
            problems.append("%s says %s, docs/data/release.json %s is %s" % (where, found, key, want))
    assets = set(manifest["assets"])
    for name in expand(manifest):
        if name not in assets:
            problems.append("docs/data/release.json: the asset list has no %s" % name)
    reference = manifest.get("upstream", manifest)
    allowed_assets = assets | set(reference["assets"])
    for historical in manifest.get("historical_releases", []):
        if historical.get("tag") != "v" + historical.get("product_version", ""):
            problems.append("historical release tag/version mismatch")
        for asset in historical.get("assets", []):
            match = INSTALLER_RE.fullmatch(asset)
            if not match or match.group(1) != historical.get("product_version"):
                problems.append("historical release has invalid installer: " + asset)
            allowed_assets.add(asset)
    if reference["tag"] != "v" + reference["product_version"]:
        problems.append("upstream reference tag/version mismatch")
    for rel_path in doc_pages.current_pages():
        text = read(rel_path)
        for m in INSTALLER_RE.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            tail = re.match(r"[A-Za-z0-9_.-]*", text[m.start():]).group(0).rstrip(".")
            if tail not in allowed_assets:
                problems.append("%s:%d names %s, which is not in a declared release's asset list" % (rel_path, line, tail))
        for link in re.finditer(r"https://github\.com/[^\s/]+/Strata/releases/download/v([0-9.]+)/(?P<asset>Strata-[^\s)`]+)", text):
            installer = INSTALLER_RE.match(link.group('asset'))
            if installer and installer.group(1) != link.group(1):
                problems.append("%s: installer version differs from its immutable download tag" % rel_path)
    for page, regex, keys in LINES:
        text = read(page)
        matches = list(re.finditer(regex, text, re.M))
        if not matches:
            problems.append("%s: the version-line rule /%s/ matches nothing (line reworded? update LINES in tools/release_manifest.py)" % (page, regex[:50]))
        for m in matches:
            line = text.count("\n", 0, m.start()) + 1
            for i, key in enumerate(keys, 1):
                if m.group(i) != reference[key]:
                    problems.append("%s:%d says %s, the upstream reference's %s is %s" % (page, line, m.group(i), key, reference[key]))
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    manifest = load()
    if args.write:
        changed = write(manifest)
        print("updated %d page(s)%s" % (len(changed), "".join("\n  " + c for c in changed)))
    problems = check(manifest)
    for p in problems:
        print("FAIL " + p)
    print("release manifest: %s" % ("%d problem(s)" % len(problems) if problems else "ok (%s, engine %s)" % (manifest["product_version"], manifest["engine_version"])))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
