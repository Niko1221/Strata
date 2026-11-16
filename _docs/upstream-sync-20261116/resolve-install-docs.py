"""Resolve installation documentation boundaries without relabelling old evidence."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('H:/Strata/evidence/upstream-sync-20261116/install-docs-resolution.json')
INTRO = {
    'docs/INSTALL.md': '''For this checkout's Hakua Windows package, follow [Install Hakua / Strata](HAKUA_INSTALL.md)
and use the zapabob fork release. The 0.6.0 source is an integration candidate;
consult the delivery record before treating it as a published or verified installer.
The platform recipes below describe the upstream Niko1221 distribution and its
separate packages. They do not install the Hakua fork.

This upstream reference is for anyone installing or updating Strata (version 0.2.14). Use the **desktop app** for a graphical
''',
    'packaging/README.md': '''This page is for developers who build the desktop app, Terminal and engine packages.
Upstream reference: app and Terminal 0.2.14, Control and engines 0.1.60.
The Hakua fork uses independent versions; its current source candidate is 0.6.0.
To install the fork on Windows, use [docs/HAKUA_INSTALL.md](../docs/HAKUA_INSTALL.md).
The cross-platform recipes below describe upstream builds, not measured Hakua
inference on every platform.

The Windows fork keeps the upstream per-user Tauri NSIS layout, private Node and
Python, and independently signed engine/Control packages. Its tar.7z contains
the identical installer plus components, instructions and checksums. Model
weights and personal data are excluded. Component/feed signatures are separate
from Windows Authenticode; the installer has no Windows code signature.
Actual build, installation, publication and qualification evidence belongs in
[HAKUA_RELEASE.md](../docs/HAKUA_RELEASE.md); old measurements remain historical.

Before staging the Windows app, run
`node apps/desktop/scripts/stage-rustxng.mjs --execute` from the repository root.
It verifies the pinned RustXNG Full binary, attribution and corresponding source.
The files remain under installer-managed `harness/rustxng`, with normal NSIS
upgrade cleanup. Never substitute an old signed component for a newly labelled
source build. Preserve the 3 GiB RAM, 4 GiB commit and disk floors and use at
most five compiler jobs for this PC's candidate builds.

''',
}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = []
    for relative, intro in INTRO.items():
        path = ROOT / relative
        before = path.read_bytes()
        content, count = re.subn(r'^<<<<<<< HEAD\n.*?^=======\n.*?^>>>>>>> [a-f0-9]+\n',
                                 lambda _: intro, before.decode().replace('\r\n', '\n'),
                                 flags=re.M | re.S)
        assert count == 1 and not re.search(r'^(<<<<<<<|=======|>>>>>>>)', content, re.M)
        path.write_text(content, encoding='utf-8', newline='\n')
        rows.append({'path': relative, 'beforeSHA256': hashlib.sha256(before).hexdigest(),
                     'afterSHA256': hashlib.sha256(path.read_bytes()).hexdigest()})
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump({'rows': rows, 'scope': 'fork installation versus official reference'}, stream, indent=2)


if __name__ == '__main__':
    main()
