"""Stage the public desktop shell without embedding supplied UI ZIP assets.

Build the resulting directory with Cargo --release --locked --features
public-full-package in a configured Windows MSVC environment. Runtime resources
are enumerated separately by full_windows.py, never embedded by this shell.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from native_preview import no_reparse

MINIMAL_HTML = '<!doctype html><html><head><title>Strata</title></head><body>Starting Strata.</body></html>\n'


def stage(source: Path, assets: Path, output: Path) -> dict:
    for path in [source, assets, output]:
        no_reparse(path)
    if output.exists() or not output.parent.is_dir():
        raise ValueError('shell stage must be fresh under an existing parent')
    files = [source/name for name in ['Cargo.toml','Cargo.lock','build.rs','tauri.conf.json']]
    files += sorted(p for p in (source/'src').rglob('*') if p.is_file())
    for path in files:
        no_reparse(path)
    for name in ['icon.png','icon.ico']:
        no_reparse(assets/name)
        if not (assets/name).is_file():
            raise ValueError('owned shell icon missing')
    config = json.loads((source/'tauri.conf.json').read_text(encoding='utf-8'))
    config['build']['frontendDist'] = 'minimal-ui'
    config['bundle']['resources'] = {}
    config['bundle']['windows']['nsis'] = {'installMode':'currentUser'}
    config['plugins'] = {}
    output.mkdir()
    for path in files:
        target = output/path.relative_to(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    (output/'minimal-ui').mkdir()
    (output/'minimal-ui/index.html').write_text(MINIMAL_HTML,encoding='utf-8')
    (output/'icons').mkdir()
    for name in ['icon.png','icon.ico']:
        shutil.copy2(assets/name,output/'icons'/name)
    (output/'tauri.conf.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    return {p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob('*')) if p.is_file()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['source','assets','output']:
        parser.add_argument('--'+flag,type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.source,args.assets,args.output),indent=2))
