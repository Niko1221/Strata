"""Stage only this candidate's resources, comparing every copied byte."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('H:/Strata/releases/local-052-20261106-opencode')
FROZEN = Path('H:/Strata/worktrees/upstream-0211-latest-20261105')
SIGNED = Path('H:/Strata/releases/strata-0.5.1/component-r1/0.5.1-windows-x86_64-cuda-cuda13')
NATIVE_SHA = 'e75716a9d5deb6289e87741d42e7003ba82b7696f6121d5203e615d9fe596a96'

def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def inventory(root: Path) -> dict[str, str]:
    paths = sorted(root.rglob('*'))
    assert all(not p.is_symlink() and not p.is_junction() for p in paths)
    return {p.relative_to(root).as_posix(): sha(p) for p in paths if p.is_file()}

def main() -> None:
    assert OUT.is_dir() and not (OUT/'resources.json').exists()
    assert subprocess.check_output(['git','-C',str(FROZEN),'rev-parse','HEAD'],text=True).strip() == 'c50ca06a863a98ff121cbff30365a035a90205e2'
    sys.path.insert(0, str(ROOT))
    from terminal.strata_control import signing, engine_component
    manifest = json.loads((SIGNED/'engine-component.json').read_text())
    signing.verify_bytes(signing.load_trusted_keys(environ={}), engine_component.signing_payload(manifest), manifest['signature'])
    assert sha(SIGNED/'strata.exe') == NATIVE_SHA
    report: dict = {'ok':False, 'trees':{}, 'nativeSourceCommit':'c50ca06a863a98ff121cbff30365a035a90205e2', 'nativeSHA256':NATIVE_SHA, 'nativeManifestSHA256':sha(SIGNED/'engine-component.json'), 'signatureVerified':True}
    resources = ROOT/'apps/desktop/src-tauri/resources'
    old = FROZEN/'apps/desktop/src-tauri/resources'
    for name in ['node','harness','notices','local-backends']:
        source, destination = old/name, resources/name
        expected = inventory(source)
        if destination.exists():
            actual = inventory(destination)
            assert all(n == '.gitignore' or expected.get(n) == digest for n,digest in actual.items()), 'Unfamiliar resource staging'
        shutil.copytree(source, destination, dirs_exist_ok=True)
        assert inventory(destination) == expected
        report['trees'][name] = {'files':len(expected), 'inventorySHA256':hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()}
    native = resources/'engine-bundle'/SIGNED.name
    assert not native.exists()
    expected = inventory(SIGNED)
    shutil.copytree(SIGNED,native)
    assert inventory(native) == expected
    report['trees']['signed-native'] = {'files':len(expected),'inventorySHA256':hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()}
    installed = Path('C:/Users/downl/AppData/Local/Strata/components/engine')/SIGNED.name
    assert inventory(installed) == expected, 'Installed component must be identical to retained signed component'
    for name in ['icon.ico','icon.png']:
        source = Path('H:/Strata/releases/hakua033-public-assets')/name
        shutil.copy2(source,ROOT/'apps/desktop/src-tauri/icons'/name)
        assert sha(source) == sha(ROOT/'apps/desktop/src-tauri/icons'/name)
    report['ok'] = True
    (OUT/'resources.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__ == '__main__':
    main()
