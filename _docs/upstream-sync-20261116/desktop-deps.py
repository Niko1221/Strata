"""Stage the exact locked desktop CLI in an isolated dependency directory."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('H:/Strata/evidence/upstream-sync-20261116/desktop-deps-060')
NODE = ROOT / 'apps/desktop/src-tauri/resources/node/node.exe'

def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--link-existing', action='store_true')
args = parser.parse_args()
if args.link_existing:
    assert OUT.is_dir() and not (OUT / 'receipt.json').exists()
    previous = json.loads((OUT.parent / 'receipts/desktop-cli-deps-v1.json').read_text())
    assert previous['run']['status'] == 'FINISHED' and previous['run']['job_active_after'] == 0
else:
    assert not OUT.exists()
    OUT.mkdir()
inputs = {}
for name in ('package.json', 'package-lock.json'):
    source = ROOT / 'apps/desktop' / name
    if not args.link_existing:
        shutil.copy2(source, OUT / name)
    assert sha(source) == sha(OUT / name)
    inputs[name] = sha(source)
if not args.link_existing:
    subprocess.run([str(NODE), 'C:/Program Files/nodejs/node_modules/npm/bin/npm-cli.js',
                    'ci', '--ignore-scripts', '--no-audit', '--no-fund'], cwd=OUT, check=True, timeout=300)
actual = subprocess.check_output([str(NODE), str(OUT / 'node_modules/@tauri-apps/cli/tauri.js'),
                                  '--version'], cwd=OUT, text=True, timeout=30).strip()
assert actual == 'tauri-cli 2.12.3'
target = ROOT / 'apps/desktop/node_modules'
assert not target.exists() and not target.is_symlink() and not target.is_junction()
subprocess.run([str(NODE), '-e',
                "require('node:fs').symlinkSync(process.argv[1],process.argv[2],'junction')",
                str(OUT / 'node_modules'), str(target)], check=True, timeout=30)
assert target.is_junction() and target.resolve() == (OUT / 'node_modules').resolve()
with (OUT / 'receipt.json').open('x', encoding='utf-8') as stream:
    json.dump({'ok': True, 'sourceInputs': inputs, 'actualCLI': actual,
               'primaryDependenciesChanged': False, 'installScriptsRun': False}, stream, indent=2)
print(json.dumps({'ok': True, 'actualCLI': actual, 'primaryDependenciesChanged': False}))
