"""Pack and verify the newly signed coordinated 0.5.8 components and feed."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('H:/Strata/releases/stable-058-20261111-codex')
RES = ROOT / 'apps/desktop/src-tauri/resources'
PRODUCT = '0.5.8'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def run(*args):
    subprocess.run([sys.executable, '-I', '-B', str(ROOT / 'packaging/feed_tool.py'), *map(str, args)],
        cwd=ROOT, check=True, timeout=600, stdout=subprocess.DEVNULL)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--key', type=Path, required=True)
parser.add_argument('--execute', action='store_true')
args = parser.parse_args()
if not args.execute:
    print(json.dumps({'execute': False, 'output': str(OUT / 'component-assets'), 'version': PRODUCT}))
    raise SystemExit(0)
assert args.key.is_absolute() and not args.key.is_relative_to(ROOT) and not args.key.is_relative_to(OUT)
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)
native = load(OUT / 'app-058-components/staged-native.json')
control = load(OUT / 'app-058-control-r2/control-build.json')
frozen = load(OUT / 'frozen-source.json')
subprocess.run([str(RES / 'node/node.exe'), str(Path(__file__).with_name('source-binding.mjs')), frozen['sourceRoot'], frozen['sourceCommit']], check=True, timeout=90)
assert native['ok'] and native['signatureVerified'] and native['nativeSourceCommit'] == frozen['sourceCommit']
assert control['ok'] and control['controlVersion'] == PRODUCT
assert (RES / 'control/COMMIT').read_text().strip() == control['sourceCommit']
subprocess.run([str(RES / 'node/node.exe'), str(Path(__file__).with_name('control-binding.mjs'))], check=True, timeout=90)
assert (RES / 'control/VERSION').read_text().strip() == PRODUCT
signed = RES / f'engine-bundle/{PRODUCT}-windows-x86_64-cuda-cuda13'
assert sha(signed / 'engine-component.json') == native['signedManifestSHA256']
assets = OUT / 'component-assets'
assets.mkdir()
run('pack-engine', signed, '--out', assets, '--format', 'tar.gz', '--url-prefix=')
run('pack-control', RES / 'control', '--out', assets, '--version', PRODUCT, '--platform', 'windows-x86_64', '--format', 'zip', '--url-prefix=')
previous = load(Path('H:/Strata/releases/public-055-20261111-opencode/assets/strata-feed.json'))
inputs = OUT / 'component-feed-inputs'
for family in ('engine', 'control'):
    target = inputs / family
    target.mkdir(parents=True)
    entries = list(assets.glob(f'strata-{family}-*.entry.json'))
    assert len(entries) == 1
    shutil.copy2(entries[0], target / entries[0].name)
run('build-feed', '--out', assets, '--key', args.key, '--sequence', previous['sequence'] + 1,
    '--channel', 'stable', '--engine', inputs / 'engine', '--control', inputs / 'control')
for name in ('install.ps1', 'install.sh'):
    shutil.copy2(ROOT / 'packaging' / name, assets / name)
code = "import sys,json;sys.path.insert(0,sys.argv[1]);from terminal.strata_control import feed,signing;f=feed.fetch(sys.argv[2],signing.load_trusted_keys(environ={}));assert len(f.engine)==len(f.control)==1;assert f.engine[0].version==f.control[0].version=='0.5.8';print(json.dumps({'ok':True,'version':'0.5.8','sequence':f.sequence}))"
verified = subprocess.check_output([str(RES / 'control/python/python.exe'), '-I', '-B', '-c', code,
    str(RES / 'control'), str(assets)], cwd=ROOT, text=True, timeout=60)
assert json.loads(verified)['ok']
records = {path.name: {'bytes': path.stat().st_size, 'sha256': sha(path)} for path in assets.iterdir() if path.is_file()}
result = {'ok': True, 'sourceCommit': commit, 'coordinatedSemVer': PRODUCT,
    'signatureVerifiedWithBundledPublicKeys': True, 'feed': json.loads(verified), 'assets': records,
    'native': native, 'control': {'sourceCommit': control['sourceCommit'], 'version': control['controlVersion'], 'archiveSHA256': control['archiveSHA256']}}
with (OUT / 'components-verified.json').open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps({'ok': True, 'version': PRODUCT, 'assets': len(records)}))
