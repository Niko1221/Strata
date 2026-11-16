"""Bounded static or native CPU checks for the fixed official intake."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = Path('H:/Strata/private-backups/cleanup-20261116/upstream-c775-intake-v1')
BASE = Path('H:/Strata/evidence/upstream-sync-20261116')


def run(argv, timeout=180):
    print(json.dumps({'command': list(map(str, argv))}), flush=True)
    subprocess.run(list(map(str, argv)), cwd=ROOT, check=True, timeout=timeout)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('mode', choices=['static', 'native-cpu'])
args = parser.parse_args()
if args.mode == 'static':
    paths = set(json.loads((PRIVATE/'preview.json').read_text())['rows'])
    paths.update(subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT, text=True).splitlines())
    paths.update(['_docs/upstream-sync-20261116/'+n for n in ('intake-c775.py', 'resolve-c775.py', 'test_c775_guard.py', 'check-c775.py', 'freeze-c775.py')])
    paths.add('apps/host/tests/helpers/verified-local-runtime.mjs')
    python = sorted(n for n in paths if n.endswith('.py'))
    modules = sorted(n for n in paths if n.endswith('.mjs'))
    for name in python:
        ast.parse((ROOT/name).read_text(encoding='utf-8-sig'), filename=name)
    for name in sorted(n for n in paths if n.endswith('.json')):
        json.loads((ROOT/name).read_text(encoding='utf-8-sig'))
    run([sys.executable, '-m', 'flake8', '--isolated', '--select=E9,F63,F7,F82', *python])
    for name in modules:
        run([ROOT/'apps/desktop/src-tauri/resources/node/node.exe', '--check', name])
    run(['C:/Users/downl/go/bin/actionlint.exe', '-shellcheck=', '.github/workflows/snapshot.yml'])
    run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--check'])
    run([ROOT/'apps/desktop/src-tauri/resources/node/node.exe', 'apps/host/node_modules/typescript/bin/tsc', '--noEmit', '-p', 'apps/host/tsconfig.json'])
    print(json.dumps({'pythonAST': len(python), 'nodeSyntax': len(modules), 'actionlint': 1, 'typescript': 'pass'}))
else:
    out = BASE/'c775-native-cpu-v1'
    out.mkdir(exist_ok=False)
    compiler = Path('C:/Program Files/LLVM/bin/clang++.exe')
    run([compiler, '--version'])
    for name in ('conversation_segment_pool_test', 'conversation_buffer_pool_test'):
        exe = out/(name+'.exe')
        run([compiler, '-std=c++17', '-O2', '-I', ROOT/'include', ROOT/('src/core/'+name+'.cpp'), '-o', exe])
        run([exe], timeout=180)
    os.environ['PATH'] = str(compiler.parent)+';'+os.environ['PATH']
    run([sys.executable, '-m', 'pytest', 'tools/test_dit_rope.py', 'tools/test_te_tokenizer.py', 'tools/test_qimage.py',
         '-q', '--tb=short', '-p', 'no:cacheprovider', '--junitxml='+str(BASE/'c775-native-cpu-v1.xml')], timeout=300)
