"""Build against the inspected, pinned llama.cpp tree without modifying it."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PIN = '972d2313bc0bf0a45f634f77d95c9fb03aeab12c'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--llama', type=Path, required=True)
    args = p.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.llama), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != PIN or subprocess.check_output(['git', '-C', str(args.llama), 'status', '--porcelain']):
        raise SystemExit('Backend source must be clean and match the inspected pin')
    out = ROOT/'build'
    out.mkdir(exist_ok=True)
    lib = args.llama/'build/bin'
    common = ['c++', '-std=c++17', '-O2', '-I'+str(args.llama/'include'),
              '-I'+str(args.llama/'ggml/include'), '-L'+str(lib), '-Wl,-rpath,'+str(lib)]
    commands = [common+[str(ROOT/'native/fresh_oracle.cpp'), '-o', str(out/'fresh-oracle'), '-lllama', '-lggml', '-lggml-base'],
                common+['-shared', '-fPIC', str(ROOT/'native/llama_bridge.cpp'), '-o', str(out/'libctt.so'), '-lllama', '-lggml', '-lggml-base']]
    for command in commands:
        subprocess.run(command, check=True)
    hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in [out/'fresh-oracle', out/'libctt.so', lib/'libllama.so', args.llama/'include/llama.h']}
    (out/'build.json').write_text(json.dumps(dict(backend_revision=revision, commands=commands, sha256=hashes), indent=2)+'\n')
    print(out/'build.json')


if __name__ == '__main__':
    main()
