"""Typecheck or build current Host using the checksum-verified private Node."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['types', 'build'])
    args = parser.parse_args()
    env = dict(os.environ)
    env.update(STRATA_PUBLIC_PACKAGE='1', STRATA_PUBLIC_ASSETS='H:/Strata/releases/hakua033-public-assets',
               STRATA_PUBLIC_ASSETS_RECEIPT='H:/Strata/releases/hakua033-public-assets.receipt.json',
               NODE_OPTIONS='--max-old-space-size=1024')
    command = ([str(ROOT / 'apps/desktop/src-tauri/resources/node/node.exe')]
               + (['node_modules/typescript/bin/tsc', '--noEmit'] if args.action == 'types' else ['scripts/build.mjs']))
    subprocess.run(command, cwd=ROOT / 'apps/host', env=env, check=True, timeout=240)


if __name__ == '__main__':
    main()
