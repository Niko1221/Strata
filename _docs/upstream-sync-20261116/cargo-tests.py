"""Compile/test current desktop Rust with its locked, audited Windows dependencies."""
from pathlib import Path
import argparse
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('H:/Strata/evidence/upstream-sync-20261116')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit-only', action='store_true')
    args = parser.parse_args()
    env = dict(os.environ)
    env.update(CARGO_TARGET_DIR=str(BASE / 'cargo-target'), CARGO_BUILD_JOBS='5',
               RUSTC_WRAPPER='', RUSTC_WORKSPACE_WRAPPER='')
    if args.unit_only:
        # Unit compilation does not require a staged distribution. The real
        # NSIS build must use unmodified bundle resources and verify the payload.
        env['TAURI_CONFIG'] = json.dumps({'bundle': {'resources': []}})
        print('UNIT-ONLY resource configuration; packaging validation NOT_RUN', flush=True)
    subprocess.run(['C:/Users/downl/.cargo/bin/cargo.exe', 'test', '--manifest-path',
                    str(ROOT / 'apps/desktop/src-tauri/Cargo.toml'), '--locked', '--offline',
                    '--', '--test-threads=1'], cwd=ROOT, env=env, check=True, timeout=1700)


if __name__ == '__main__':
    main()
