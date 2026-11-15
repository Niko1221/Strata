"""Resolve only identified patched crate releases; retain the original lock."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('H:/Strata/evidence/upstream-sync-20261116/cargo-security-update-v1')
CARGO = Path('C:/Users/downl/.cargo/bin/cargo.exe')


def main():
    OUT.mkdir()
    lock = ROOT / 'apps/desktop/src-tauri/Cargo.lock'
    (OUT / 'Cargo.before.lock').write_bytes(lock.read_bytes())
    updates = [('tauri-plugin-updater', '2.14.1'), ('rustix', '1.1.6'),
               ('rustls-webpki', '0.103.16'), ('webpki-root-certs', '1.0.10')]
    for name, version in updates:
        subprocess.run([str(CARGO), 'update', '--manifest-path',
                        str(ROOT / 'apps/desktop/src-tauri/Cargo.toml'),
                        '--package', name, '--precise', version],
                       cwd=ROOT, check=True, timeout=150)
    (OUT / 'Cargo.after.lock').write_bytes(lock.read_bytes())
    (OUT / 'receipt.json').write_text(json.dumps({
        'beforeSHA256': hashlib.sha256((OUT / 'Cargo.before.lock').read_bytes()).hexdigest(),
        'afterSHA256': hashlib.sha256(lock.read_bytes()).hexdigest(), 'updates': updates,
        'scope': 'Locked targeted security releases; still requires re-audit and build.'}, indent=2))


if __name__ == '__main__':
    main()
