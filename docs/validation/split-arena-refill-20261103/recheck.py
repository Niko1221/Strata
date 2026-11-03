"""Recheck archived accounting and regenerate both benchmark summaries, without a GPU."""
import hashlib
import json
import pathlib
import subprocess
import sys
import tarfile
import tempfile

here = pathlib.Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='strata-refill-evidence-') as directory:
    root = pathlib.Path(directory)
    with tarfile.open(here / 'evidence.tar.gz') as archive:
        archive.extractall(root, filter='data')
    manifest = json.loads((root / 'SHA256.json').read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    print(f'Archive integrity: {len(manifest)} files verified', flush=True)
    for campaign in ('short', 'long'):
        subprocess.run([sys.executable, str(root / campaign / 'analyze.py')], check=True)
    long = json.loads((root / 'long/summary.json').read_text())
    assert len(long['comparisons']) == 6
    print('Both archived campaigns passed their original accounting checks.')
