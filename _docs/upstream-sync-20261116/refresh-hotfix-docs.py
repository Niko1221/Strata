"""Refresh current official provenance while retaining all historical release records."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = '09b603886925994d0d4461857b01321ef3aa431c'
NEW = 'baf16363b4403ef0ed4f538f5d5ed11c43fb9535'
for name in ['README.md', 'README.de.md', 'README.es.md', 'README.fr.md', 'README.ja.md',
             'README.pl.md', 'README.pt-BR.md', 'README.zh-CN.md',
             '_docs/upstream-sync-20261116/rewrite-readmes.py']:
    path = ROOT / name
    text = path.read_text(encoding='utf-8')
    assert OLD in text
    path.write_text(text.replace(OLD, NEW).replace('09b6038869', 'baf16363b4'), encoding='utf-8', newline='\n')
path = ROOT / 'AGENTS.md'
text = path.read_text(encoding='utf-8')
assert OLD in text
path.write_text(text.replace(OLD, NEW), encoding='utf-8', newline='\n')
print('Updated eight current README references, their generator, and current AGENTS source pin.')
