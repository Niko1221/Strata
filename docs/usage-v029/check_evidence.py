"""Check the published aggregate snapshot through Usage without touching production.
Run from the checkout root: python docs/usage-v029/check_evidence.py
"""
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.usage import Usage

evidence = json.loads(Path(__file__).with_name('production-counters.json').read_text())
life, imported = evidence['lifetime'], evidence['imported']
assert life['output_tokens'] + imported['output_tokens'] == 22020156
assert life['requests'] + imported['requests'] == 23837
assert life['prompt_tokens'] - life['reused'] == 25040612
with tempfile.TemporaryDirectory() as folder:
    p = Path(folder) / 'sanitized.json'
    p.write_text(json.dumps({'since': evidence['tracked_since'], 'lifetime': life, 'imported': imported}))
    for _ in range(2):
        u = Usage(p)
        try:
            assert u.enabled and not u.persistence_error
            assert u.lifetime == life and u.imported == imported
            u.flush()
            assert not u.persistence_error
        finally:
            u.close()
print(json.dumps({'output_total': 22020156, 'requests_total': 23837,
                  'newly_read_prompt_tokens': 25040612, 'aggregate_roundtrip_preserved': True,
                  'date_buckets_included': False}))
